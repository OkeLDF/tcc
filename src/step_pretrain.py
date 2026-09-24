from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from model import LoRAViTModel

import pretraining

from concat_datasets import (
    BTMDataset,
    CPSMI2025Dataset,
    HerlevDataset,
    HiCervixDataset,
    MendeleyLBCDataset,
    SIPaKMeDDataset,
    PapicitoDataset,
    build_pretraining_dataset,
)
from image_transforms import SimCLRTransform
from log import logger
from utils import EarlyStopping, epoch_stats, save_checkpoint, _mem


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _set_trainable_lora(peft_model, enabled):
    for name, parameter in peft_model.named_parameters():
        if 'lora_' in name:
            parameter.requires_grad = enabled


def _build_dataset(split, data_root, configs, transform, test_size, validation_size):
    return build_pretraining_dataset([
        CPSMI2025Dataset(
            data_root / configs['PATH_CPSMI2025'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        HerlevDataset(
            data_root / configs['PATH_HERLEV'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        MendeleyLBCDataset(
            data_root / configs['PATH_MENDELEYLBC'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        SIPaKMeDDataset(
            data_root / configs['PATH_SIPAKMED'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        BTMDataset(
            data_root / configs['PATH_BTM'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        HiCervixDataset(
            data_root / configs['PATH_HICERVIX'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        ),
        PapicitoDataset(
            data_root / configs['PATH_PAPICITO'], task='pretraining', split=split,
            transform=transform, test_size=test_size, validation_size=validation_size,
        )
    ])


def main():
    configs = yaml.safe_load((Path(__file__).with_name('configs.yaml')).read_text())
    data_root = PROJECT_ROOT / configs['PROJECT_DATA']
    log_root = PROJECT_ROOT / configs['PROJECT_LOG']
    log_root.mkdir(parents=True, exist_ok=True)

    test_size = float(configs['DATA']['TEST_SIZE'])
    validation_size = float(configs['PRETRAINING']['VALIDATION_SIZE'])
    num_workers = int(configs['DATA']['NUM_WORKERS'])
    batch_size = int(configs['PRETRAINING']['BATCH_SIZE'])
    save_every = int(configs['PRETRAINING']['SAVE_EVERY'])

    transform = SimCLRTransform()
    train_dataset = _build_dataset('train', data_root, configs, transform, test_size, validation_size)
    valid_dataset = _build_dataset('valid', data_root, configs, transform, test_size, validation_size)

    loader_options = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_options)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **loader_options)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running pretraining on {device}')

    lora_args = configs['PRETRAINING']['LORA']
    r       = lora_args.get('R', 8)
    alpha   = lora_args.get('ALPHA', 16)
    dropout = lora_args.get('DROPOUT', 0.0)
    
    lora_vit = LoRAViTModel(r=r, alpha=alpha, dropout=dropout, device=device)
    
    criterion = pretraining.NTXentLoss(float(configs['PRETRAINING']['TEMPERATURE']))

    output_dir = PROJECT_ROOT / configs['PATH_PRETRAINED']
    checkpoint_dir = output_dir / 'checkpoint'
    history = []
    phases = [
        ('frozen', int(configs['PRETRAINING']['FROZEN_EPOCHS']), float(configs['PRETRAINING']['FROZEN_LR']), False),
        ('unfrozen', int(configs['PRETRAINING']['UNFROZEN_EPOCHS']), float(configs['PRETRAINING']['UNFROZEN_LR']), True),
    ]
    
    resume_path = checkpoint_dir / 'last_vit_pretrained_encoder.pt'

    resume_phase, resume_epoch, resume_optimizer_state = 0, 0, None

    if configs['PRETRAINING'].get('RESUME', False) and resume_path.exists():
        ckpt = load_checkpoint(lora_vit, resume_path, device)
        resume_phase = ckpt['phase']
        resume_epoch = ckpt['epoch'] + 1
        resume_optimizer_state = ckpt['optimizer_state']

        # If the saved epoch was the last of its phase, start the next phase
        if resume_epoch >= phases[resume_phase][1]:
            resume_phase += 1
            resume_epoch = 0
            resume_optimizer_state = None         # new phase, fresh optimizer

        # Keep the previous history instead of overwriting the CSV
        history_path = log_root / 'pretraining_history.csv'
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict('records')
    
    autocast_dtype = torch.bfloat16
    scaler = None

    stop_training = False
    completed_epochs = 0
    for phase_idx, (phase, epochs, learning_rate, encoder_trainable) in enumerate(phases):
        if epochs == 0 or phase_idx < resume_phase:
            continue

        _set_trainable_lora(lora_vit.encoder, encoder_trainable)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, lora_vit.parameters()),
            lr=learning_rate,
        )
        
        early_stopping = EarlyStopping(patience=5)
        best_score = float('inf')

        checkpoint_args = dict(
            model=lora_vit,
            optimizer=optimizer,
            scheduler=None,
            phase=phase_idx,
        )
        
        first_epoch = 0
        if phase_idx == resume_phase:
            first_epoch = resume_epoch
            if resume_optimizer_state is not None:
                optimizer.load_state_dict(resume_optimizer_state)

        for epoch in range(first_epoch, epochs):
            _mem(f"epoch {epoch} start")
            train_loss = pretraining.train_step(
                lora_vit, train_loader, criterion, optimizer, autocast_dtype=autocast_dtype, scaler=scaler, device=device)
            eval_loss = pretraining.eval_step(lora_vit, valid_loader, criterion, device=device)
            completed_epochs += 1
            
            result = {
                'epoch': epoch,
                'phase': phase,
                'phase_idx': phase_idx,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
            }
            epoch_stats(epoch, result)
            history.append(result)

            if save_every != 0 and epoch % save_every == 0:
                pd.DataFrame(history).to_csv(log_root / 'pretraining_history.csv', index=False)
                save_checkpoint(
                    **checkpoint_args,
                    epoch=epoch,
                    metrics=result,
                    path=checkpoint_dir / 'last_vit_pretrained_encoder.pt'
                )
            
            if eval_loss < best_score:
                best_score = eval_loss
                save_checkpoint(
                    **checkpoint_args,
                    epoch=epoch,
                    metrics=result,
                    path=checkpoint_dir / 'vit_pretrained_encoder.pt'
                )

            if early_stopping.step(eval_loss):
                logger.warning(f'EarlyStopping stopped execution at epoch {epoch} in {phase} phase')
                stop_training = True
                break
                
            _mem(f"epoch {epoch} end")
            torch.cuda.empty_cache()
            _mem(f"epoch {epoch} end (post empty_cache)")

    pd.DataFrame(history).to_csv(log_root / 'pretraining_history.csv', index=False)


if __name__ == '__main__':
    main()
