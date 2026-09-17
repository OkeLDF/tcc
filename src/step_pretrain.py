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
from utils import EarlyStopping, epoch_stats, save_checkpoint


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
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
    early_stopping = EarlyStopping(patience=5)
    phases = [
        ('frozen', int(configs['PRETRAINING']['FROZEN_EPOCHS']), float(configs['PRETRAINING']['FROZEN_LR']), False),
        ('unfrozen', int(configs['PRETRAINING']['UNFROZEN_EPOCHS']), float(configs['PRETRAINING']['UNFROZEN_LR']), True),
    ]
    
    autocast_dtype = torch.bfloat16
    scaler = None

    stop_training = False
    completed_epochs = 0
    for phase, epochs, learning_rate, encoder_trainable in phases:
        if epochs == 0:
            continue

        _set_trainable_lora(lora_vit.encoder, encoder_trainable)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, lora_vit.parameters()),
            lr=learning_rate,
        )

        for epoch in range(epochs):
            train_loss = pretraining.train_step(
                lora_vit, train_loader, criterion, optimizer, autocast_dtype=autocast_dtype, scaler=scaler, device=device)
            eval_loss = pretraining.eval_step(lora_vit, valid_loader, criterion, device=device)

            result = {
                'epoch': epoch,
                'phase': phase,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
            }
            epoch_stats(epoch, result)
            history.append(result)

            if save_every != 0 and epoch % save_every == 0:
                pd.DataFrame(history).to_csv(log_root / 'pretraining_history.csv', index=False)
                save_checkpoint(lora_vit, optimizer, None, epoch, result, checkpoint_dir / 'last_vit_pretrained_encoder.pt')

            if early_stopping.step(eval_loss):
                logger.warning(f'EarlyStopping stopped execution at epoch {epoch} in {phase} phase')
                stop_training = True
                break

        if stop_training:
            break

    pd.DataFrame(history).to_csv(log_root / 'pretraining_history.csv', index=False)
    save_checkpoint(lora_vit, optimizer, None, completed_epochs, history[-1] if history else {}, output_dir / 'vit_pretrained_encoder.pt')


if __name__ == '__main__':
    main()
