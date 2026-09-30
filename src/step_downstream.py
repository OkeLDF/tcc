
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassRecall

from model import LoRAViTModel, LoRAViTClassifier

import downstream

from concat_datasets import (BETHESDA_CLASSES, BETHESDA_LABEL_MAPS, MORPHOLOGICAL_CLASSES,
    MORPHOLOGICAL_LABEL_MAPS, BTMDataset, CPSMI2025Dataset, HerlevDataset, HiCervixDataset, PapicitoDataset, MendeleyLBCDataset,
    SIPaKMeDDataset, build_downstream_dataset)

from image_transforms import EvaluationTransform
from utils import save_checkpoint, load_checkpoint, epoch_stats, EarlyStopping
from log import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# def _set_trainable(module, enabled):
#     for parameter in module.parameters():
#         parameter.requires_grad = enabled

def _set_trainable_lora(peft_model, enabled):
    for name, parameter in peft_model.named_parameters():
        if 'lora_' in name:
            parameter.requires_grad = enabled

def _datasets(schema, data_root, configs, transform, test_size, validation_size):
    if schema == 'bethesda':
        classes, label_maps = BETHESDA_CLASSES, BETHESDA_LABEL_MAPS
        dataset_types = [MendeleyLBCDataset, CPSMI2025Dataset, BTMDataset, HiCervixDataset, PapicitoDataset]
        paths = [
            data_root / configs['PATH_MENDELEYLBC'],
            data_root / configs['PATH_CPSMI2025'],
            data_root / configs['PATH_BTM'],
            data_root / configs['PATH_HICERVIX'],
            data_root / configs['PATH_PAPICITO'],
        ]
    elif schema == 'morphological':
        classes, label_maps = MORPHOLOGICAL_CLASSES, MORPHOLOGICAL_LABEL_MAPS
        dataset_types = [HerlevDataset, SIPaKMeDDataset, HiCervixDataset, PapicitoDataset]
        paths = [
            data_root / configs['PATH_HERLEV'],
            data_root / configs['PATH_SIPAKMED'],
            data_root / configs['PATH_HICERVIX'],
            data_root / configs['PATH_PAPICITO'],
        ]
    else:
        raise ValueError(f'Unknown schema: {schema!r}')

    def build(split):
        instances = [
            dataset_type(
                path,
                task='downstream',
                split=split,
                transform=transform,
                test_size=test_size,
                validation_size=validation_size,
                **({'label_level': 2 if schema == 'bethesda' else 1}
                   if dataset_type is HiCervixDataset else {}),
            )
            for dataset_type, path in zip(dataset_types, paths)
        ]
        return build_downstream_dataset(instances, label_maps, classes)

    return build('train'), build('valid'), build('test'), classes


def main(schema='bethesda'):
    configs = yaml.safe_load((Path(__file__).with_name('configs.yaml')).read_text())
    data_root = PROJECT_ROOT / configs['PROJECT_DATA']
    log_root = PROJECT_ROOT / configs['PROJECT_LOG']
    log_root.mkdir(parents=True, exist_ok=True)
    from_pretrained = str(configs['DOWNSTREAM']['FROM_PRETRAINED'])
    num_workers = int(configs['DATA']['NUM_WORKERS'])
    batch_size = int(configs['DOWNSTREAM']['BATCH_SIZE'])
    save_every = int(configs['DOWNSTREAM']['SAVE_EVERY'])
    output_dir = PROJECT_ROOT / configs['PATH_FINETUNED'] / schema
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, classes = _datasets(
        schema, data_root, configs, EvaluationTransform(),
        float(configs['DATA']['TEST_SIZE']), float(configs['PRETRAINING']['VALIDATION_SIZE'])
    )

    loader_options = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running downstream on {device}')
    
    lora_args = configs['PRETRAINING']['LORA']
    r       = lora_args.get('R', 8)
    alpha   = lora_args.get('ALPHA', 16)
    dropout = lora_args.get('DROPOUT', 0.0)

    if from_pretrained == 'local':
        checkpoint_path = PROJECT_ROOT / configs['PATH_PRETRAINED'] / 'vit_pretrained_encoder.pt'
    
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Pretrained encoder not found: {checkpoint_path}')
    
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        lora_vit = LoRAViTModel(r=r, alpha=alpha, dropout=dropout, device=device)
        lora_vit.load_state_dict(checkpoint['model_state'])
    
        classifier = LoRAViTClassifier(lora_vit, classes, device=device)
    
    elif from_pretrained == 'base':
        base_vit = LoRAViTModel(r=r, alpha=alpha, dropout=dropout, device=device)
        
        classifier = LoRAViTClassifier(base_vit, classes, device=device)

    else:
        raise ValueError(f"Invalid `from_pretrained`. Should be 'local' or 'base', but got: {from_pretrained!r}")

    criterion = nn.CrossEntropyLoss()

    phases = [
        ('frozen', int(configs['DOWNSTREAM']['FROZEN_EPOCHS']), float(configs['DOWNSTREAM']['FROZEN_LR']), False),
        ('unfrozen', int(configs['DOWNSTREAM']['UNFROZEN_EPOCHS']), float(configs['DOWNSTREAM']['UNFROZEN_LR']), True),
    ]

    history = []
    history_path = log_root / f'downstream_history_{schema}_{from_pretrained}.csv'
    test_results_path = log_root / f'downstream_test_{schema}_{from_pretrained}.csv'
    run_dir = output_dir / from_pretrained

    metrics = MetricCollection({
        'auroc': MulticlassAUROC(num_classes=len(classes)),
        'average_precision': MulticlassAveragePrecision(num_classes=len(classes)),
        'recall': MulticlassRecall(num_classes=len(classes), average='macro'),
    }).to(device)

    for phase_idx, (phase, epochs, learning_rate, encoder_trainable) in enumerate(phases):
        if epochs == 0:
            continue

        early_stopping = EarlyStopping(patience=5)
        best_score = float('inf')

        _set_trainable_lora(classifier.encoder, encoder_trainable)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate)

        checkpoint_args = dict(
            model=classifier,
            optimizer=optimizer,
            scheduler=None,
            phase=phase_idx,
        )
        
        for epoch in range(epochs):
            train_loss = downstream.train_step(classifier, train_loader, criterion, optimizer, device=device)
            metrics.reset()
            eval_loss = downstream.eval_step(classifier, valid_loader, criterion, metrics=metrics, device=device)

            result = {
                'epoch': epoch,
                'phase': phase,
                'train_loss': train_loss,
                'eval_loss': eval_loss
            }
            result.update({name: float(value) for name, value in metrics.compute().items()})
            epoch_stats(epoch, result)
            history.append(result)

            if eval_loss < best_score:
                best_score = eval_loss
                save_checkpoint(
                    **checkpoint_args,
                    epoch=epoch,
                    metrics=result,
                    path=run_dir / f'vit_classifier_{phase}.pt'
                )
                

            if save_every != 0 and epoch % save_every == 0:
                pd.DataFrame(history).to_csv(history_path, index=False)
                save_checkpoint(
                    **checkpoint_args,
                    epoch=epoch,
                    metrics=result,
                    path=run_dir / 'checkpoint' / 'last_vit_classifier.pt'
                )

            if early_stopping.step(eval_loss):
                logger.warning(f'EarlyStopping stopped execution at epoch {epoch} in {phase} phase')
                break

    pd.DataFrame(history).to_csv(history_path, index=False)

    # The test set is used only here, once per phase, on the checkpoint that
    # had the lowest validation loss in that phase.
    test_results = []
    for phase, epochs, _, _ in phases:
        best_path = run_dir / f'vit_classifier_{phase}.pt'
        if epochs == 0 or not best_path.exists():
            continue

        checkpoint = load_checkpoint(classifier, best_path, device)
        metrics.reset()
        test_loss = downstream.eval_step(classifier, test_loader, criterion, metrics=metrics, device=device)

        result = {'phase': phase, 'best_epoch': checkpoint['epoch'], 'test_loss': test_loss}
        result.update({name: float(value) for name, value in metrics.compute().items()})
        logger.info(f'[test] {result}')
        test_results.append(result)

    pd.DataFrame(test_results).to_csv(test_results_path, index=False)

if __name__ == '__main__':
    main()
