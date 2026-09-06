from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassRecall
from transformers import ViTConfig, ViTForImageClassification, ViTModel

import downstream

from concat_datasets import (BETHESDA_CLASSES, BETHESDA_LABEL_MAPS, MORPHOLOGICAL_CLASSES,
    MORPHOLOGICAL_LABEL_MAPS, BTMDataset, CPSMI2025Dataset, HerlevDataset, HiCervixDataset, MendeleyLBCDataset,
    SIPaKMeDDataset, build_downstream_dataset)

from image_transforms import EvaluationTransform
from utils import save_checkpoint, epoch_stats, EarlyStopping
from log import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _set_trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _datasets(schema, data_root, configs, transform, test_size):
    if schema == 'bethesda':
        classes, label_maps = BETHESDA_CLASSES, BETHESDA_LABEL_MAPS
        dataset_types = [MendeleyLBCDataset, CPSMI2025Dataset, BTMDataset, HiCervixDataset]
        paths = [
            data_root / configs['PATH_MENDELEYLBC'],
            data_root / configs['PATH_CPSMI2025'],
            data_root / 'BTM',
            data_root / configs['PATH_HICERVIX'],
        ]
    elif schema == 'morphological':
        classes, label_maps = MORPHOLOGICAL_CLASSES, MORPHOLOGICAL_LABEL_MAPS
        dataset_types = [HerlevDataset, SIPaKMeDDataset, HiCervixDataset]
        paths = [
            data_root / configs['PATH_HERLEV'],
            data_root / configs['PATH_SIPAKMED'],
            data_root / configs['PATH_HICERVIX'],
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
                **({'label_level': 2 if schema == 'bethesda' else 1}
                   if dataset_type is HiCervixDataset else {}),
            )
            for dataset_type, path in zip(dataset_types, paths)
        ]
        return build_downstream_dataset(instances, label_maps, classes)

    return build('train'), build('test'), classes


def main(schema='bethesda'):
    configs = yaml.safe_load((Path(__file__).with_name('configs.yaml')).read_text())
    data_root = PROJECT_ROOT / configs['PROJECT_DATA']
    log_root = PROJECT_ROOT / configs['PROJECT_LOG']
    log_root.mkdir(parents=True, exist_ok=True)
    num_workers = int(configs['DATA']['NUM_WORKERS'])
    batch_size = int(configs['DOWNSTREAM']['BATCH_SIZE'])
    save_every = int(configs['DOWNSTREAM']['SAVE_EVERY'])
    output_dir = PROJECT_ROOT / configs['PATH_FINETUNED'] / schema
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, classes = _datasets(
        schema, data_root, configs, EvaluationTransform(), float(configs['DATA']['TEST_SIZE'])
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running downstream on {device}')

    checkpoint_path = PROJECT_ROOT / configs['PATH_PRETRAINED'] / 'vit_pretrained_encoder.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Pretrained encoder not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
    encoder.load_state_dict(checkpoint['model_state'])

    classifier = ViTForImageClassification(
        ViTConfig.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=len(classes),
            id2label=dict(enumerate(classes)),
            label2id={label: index for index, label in enumerate(classes)},
        )
    )
    classifier.vit.load_state_dict(encoder.state_dict())
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()

    phases = [
        ('frozen', int(configs['DOWNSTREAM']['FROZEN_EPOCHS']), float(configs['DOWNSTREAM']['FROZEN_LR']), False),
        ('unfrozen', int(configs['DOWNSTREAM']['UNFROZEN_EPOCHS']), float(configs['DOWNSTREAM']['UNFROZEN_LR']), True),
    ]

    history = []

    metrics = MetricCollection({
        'auroc': MulticlassAUROC(num_classes=len(classes)),
        'average_precision': MulticlassAveragePrecision(num_classes=len(classes)),
        'recall': MulticlassRecall(num_classes=len(classes), average='macro'),
    }).to(device)

    early_stopping = EarlyStopping(patience=5)

    stop_training = False
    for phase, epochs, learning_rate, encoder_trainable in phases:
        if epochs == 0:
            continue

        _set_trainable(classifier.vit, encoder_trainable)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = downstream.train_step(classifier, train_loader, criterion, optimizer, device=device)
            metrics.reset()
            eval_loss = downstream.eval_step(classifier, test_loader, criterion, metrics=metrics, device=device)

            result = {
                'epoch': epoch,
                'phase': phase,
                'train_loss': train_loss,
                'eval_loss': eval_loss
            }
            result.update({name: float(value) for name, value in metrics.compute().items()})
            epoch_stats(epoch, result)
            history.append(result)

            if save_every != 0 and epoch % save_every == 0:
                pd.DataFrame(history).to_csv(log_root / 'downstream_history.csv', index=False)
                save_checkpoint(
                    classifier, optimizer, None, epoch, result, output_dir / 'checkpoint' / 'last_vit_classifier.pt'
                )

            if early_stopping.step(eval_loss):
                logger.warning(f'EarlyStopping stopped execution at epoch {epoch} in {phase} phase')
                stop_training = True
                break

        if stop_training:
            break

    pd.DataFrame(history).to_csv(log_root / 'downstream_history.csv', index=False)
    save_checkpoint(classifier, optimizer, None, sum(phase[1] for phase in phases), result, output_dir / 'vit_classifier.pt')

if __name__ == '__main__':
    main()
