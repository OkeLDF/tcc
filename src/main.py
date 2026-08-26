# # Setup

import torch
import torch.nn as nn

import pretraining
import downstream

from log import logger  # local log.py with global logger
from utils import (
    print_message,
    time_diff,
    epoch_stats,
    save_checkpoint,
    EarlyStopping
)

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassRecall
from transformers import ViTForImageClassification
from concat_datasets import *
from pathlib import Path


PROJECT_ROOT = Path.home() / 'git/tcc'
PROJECT_DATA = PROJECT_ROOT / 'data'
PATH_CPSMI2025   = PROJECT_DATA / 'CPSMI2025/CPSMI2025'
PATH_HERLEV      = PROJECT_DATA / 'Herlev Dataset'
PATH_MENDELEYLBC = PROJECT_DATA / 'MendeleyLBC'
PATH_SIPAKMED    = PROJECT_DATA / 'SIPaKMeD'

PRETRAINING_EPOCHS = 3 # 50
DOWNSTREAM_EPOCHS  = 3 # 50

LR = 1e-4
NUM_CLASSES = 4

device = 'cuda' if torch.cuda.is_available else 'cpu'
autocast_dtype = None
accumulation_steps = 1


# # Datasets

pretraining_ds = build_pretraining_dataset([
    CPSMI2025Dataset(PATH_CPSMI2025, task='pretraining', split='train',),
    HerlevDataset(PATH_HERLEV, task='pretraining', split='train',),
    MendeleyLBCDataset(PATH_MENDELEYLBC, task='pretraining', split='train',),
    SIPaKMeDDataset(PATH_SIPAKMED, task='pretraining', split='train',),
])

bethesda_train = build_downstream_dataset(
    [
        MendeleyLBCDataset(PATH_MENDELEYLBC, task='downstream', split='train'),
        CPSMI2025Dataset(PATH_CPSMI2025, task='downstream', split='train'),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)
bethesda_test = build_downstream_dataset(
    [
        MendeleyLBCDataset(PATH_MENDELEYLBC, task='downstream', split='test'),
        CPSMI2025Dataset(PATH_CPSMI2025, task='downstream', split='test'),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)

morphological_train = build_downstream_dataset(
    [
        HerlevDataset(PATH_HERLEV, task='downstream', split='train'),
        SIPaKMeDDataset(PATH_SIPAKMED, task='downstream', split='train'),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)
morphological_test = build_downstream_dataset(
    [
        HerlevDataset(PATH_HERLEV, task='downstream', split='test'),
        SIPaKMeDDataset(PATH_SIPAKMED, task='downstream', split='test'),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)


# # Model

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)


# # Training Config

# ## DataLoaders

pretraining_train_loader = None
pretraining_valid_loader = None
downstream_train_loader = None
downstream_valid_loader = None


# ## Pretraining Config

pretraining_args = dict(
    criterion = nn.CrossEntropyLoss(),
    optimizer = torch.optim.Adamw(
        {'params': model.parameters(), 'lr': LR}
    ),
    metrics = MetricCollection(
        MulticlassAUROC(num_classes=NUM_CLASSES),
        MulticlassAveragePrecision(num_classes=NUM_CLASSES),
        MulticlassRecall(num_classes=NUM_CLASSES),
    )
)


# ## Downstream Config

downstream_criterion = nn.CrossEntropyLoss()

downstream_optimizer = torch.optim.Adamw(
    {'params': model.parameters(), 'lr': LR}
)

downstream_metrics = MetricCollection(
    MulticlassAUROC(num_classes=NUM_CLASSES),
    MulticlassAveragePrecision(num_classes=NUM_CLASSES),
    MulticlassRecall(num_classes=NUM_CLASSES),
)

downstream_scheduler = None


# # Training

# ## Pretraining

for epoch in range(PRETRAINING_EPOCHS):
    train_loss = pretraining.train_step(
        model=model,
        loader=pretraining_train_loader,
        criterion=downstream_criterion,
        optimizer=downstream_optimizer,
        scheduler=downstream_scheduler,
        device=device,
        accumulation_steps=accumulation_steps,
        autocast_dtype=autocast_dtype)
    
    valid_loss = pretraining.eval_step(
        model=model,
        loader=pretraining_valid_loader,
        criterion=downstream_criterion,
        metrics=downstream_metrics,
        device=device,
        autocast_dtype=autocast_dtype)


# ## Downstream

for epoch in range(DOWNSTREAM_EPOCHS):
    train_loss = downstream.train_step(model, downstream_train_loader)
    valid_loss = downstream.eval_step(model, downstream_valid_loader)

