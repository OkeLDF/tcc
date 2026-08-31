# # Setup

import glob
import yaml

import torch
import torch.nn as nn

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
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassRecall
)
from transformers import (
    ViTForImageClassification,
    ViTConfig,
    ViTModel
)
from concat_datasets import *
from pathlib import Path

configs = yaml.safe_load(open('configs.yaml'))

PROJECT_ROOT_FROM_HOME = Path.home() / configs.get('PROJECT_ROOT_FROM_HOME')
if not PROJECT_ROOT_FROM_HOME.exists():
    PROJECT_ROOT_FROM_HOME = Path.home() / 'tcc'
   
PROJECT_DATA = configs['PROJECT_DATA']
PATH_CPSMI2025 = configs['PATH_CPSMI2025']
PATH_HERLEV = configs['PATH_HERLEV']
PATH_MENDELEYLBC = configs['PATH_MENDELEYLBC']
PATH_SIPAKMED = configs['PATH_SIPAKMED']

PATH_PRETRAINED = configs['PATH_PRETRAINED']
PATH_FINETUNED = configs['PATH_FINETUNED']

NUM_CLASSES = configs['DOWNSTREAM']['NUM_CLASSES']
EPOCHS = configs['DOWNSTREAM']['EPOCHS']
LR = configs['DOWNSTREAM']['LR']

device = 'cuda' if torch.cuda.is_available else 'cpu'
autocast_dtype = None
accumulation_steps = 1

PRETRAINED_MODEL = configs.get('PRETRAINED_MODEL', None)

if PRETRAINED_MODEL is None:
    candidates = glob.glob(PATH_PRETRAINED / '*.pt')
    if len(candidates) == 0:
        raise ValueError(f"No pretrained model found on '{str(PATH_PRETRAINED)}'")
    PRETRAINED_MODEL = candidates[0]


# # Datasets

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


# # DataLoaders Config

downstream_train_loader = None
downstream_valid_loader = None


# # Downstream

# ## Config

pretrained_encoder = ViTModel.from_pretrained(PRETRAINED_MODEL)

clf_config = ViTConfig.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=NUM_CLASSES,
)
classifier = ViTForImageClassification(clf_config)
classifier.vit.load_state_dict(pretrained_encoder.state_dict())

downstream_criterion = nn.CrossEntropyLoss()

downstream_optimizer = torch.optim.Adamw(
    {'params': classifier.parameters(), 'lr': LR}
)

downstream_metrics = MetricCollection(
    MulticlassAUROC(num_classes=NUM_CLASSES),
    MulticlassAveragePrecision(num_classes=NUM_CLASSES),
    MulticlassRecall(num_classes=NUM_CLASSES),
)

downstream_scheduler = None


# ## Training

for epoch in range(EPOCHS):
    train_loss = downstream.train_step(
        model=classifier,
        loader=downstream_train_loader,
        criterion=downstream_criterion,
        optimizer=downstream_optimizer,
        scheduler=downstream_scheduler,
        device=device,
        accumulation_steps=accumulation_steps,
        autocast_dtype=autocast_dtype)
    
    valid_loss = downstream.eval_step(
        model=classifier,
        loader=downstream_valid_loader,
        criterion=downstream_criterion,
        metrics=downstream_metrics,
        device=device,
        autocast_dtype=autocast_dtype)

