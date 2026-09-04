# # Setup

import yaml

import torch
import torch.nn as nn

import pretraining

from log import logger  # local log.py with global logger
from utils import (
    print_message,
    time_diff,
    epoch_stats,
    save_checkpoint,
    EarlyStopping
)

from torchmetrics import MetricCollection

from transformers import (
    ViTModel
)
from concat_datasets import *
from pathlib import Path

configs = yaml.safe_load(open('configs.yaml'))

PROJECT_ROOT_FROM_HOME = Path.home() / configs.get('PROJECT_ROOT_FROM_HOME')

if not PROJECT_ROOT_FROM_HOME.exists():
    PROJECT_ROOT_FROM_HOME = Path.home() / 'git/tcc'

if not PROJECT_ROOT_FROM_HOME.exists():
    PROJECT_ROOT_FROM_HOME = Path('/mnt/data/home/otavio/UEPG/tcc')
   
PROJECT_DATA     = PROJECT_ROOT_FROM_HOME / configs['PROJECT_DATA']
PATH_CPSMI2025   = PROJECT_DATA / configs['PATH_CPSMI2025']
PATH_HERLEV      = PROJECT_DATA / configs['PATH_HERLEV']
PATH_MENDELEYLBC = PROJECT_DATA / configs['PATH_MENDELEYLBC']
PATH_SIPAKMED    = PROJECT_DATA / configs['PATH_SIPAKMED']

PATH_PRETRAINED = PROJECT_ROOT_FROM_HOME / configs['PATH_PRETRAINED']

FROZEN_EPOCHS = configs['PRETRAINING']['FROZEN_EPOCHS']
UNFROZEN_EPOCHS = configs['PRETRAINING']['UNFROZEN_EPOCHS']

FROZEN_LR = float(configs['PRETRAINING']['FROZEN_LR'])
UNFROZEN_LR = float(configs['PRETRAINING']['UNFROZEN_LR'])

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


# # DataLoaders Config

pretraining_train_loader = None
pretraining_valid_loader = None


# # Pretraining

# ## Config

encoder = ViTModel.from_pretrained(
    'google/vit-base-patch16-224',
    add_pooling_layer=False
)

projector = nn.Sequential(
    nn.Linear(768, 768),
    nn.ReLU(),
    nn.Linear(768, 128)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW([
    {'params': encoder.parameters(), 'lr': UNFROZEN_LR},
    {'params': projector.parameters(), 'lr': UNFROZEN_LR}
])

metrics = None

scheduler = None

early_stopping = EarlyStopping(patience=5)


# ## Training

for epoch in range(FROZEN_EPOCHS):
    train_loss = pretraining.train_step(
        encoder=encoder,
        projector=projector,
        loader=pretraining_train_loader,
        contrastive_loss=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        accumulation_steps=accumulation_steps,
        autocast_dtype=autocast_dtype)

    metrics.reset()
    
    valid_loss = pretraining.eval_step(
        encoder=encoder,
        projector=projector,
        loader=pretraining_valid_loader,
        contrastive_loss=criterion,
        metrics=metrics,
        device=device,
        autocast_dtype=autocast_dtype)

    computed = metrics.compute()
    

    if early_stopper.step(monitored):
        print(f"\nEarly stopping at epoch {epoch}.")
        break