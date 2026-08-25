from concat_datasets import *
from pathlib import Path

import data

PROJECT_ROOT = Path.home() / 'git/tcc'
PROJECT_DATA = PROJECT_ROOT / 'data'
PATH_CPSMI2025   = PROJECT_DATA / 'CPSMI2025/CPSMI2025'
PATH_HERLEV      = PROJECT_DATA / 'Herlev Dataset'
PATH_MENDELEYLBC = PROJECT_DATA / 'MendeleyLBC'
PATH_SIPAKMED    = PROJECT_DATA / 'SIPaKMeD'

split='train'
task='downstream'

print('Project root path:', str(PROJECT_ROOT), end='\n\n')

print('CPSMI2025Dataset:')
cpsmi2025 = data.CPSMI2025Dataset(PATH_CPSMI2025, split=split, task=task)
print('  OK')

print('HerlevDataset:')
herlev = data.HerlevDataset(PATH_HERLEV, split=split, task=task)
print('  OK')

print('MendeleyLBCDataset:')
mendeleylbc = data.MendeleyLBCDataset(PATH_MENDELEYLBC, split=split, task=task)
print('  OK')

print('SIPaKMeDDataset:')
sipakmed = data.SIPaKMeDDataset(PATH_SIPAKMED, split=split, task=task)
print('  OK')


print(f'{len(cpsmi2025)   =}')
print(f'{len(herlev)      =}')
print(f'{len(mendeleylbc) =}')
print(f'{len(sipakmed)    =}')



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