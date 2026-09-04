from concat_datasets import *
from pathlib import Path

import data
from image_transforms import EvaluationTransform, SimCLRTransform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA = PROJECT_ROOT / 'data'
PATH_CPSMI2025   = PROJECT_DATA / 'CPSMI2025/CPSMI2025'
PATH_HERLEV      = PROJECT_DATA / 'Herlev Dataset'
PATH_MENDELEYLBC = PROJECT_DATA / 'MendeleyLBC'
PATH_SIPAKMED    = PROJECT_DATA / 'SIPaKMeD'
PATH_BTM         = PROJECT_DATA / 'BTM'

split='train'
task='downstream'
eval_transform = EvaluationTransform()

print('Project root path:', str(PROJECT_ROOT), end='\n\n')

print('CPSMI2025Dataset:')
cpsmi2025 = data.CPSMI2025Dataset(PATH_CPSMI2025, split=split, task=task, transform=eval_transform)
print('  OK')

print('HerlevDataset:')
herlev = data.HerlevDataset(PATH_HERLEV, split=split, task=task, transform=eval_transform)
print('  OK')

print('MendeleyLBCDataset:')
mendeleylbc = data.MendeleyLBCDataset(PATH_MENDELEYLBC, split=split, task=task, transform=eval_transform)
print('  OK')

print('SIPaKMeDDataset:')
sipakmed = data.SIPaKMeDDataset(PATH_SIPAKMED, split=split, task=task, transform=eval_transform)
print('  OK')

print('BTMDataset:')
btm = data.BTMDataset(PATH_BTM, split=split, task=task, transform=eval_transform)
print('  OK')


print(f'{len(cpsmi2025)   =}')
print(f'{len(herlev)      =}')
print(f'{len(mendeleylbc) =}')
print(f'{len(sipakmed)    =}')
print(f'{len(btm)         =}')



pretraining_ds = build_pretraining_dataset([
    CPSMI2025Dataset(PATH_CPSMI2025, task='pretraining', split='train', transform=SimCLRTransform()),
    HerlevDataset(PATH_HERLEV, task='pretraining', split='train', transform=SimCLRTransform()),
    MendeleyLBCDataset(PATH_MENDELEYLBC, task='pretraining', split='train', transform=SimCLRTransform()),
    SIPaKMeDDataset(PATH_SIPAKMED, task='pretraining', split='train', transform=SimCLRTransform()),
    BTMDataset(PATH_BTM, task='pretraining', split='train', transform=SimCLRTransform()),
])


bethesda_train = build_downstream_dataset(
    [
        MendeleyLBCDataset(PATH_MENDELEYLBC, task='downstream', split='train', transform=eval_transform),
        CPSMI2025Dataset(PATH_CPSMI2025, task='downstream', split='train', transform=eval_transform),
        BTMDataset(PATH_BTM, task='downstream', split='train', transform=eval_transform),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)
bethesda_test = build_downstream_dataset(
    [
        MendeleyLBCDataset(PATH_MENDELEYLBC, task='downstream', split='test', transform=eval_transform),
        CPSMI2025Dataset(PATH_CPSMI2025, task='downstream', split='test', transform=eval_transform),
        BTMDataset(PATH_BTM, task='downstream', split='test', transform=eval_transform),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)


morphological_train = build_downstream_dataset(
    [
        HerlevDataset(PATH_HERLEV, task='downstream', split='train', transform=eval_transform),
        SIPaKMeDDataset(PATH_SIPAKMED, task='downstream', split='train', transform=eval_transform),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)
morphological_test = build_downstream_dataset(
    [
        HerlevDataset(PATH_HERLEV, task='downstream', split='test', transform=eval_transform),
        SIPaKMeDDataset(PATH_SIPAKMED, task='downstream', split='test', transform=eval_transform),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)
