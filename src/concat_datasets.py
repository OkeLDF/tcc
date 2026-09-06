"""
Concatenation utilities for the cervical cytology datasets defined in `data.py`.

Pretraining (unsupervised) doesn't care about label semantics, so a plain
`torch.utils.data.ConcatDataset` over the raw dataset instances is enough --
see `build_pretraining_dataset`.

Downstream (supervised) is trickier: each dataset's raw label set means
something different, so labels have to be remapped onto a *shared* schema
before concatenation is meaningful. Two schemas are defined here, matching
what each dataset can honestly support:

  - BETHESDA schema: for datasets whose labels are (or cleanly correspond
    to) Bethesda System diagnostic categories. Native fit: MendeleyLBC and
    BTM (raw labels already ARE Bethesda terms). Direct mapping (dataset
    already encodes lesion grade): CPSMI2025.

  - MORPHOLOGICAL schema: for datasets labeled by cell morphology / dysplasia
    grade rather than a diagnostic category. Herlev and SIPaKMeD don't share
    a common fine-grained vocabulary with each other, so they're collapsed
    to a coarser Normal / Abnormal split, which both support without
    inventing correspondences the datasets don't actually encode.
"""

from torch.utils.data import Dataset, ConcatDataset

from data import (
    CPSMI2025Dataset,
    HerlevDataset,
    MendeleyLBCDataset,
    SIPaKMeDDataset,
    BTMDataset,
    HiCervixDataset,
)


# ---------------------------------------------------------------------------
# Schema 1: Bethesda System diagnostic categories
# ---------------------------------------------------------------------------

BETHESDA_CLASSES = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AdenoCA']

BETHESDA_LABEL_MAPS = {
    'MendeleyLBCDataset': {
        'Negative for Intraepithelial malignancy': 'NILM',
        'Low squamous intra-epithelial lesion': 'LSIL',
        'High squamous intra-epithelial lesion': 'HSIL',
        'Squamous cell carcinoma': 'SCC',
    },
    'CPSMI2025Dataset': {
        # normal/* -> NILM (Bethesda groups benign reactive/inflammatory
        # changes under NILM too)
        'endocervical': 'NILM',
        'metaplasia': 'NILM',
        'squamous': 'NILM',
        'inflammatory': 'NILM',
        # lesion/* maps directly, it's already graded like Bethesda
        'low grade': 'LSIL',
        'high grade': 'HSIL',
        # cancer/* -> carcinoma is CIN3/CIS's endpoint, grouped with HSIL
        # in Bethesda 2014; invasive squamous carcinoma is its own SCC
        # category; adeno is kept separate since it's a different lineage
        'in situ': 'HSIL',
        'invasive': 'SCC',
        'adeno': 'AdenoCA',
    },
    # Optional / extended: Herlev's dysplasia grades correspond fairly
    # directly to Bethesda's LSIL/HSIL split and are commonly remapped this
    # way in the literature, but it IS an interpretive mapping rather than
    # a dataset-native one -- cite this choice if you use it.
    'HerlevDataset': {
        'normal columnar': 'NILM',
        'normal intermediate': 'NILM',
        'normal superficiel': 'NILM',
        'light dysplastic': 'LSIL',
        'moderate dysplastic': 'HSIL',
        'severe dysplastic': 'HSIL',
        'carcinoma in situ': 'HSIL',
    },
    # Native fit, just like MendeleyLBC -- BTM's raw labels already ARE
    # Bethesda terms (NIL is the same category as NILM, just abbreviated).
    'BTMDataset': {
        'NIL': 'NILM',
        'LSIL': 'LSIL',
        'HSIL': 'HSIL',
    },
    'HiCervixDataset': {
        # Benign epithelial-cell types and organisms are reported under NILM.
        'Normal': 'NILM',
        'ECC': 'NILM',
        'RPC': 'NILM',
        'MPC': 'NILM',
        'Atrophy': 'NILM',
        'EMC': 'NILM',
        'FUNGI': 'NILM',
        'ACTINO': 'NILM',
        'TRI': 'NILM',
        'HSV': 'NILM',
        'CC': 'NILM',
        'ASC-US': 'ASC-US',
        'LSIL': 'LSIL',
        'ASC-H': 'ASC-H',
        'HSIL': 'HSIL',
        'SCC': 'SCC',
        'ADC': 'AdenoCA',
        # These categories do not have a defensible equivalent in this shared
        # seven-class schema, so they are excluded from its training samples.
        'PG': None,
        'HCG': None,
        'AGC-NOS': None,
        'AGC-FN': None,
    },
}


# ---------------------------------------------------------------------------
# Schema 2: coarse morphological Normal / Abnormal split
# ---------------------------------------------------------------------------
# For datasets whose native classes describe cell type / morphology rather
# than a diagnostic category, and which don't share a vocabulary with each
# other at fine granularity (Herlev's 7 dysplasia/cell-type classes vs.
# SIPaKMeD's 5 cell-type classes). Collapsing to Normal/Abnormal is the
# standard way both are benchmarked individually in the literature, and it's
# the coarsest common ground that doesn't require inventing correspondences.

MORPHOLOGICAL_CLASSES = ['normal', 'abnormal']

MORPHOLOGICAL_LABEL_MAPS = {
    'HerlevDataset': {
        'normal columnar': 'normal',
        'normal intermediate': 'normal',
        'normal superficiel': 'normal',
        'light dysplastic': 'abnormal',
        'moderate dysplastic': 'abnormal',
        'severe dysplastic': 'abnormal',
        'carcinoma in situ': 'abnormal',
    },
    'SIPaKMeDDataset': {
        'Superficial-Intermediate': 'normal',
        'Parabasal': 'normal',
        'Metaplastic': 'normal',       # benign squamous metaplasia
        'Koilocytotic': 'abnormal',    # classic HPV/LSIL cytologic marker
        'Dyskeratotic': 'abnormal',
    },
    'HiCervixDataset': {
        'normal': 'normal',
        'abnormal': 'abnormal',
    },
}


# ---------------------------------------------------------------------------
# Wrapper that remaps a dataset's raw labels onto a shared target schema
# ---------------------------------------------------------------------------

class RemappedLabelDataset(Dataset):
    """
    Wraps a `BaseCervicalCytologyDataset` instance and rewrites its integer
    label onto a shared `target_classes` vocabulary, using a
    raw-label -> target-label string map. This is what makes datasets with
    different native label sets concatenable for a single classification
    head.
    """

    def __init__(self, dataset, label_map, target_classes):
        self.dataset = dataset
        self.label_map = label_map
        self.target_classes = target_classes
        self.target_class_to_idx = {c: i for i, c in enumerate(target_classes)}

        self._idx_to_raw_label = {v: k for k, v in dataset.class_to_idx.items()}
        unmapped = set(self._idx_to_raw_label.values()) - set(label_map.keys())
        assert not unmapped, (
            f"{type(dataset).__name__} has raw labels with no entry in the "
            f"label map: {sorted(unmapped)}."
        )
        self.indices = [
            idx for idx, (_, raw_label) in enumerate(dataset.samples)
            if label_map[raw_label] is not None
        ]
        assert self.indices, f'{type(dataset).__name__} has no samples in the target schema.'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        raw_label = self._idx_to_raw_label[item['label']]
        item['label'] = self.target_class_to_idx[self.label_map[raw_label]]
        return item


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_pretraining_dataset(dataset_instances):
    """
    dataset_instances: list of dataset objects already constructed with
    task='pretraining', split='train'. Labels aren't used, so no remapping
    is needed -- a plain ConcatDataset is enough.
    """
    return ConcatDataset(dataset_instances)


def build_downstream_dataset(dataset_instances, label_maps, target_classes):
    """
    dataset_instances: list of dataset objects constructed with
    task='downstream' and a shared split ('train' or 'test').
    label_maps: BETHESDA_LABEL_MAPS or MORPHOLOGICAL_LABEL_MAPS (or your own).
    """
    wrapped = [
        RemappedLabelDataset(ds, label_maps[type(ds).__name__], target_classes)
        for ds in dataset_instances
    ]
    return ConcatDataset(wrapped)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
"""
paths = {
    'cpsmi2025': '/data/CPSMI2025',
    'herlev': '/data/Herlev Dataset',
    'mendeley': '/data/MendeleyLBC',
    'sipakmed': '/data/SIPaKMeD',
    'btm': '/data/BTM',
}

# --- pretraining: every dataset, labels irrelevant ---
pretraining_ds = build_pretraining_dataset([
    CPSMI2025Dataset(paths['cpsmi2025'], task='pretraining', split='train', transform=pretrain_transform),
    HerlevDataset(paths['herlev'], task='pretraining', split='train', transform=pretrain_transform),
    MendeleyLBCDataset(paths['mendeley'], task='pretraining', split='train', transform=pretrain_transform),
    SIPaKMeDDataset(paths['sipakmed'], task='pretraining', split='train', transform=pretrain_transform),
    BTMDataset(paths['btm'], task='pretraining', split='train', transform=pretrain_transform),
])

# --- downstream, Bethesda schema: MendeleyLBC + BTM (native) + CPSMI2025 (mapped) ---
# Add HerlevDataset(...) to this list too if you're OK with its extended mapping.
bethesda_train = build_downstream_dataset(
    [
        MendeleyLBCDataset(paths['mendeley'], task='downstream', split='train', transform=eval_transform),
        CPSMI2025Dataset(paths['cpsmi2025'], task='downstream', split='train', transform=eval_transform),
        BTMDataset(paths['btm'], task='downstream', split='train', transform=eval_transform),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)
bethesda_test = build_downstream_dataset(
    [
        MendeleyLBCDataset(paths['mendeley'], task='downstream', split='test', transform=eval_transform),
        CPSMI2025Dataset(paths['cpsmi2025'], task='downstream', split='test', transform=eval_transform),
        BTMDataset(paths['btm'], task='downstream', split='test', transform=eval_transform),
    ],
    BETHESDA_LABEL_MAPS,
    BETHESDA_CLASSES,
)

# --- downstream, morphological schema: Herlev + SIPaKMeD ---
morphological_train = build_downstream_dataset(
    [
        HerlevDataset(paths['herlev'], task='downstream', split='train', transform=eval_transform),
        SIPaKMeDDataset(paths['sipakmed'], task='downstream', split='train', transform=eval_transform),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)
morphological_test = build_downstream_dataset(
    [
        HerlevDataset(paths['herlev'], task='downstream', split='test', transform=eval_transform),
        SIPaKMeDDataset(paths['sipakmed'], task='downstream', split='test', transform=eval_transform),
    ],
    MORPHOLOGICAL_LABEL_MAPS,
    MORPHOLOGICAL_CLASSES,
)
"""
