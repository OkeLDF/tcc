import warnings

import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def assert_dataset_attributes(path, split, task):
    assert task in ('pretraining', 'downstream'), (
        f"`task` should be 'pretraining' or 'downstream', got: {task!r}."
    )
    assert split in ('train', 'valid', 'test'), (
        f"`split` should be 'train', 'valid', or 'test', got: {split!r}."
    )
    assert not (task == 'pretraining' and split == 'test'), (
        "`split` can't be 'test' on unsupervised pretraining. Use split='train' or split='valid'."
    )
    assert not (task == 'downstream' and split == 'valid'), (
        "`split='valid'` is reserved for unsupervised pretraining."
    )
    assert Path(path).exists(), f"Path {path!r} does not exist."


def list_images(directory, extensions=IMAGE_EXTENSIONS):
    """Return a sorted list of image file paths directly inside `directory` (non-recursive)."""
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


class BaseCervicalCytologyDataset(Dataset):
    """
    Common interface every dataset in this module respects, so all four
    directory layouts (CPSMI2025, Herlev, MendeleyLBC, SIPaKMeD, BTM) can be
    consumed the same way by training code.

    Subclasses only implement `_collect_samples`, returning a list of
    `(image_path, label)` tuples for the *whole* dataset (unsplit, raw
    string labels). Everything else -- validation, class indexing,
    splitting, image loading -- is handled here.

    Split contract (stratified by label, seeded with RANDOM_STATE=42):
        task='pretraining', split='train'   -> SimCLR training partition
        task='pretraining', split='valid'   -> SimCLR validation partition
        task='downstream',  split='train'   -> the same training partition
        task='downstream',  split='test'    -> held-out test partition

    This is inductive self-supervised learning: pretraining and supervised
    finetuning may see the same training images, but the test images remain
    unseen until final evaluation. Labels are never returned for pretraining.
    """

    def __init__(self, path, split='train', task='pretraining', transform=None,
                 test_size=0.2, validation_size=0.1):
        super().__init__()

        assert_dataset_attributes(path=path, split=split, task=task)

        self.path = Path(path)
        self.split = split
        self.task = task
        self.transform = transform

        samples = self._collect_samples()
        assert len(samples) > 0, f"No samples were found under {str(self.path)!r}."
        self.samples = np.array(samples)

        self.classes = np.unique(self.samples[:, 1])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes.tolist())}

        self.samples = self._apply_split(test_size, validation_size)

    def _collect_samples(self):
        """Return a list of (image_path: str, label: str) tuples for every sample."""
        raise NotImplementedError

    def _apply_split(self, test_size, validation_size):
        labels = self.samples[:, 1]

        train_idx, test_idx = train_test_split(
            np.arange(len(self.samples)),
            test_size=test_size,
            stratify=labels,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        if self.task == 'downstream':
            return self.samples[train_idx] if self.split == 'train' else self.samples[test_idx]

        ssl_train_idx, ssl_valid_idx = train_test_split(
            train_idx,
            test_size=validation_size,
            stratify=labels[train_idx],
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return self.samples[ssl_train_idx] if self.split == 'train' else self.samples[ssl_valid_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path!r}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is None:
            raise RuntimeError('A transform is required to produce model-ready tensors.')

        transformed = self.transform(image)
        if self.task == 'pretraining':
            aug_i, aug_j = transformed
            return {'augmented_i': aug_i, 'augmented_j': aug_j}

        return {'pixel_values': transformed, 'label': self.class_to_idx[label]}


class CPSMI2025Dataset(BaseCervicalCytologyDataset):
    """
    Layout: <path>/<class>/<subclass>/*.<ext>
    e.g. CPSMI2025/cancer/adeno, CPSMI2025/lesion/high_grade, ...
    """


    def __init__(self, path, use_first_class_level=False, split='train', task='pretraining',
                 transform=None, test_size=0.2, validation_size=0.1):
        self.use_first_class_level = use_first_class_level
        super().__init__(
            path,
            split=split,
            task=task,
            transform=transform,
            test_size=test_size,
            validation_size=validation_size,
        )

    def _collect_samples(self):
        samples = []
        for class_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            for subclass_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                label = class_dir.name if self.use_first_class_level else subclass_dir.name
                label = label.replace('_', ' ')
                for img_path in list_images(subclass_dir):
                    samples.append((str(img_path), label))
        return samples


class HerlevDataset(BaseCervicalCytologyDataset):
    """
    Layout: <path>/{train,test}/<class>/*.bmp

    The dataset's supplied split is preserved: images under ``train`` feed
    pretraining and downstream training, while images under ``test`` are used
    only for downstream evaluation.
    """

    def _collect_samples(self):
        samples = []
        for split_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = class_dir.name.replace('_', ' ')
                for img_path in list_images(class_dir):
                    samples.append((str(img_path), label))
        return samples

    def _apply_split(self, test_size, validation_size):
        del test_size
        train_samples = [
            sample for sample in self.samples
            if Path(sample[0]).parents[1].name == 'train'
        ]
        if self.task == 'pretraining':
            labels = np.array(train_samples)[:, 1]
            train_idx, valid_idx = train_test_split(
                np.arange(len(train_samples)), test_size=validation_size,
                stratify=labels, shuffle=True, random_state=RANDOM_STATE,
            )
            selected = np.array(train_samples)[train_idx if self.split == 'train' else valid_idx]
        else:
            selected_folder = self.split
            selected = [
                sample for sample in self.samples
                if Path(sample[0]).parents[1].name == selected_folder
            ]
        assert len(selected), f'No samples found in Herlev {self.split!r} split.'
        return np.array(selected)


class MendeleyLBCDataset(BaseCervicalCytologyDataset):
    """
    Layout: <path>/<class>/*.<ext> (each class folder also has a Results.csv,
    which is skipped automatically since it isn't an image extension).
    """

    def _collect_samples(self):
        samples = []
        for class_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            label = class_dir.name
            for img_path in list_images(class_dir):
                samples.append((str(img_path), label))
        return samples


class SIPaKMeDDataset(BaseCervicalCytologyDataset):
    """
    Layout: <path>/im_<class>/im_<class>/CROPPED/*.bmp (+ *_cyt.dat / *_nuc.dat
    boundary annotations, skipped automatically since .dat isn't an image
    extension).

    NOTE: SIPaKMeD ships both the original multi-cell slide images
    (directly under im_<class>/im_<class>/) and single-cell crops (under
    .../CROPPED/). This class uses the CROPPED single-cell images, which is
    the standard choice for cell-level classification.
    Flag this if you'd rather pretrain on the uncropped slide images instead.
    """

    def _collect_samples(self):
        samples = []
        for class_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            label = class_dir.name.replace('im_', '').replace('_', ' ')
            cropped_dir = class_dir / class_dir.name / 'CROPPED'
            assert cropped_dir.exists(), f"Expected {str(cropped_dir)!r} to exist."
            for img_path in list_images(cropped_dir):
                samples.append((str(img_path), label))
        return samples


class BTMDataset(BaseCervicalCytologyDataset):
    """
    Layout: <path>/manifest.csv with columns `path` (relative, prefixed with
    'BTM/') and `name` (whose first whitespace-separated token is the label).
    """

    def _collect_samples(self):
        manifest_path = self.path / 'manifest.csv'
        assert manifest_path.exists(), f"{str(manifest_path)!r} does not exist."

        manifest = pd.read_csv(manifest_path)
        manifest['path'] = manifest['path'].str.split('BTM/').str[-1]
        manifest['label'] = manifest['name'].str.split(' ').str[0]

        return [
            (str(self.path / row.path), row.label)
            for row in manifest.itertuples()
        ]


class HiCervixDataset(BaseCervicalCytologyDataset):
    """HiCervix dataset with its supplied train/validation/test partitions.

    Layout::

        <path>/train.csv, <path>/val.csv, <path>/test.csv
        <path>/train/<image_name>, <path>/val/<image_name>, <path>/test/<image_name>

    ``label_level=1`` uses HiCervix's own coarse ``level_1`` field
    (``negative``/``microbe``/``ASC``/``AGC``) collapsed into a binary
    ``normal``/``abnormal`` target. This binary split is fully determined by
    HiCervix's own hierarchy (level_1 has exactly these 4 branches, and the
    normal-vs-abnormal assignment isn't a judgment call), so it's baked in
    here rather than left to concat_datasets.py.

    ``label_level=2`` returns the *raw* ``level_2`` value unchanged (falling
    back to ``class_name`` -- which equals whichever level is deepest for
    that row -- when ``level_2`` is missing, e.g. generic ``'AGC'`` leaf rows
    with no resolved NOS/FN subtype). Unlike level_1, this is NOT collapsed
    onto Bethesda terms here: which level_2 categories count as NILM, which
    get excluded, etc. is a shared-schema policy decision that belongs in
    concat_datasets.py's BETHESDA_LABEL_MAPS (where it's visible and easy to
    revise), not baked into the loader.

    The dataset's official partitions are never randomly repartitioned:
    ``split='train'``, ``'valid'``, and ``'test'`` select ``train/``,
    ``val/``, and ``test/`` respectively -- ``test_size``/``validation_size``
    are accepted only for interface compatibility with the other datasets
    and are ignored (a warning is raised if you pass non-default values, so
    this doesn't fail silently).
    """

    LEVEL_1_BINARY_MAP = {
        'negative': 'normal',
        'microbe': 'normal',
        'ASC': 'abnormal',
        'AGC': 'abnormal',
    }

    SPLIT_TO_SOURCE = {'train': 'train', 'valid': 'val', 'test': 'test'}

    def __init__(self, path, label_level=1, split='train', task='pretraining',
                 transform=None, test_size=0.2, validation_size=0.1):
        assert label_level in (1, 2), f'`label_level` should be 1 or 2, got: {label_level!r}.'
        self.label_level = label_level
        if test_size != 0.2 or validation_size != 0.1:
            warnings.warn(
                'HiCervixDataset always uses its own official train/val/test '
                'partitions; the `test_size`/`validation_size` you passed are ignored.'
            )
        super().__init__(
            path,
            split=split,
            task=task,
            transform=transform,
            test_size=test_size,
            validation_size=validation_size,
        )

    def _collect_samples(self):
        samples = []
        missing_images = []

        for source_split in self.SPLIT_TO_SOURCE.values():
            csv_path = self.path / f'{source_split}.csv'
            image_dir = self.path / source_split
            assert csv_path.exists(), f'Expected annotation file {str(csv_path)!r} to exist.'
            assert image_dir.is_dir(), (
                f'Expected image directory {str(image_dir)!r} to exist. '
                f'Extract {source_split}.zip into {str(self.path)!r} first.'
            )

            annotations = pd.read_csv(csv_path)
            required_columns = {'image_name', 'class_name', 'level_1', 'level_2'}
            missing_columns = required_columns - set(annotations.columns)
            assert not missing_columns, (
                f'{str(csv_path)!r} is missing required columns: {sorted(missing_columns)}.'
            )

            for row in annotations.itertuples(index=False):
                image_path = image_dir / row.image_name
                if not image_path.is_file():
                    missing_images.append(image_path)
                    continue

                if self.task == 'pretraining':
                    label = 'unlabeled'

                elif self.label_level == 1:
                    try:
                        label = self.LEVEL_1_BINARY_MAP[row.level_1]
                    except KeyError as error:
                        raise ValueError(
                            f'Unknown HiCervix level-1 label {row.level_1!r} in {str(csv_path)!r}. '
                            f'Add it to LEVEL_1_BINARY_MAP.'
                        ) from error

                else:  # label_level == 2, raw pass-through
                    label = row.class_name if pd.isna(row.level_2) else row.level_2

                samples.append((str(image_path), label))

        if missing_images:
            examples = ', '.join(str(path) for path in missing_images[:3])
            raise FileNotFoundError(
                f'HiCervix is incomplete: {len(missing_images)} CSV-referenced images are missing. '
                f'Examples: {examples}'
            )
        return samples

    def _apply_split(self, test_size, validation_size):
        del test_size, validation_size
        source_split = self.SPLIT_TO_SOURCE[self.split]
        selected = [
            sample for sample in self.samples
            if Path(sample[0]).parent.name == source_split
        ]
        assert selected, f'No samples found in HiCervix {source_split!r} partition.'
        return np.array(selected)