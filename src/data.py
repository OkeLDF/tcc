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
