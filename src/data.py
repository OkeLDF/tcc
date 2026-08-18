import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def assert_dataset_attributes(path, split, task):
    assert task in ('pretraining', 'downstream'), f"`task` should be 'pretraining' or 'downstream', got: {task!r}."
    assert split in ('train', 'test'), f"`split` should be 'train' or 'test', got: {split!r}."
    assert not (task == 'pretraining' and split == 'test'),
        f"`split` can't be 'test' on unsupervised pretraining. Use split='train' or task='downstream'."
    assert Path(path).exists(), f"Path {path!r} does not exist."


class CPSMI2025Dataset(Dataset):

    def __init__(self, path, use_first_class_level=False, split='train', task='pretraining', transform=None):
        super().__init__()
        
        assert_dataset_attributes(path=path, split=split, task=task)

        self.split = split
        self.task = task
        self.path = Path(path)
        self.transform = transform
        self.samples = []

        for class_folder in sorted(os.listdir(path), reverse=True):
            class_path = os.path.join(path, class_folder)
            
            for subclass_folder in sorted(os.listdir(class_path), reverse=True):
                subclass_path = os.path.join(class_path, subclass_folder)
                
                for img_fname in sorted(os.listdir(subclass_path), reverse=False):
                    label = class_folder if use_first_class_level else subclass_folder
                    self.samples.append((os.path.join(subclass_path, img_fname), label.replace('_', ' ')))

        self.samples = np.array(self.samples)
        self.classes = np.unique(self.samples[:, 1], sorted=False)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes.tolist())}

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'pixel_values': image,
            'label': label
        }


class MendeleyLBCDataset(Dataset):
    
    def __init__(self, path, split='train', task='pretraining', transform=None):
        super().__init__()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'pixel_values': image,
            'label': label
        }


class SIPaKMeDDataset(Dataset):
    
    def __init__(self, path, split='train', task='pretraining', transform=None):
        super().__init__()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'pixel_values': image,
            'label': label
        }
    
class HerlevDataset(Dataset):
    
    def __init__(self, path, split='train', task='pretraining', transform=None):
        super().__init__()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'pixel_values': image,
            'label': label
        }

    
class BTMDataset(Dataset):
    
    def __init__(self, path, task='pretraining', split='train', donwstream_size=0.5, test_size=0.25, transform=None):
        super().__init__()

        assert_dataset_attributes(path=path, split=split)
        
        self.split = split
        self.task = task
        self.path = Path(path)
        manifest_path = self.path / 'manifest.csv'
        assert manifest_path.exists(), f"{str(manifest_path)!r} does not exist."
        
        self.transform = transform
        self.manifest = pd.read_csv(manifest_path)
        self.manifest.path = self.manifest.path.str.split('BTM/').apply(lambda x: x[-1])
        self.manifest['label'] = self.manifest.name.str.split(" ").apply(lambda x: x[0])
        self.classes = self.manifest.label.unique()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes.tolist())}

        pretraining, downstream = train_test_split(
            self.manifest,
            test_size=downstream_size,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        self.manifest = pretraining if self.split == 'pretraining' else downstream

        if self.task == 'downstream':
            pretraining, downstream = train_test_split(
                self.manifest,
                test_size=test_size,
                stratify=self.manifest.label,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            self.manifest = train if self.split == 'train' else test

    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        img_path = self.manifest.path[idx]
        img_path = Path(img_path)
        
        label = self.manifest.label[idx]
        label = self.class_to_idx[label]
        
        image = cv2.imread(str(self.path / img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'pixel_values': image,
            'label': label
        }