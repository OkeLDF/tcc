import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class CPSMI2025Dataset(Dataset):

    def __init__(self, path, use_first_class_level=False, transform=None):
        super().__init__()

        self.path = path
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
        img_fname, label = self.samples[idx]
        image = cv2.imread(img_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class MendeleyLBCDataset(Dataset): pass

class SIPaKMeDDataset(Dataset): pass