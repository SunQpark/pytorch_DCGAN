import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

class My_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        raise NotImplementedError

    def __getitem__(self, idx):
        data, target = None, None
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        raise NotImplementedError
        return data, target


class CocoWrapper(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, "images/train2014")
        self.ann_dir = os.path.join(self.data_dir, "annotations/captions_train2014.json")
        self.coco = datasets.CocoCaptions(self.img_dir, self.ann_dir, transform=trsfm)

    def __getitem__(self, idx):
        data, target = self.coco[idx]
        return data
    
