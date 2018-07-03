import sys, os
import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader


class SVHNDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # Normalization that every pytorch pretrained models expect
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        self.dataset = datasets.SVHN(data_dir, split='train', transform=trsfm, download=True)
        super(SVHNDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers)


class CocoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):

        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
            
        self.img_dir = os.path.join(data_dir, "images/train2014")
        self.ann_dir = os.path.join(data_dir, "annotations/captions_train2014.json")

        trsfm = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            # Normalization that every pytorch pretrained models expect
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = datasets.CocoCaptions(self.img_dir, self.ann_dir, transform=trsfm)
        super(CocoDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers)



if __name__ == '__main__':

    data_dir = '../cocoapi'
    img_dir = os.path.join(data_dir, "images/train2014")
    ann_dir = os.path.join(data_dir, "annotations/captions_train2014.json")

    trsfm = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.CocoCaptions(img_dir, ann_dir, transform=trsfm)
    for index in range(8):
        data, target = dataset[index]
        print(target)
        print(data.shape)
        print(data)
        print(torch.min(data))

        if index == 7:
            save_image(data, 'index7.png')
