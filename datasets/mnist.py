from enum import Enum

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import transforms

IMAGENET_MEAN = [0.485]
IMAGENET_STD = [0.229]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MNISTDataset(Dataset):
    def __init__(self, img_dir, split: DatasetSplit, classidx_to_remove=0):
        super().__init__()
        self.img_dir = img_dir
        train = split == DatasetSplit.TRAIN
        self.remove_class = classidx_to_remove
        self.ds = torchvision.datasets.MNIST(self.img_dir, train=train, download=True)

        targets = self.ds.targets
        target_indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(target_indices, train_size=0.8)

        if split == DatasetSplit.TRAIN:
            idx_to_keep = targets[train_idx] != self.remove_class
            train_idx = train_idx[idx_to_keep]
            self.ds = Subset(self.ds, train_idx)
        else:
            self.ds = Subset(self.ds, val_idx)

        self.transform_img = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Resize((64, 64))
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        (image, target) = self.ds.__getitem__(idx)
        image = self.transform_img(image)
        return {
            "image": image,
            "classname": "",
            "anomaly": target,
            "is_anomaly": target == self.remove_class,
            "image_name": "",
            "image_path": "",
        }

if __name__ == '__main__':
    train_ds = MNISTDataset(
        img_dir="../data/mnist", split=DatasetSplit.TRAIN, classidx_to_remove=1
    )
    from matplotlib import pyplot as plt

    for im in train_ds:
        plt.imshow(im['image'].permute(1, 2, 0))
        plt.show()
        input()
