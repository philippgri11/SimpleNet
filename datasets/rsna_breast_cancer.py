import os
from enum import Enum

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import functional_pil

IMAGENET_MEAN = [0.485]
IMAGENET_STD = [0.229]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BreastCancerDataset(Dataset):
    def __init__(self, img_dir,
                 meta_data=None,
                 meta_data_csv_path=None,
                 resize=256,
                 imagesize=224,
                 split=DatasetSplit.TRAIN,
                 train_val_test_split=(0.7, 0.2, 0.1),
                 rotate_degrees=0,
                 translate=0,
                 brightness_factor=0,
                 contrast_factor=0,
                 saturation_factor=0,
                 gray_p=0,
                 h_flip_p=0,
                 v_flip_p=0,
                 scale=0):
        super().__init__()
        self.img_dir = img_dir
        self.meta_data = meta_data
        self.meta_data_csv_path = meta_data_csv_path
        self.resize = resize
        self.imagesize = imagesize
        self.split = split
        self.train_val_test_split = train_val_test_split
        self.rotate_degrees = rotate_degrees
        self.translate = translate
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.gray_p = gray_p
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.scale = scale

        # Load metadata
        if meta_data is None and meta_data_csv_path is None:
            raise ValueError(
                "Beide Parameter 'meta_data' und 'meta_data_csv_path' sind nicht definiert. Bitte definieren Sie mindestens einen der beiden.")

        if meta_data_csv_path:
            self.metaData = pd.read_csv(meta_data_csv_path)
        else:
            self.metaData = meta_data

        n_images = len(self.metaData)
        train_end = int(n_images * train_val_test_split[0])
        val_end = train_end + int(n_images * train_val_test_split[1])
        if self.split.value == DatasetSplit.TRAIN.value:
            self.metaData = self.metaData[:train_end]
        elif self.split.value == DatasetSplit.VAL.value:
            self.metaData = self.metaData[train_end:val_end]
        elif self.split.value == DatasetSplit.TEST.value:
            self.metaData = self.metaData[val_end:]

        # Define the transformations
        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            # transforms.RandomHorizontalFlip(h_flip_p),
            # transforms.RandomVerticalFlip(v_flip_p),
            # transforms.RandomGrayscale(gray_p),
            # transforms.RandomAffine(rotate_degrees,
            #                         translate=(translate, translate),
            #                         scale=(1.0 - scale, 1.0 + scale),
            #                         interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def __len__(self):
        return len(self.metaData)

    def __getitem__(self, idx):
        img_name = f"{self.metaData.iloc[idx, 1]}_{self.metaData.iloc[idx, 2]}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        image = self.transform_img(image)
        cancer = self.metaData.iloc[idx, 6]

        return {
            "image": image,
            "classname": "",
            "anomaly": cancer,
            "is_anomaly": cancer,
            "image_name": os.path.split(img_path)[-1],
            "image_path": img_path,
        }
        # return image, cancer
