import os
from enum import Enum
from typing import Tuple
import matplotlib.pyplot as plt# Convert from tensor to PIL Image
from torchvision.transforms.functional import pad
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
                 resize=(256,256),
                 imagesize=224,
                 split=DatasetSplit.TRAIN,
                 train_val_test_split: Tuple[float, float, float] | None = None,
                 num_images: Tuple[int, int, int, int] | None = None,
                 # (num_not_cancer, num_skip_not_cancer, num_cancer, num_skip_cancer)
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
        self.num_images = num_images,
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

        if self.train_val_test_split and self.num_images:
            raise ValueError(
                "Es dürfen nicht gleichzeit TrainValTestSplit und num_images gesetzt sein!"
            )

        # Load metadata
        if meta_data is None and meta_data_csv_path is None:
            raise ValueError(
                "Beide Parameter 'meta_data' und 'meta_data_csv_path' sind nicht definiert. Bitte definieren Sie mindestens einen der beiden.")

        if meta_data_csv_path:
            self.metaData = pd.read_csv(meta_data_csv_path)
        else:
            self.metaData = meta_data

        self.__load()

        # Define the transformations
        self.transform_img = [
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def __load(self):
        if self.num_images:
            self.__load_num_images()
        else:
            self.__load_train_val_test()

    def __load_num_images(self):
        num_not_cancer, num_skip_not_cancer, num_cancer, num_skip_cancer = self.num_images[0]

        not_cancer_df = self.metaData.loc[self.metaData['cancer'] == 0]
        not_cancer_df = not_cancer_df[num_skip_not_cancer:num_skip_not_cancer + num_not_cancer]
        cancer_df = self.metaData.loc[self.metaData['cancer'] == 1]
        cancer_df = cancer_df[num_skip_cancer:num_skip_cancer + num_cancer]

        self.metaData = pd.concat([not_cancer_df, cancer_df])

    def __load_train_val_test(self):
        n_images = len(self.metaData)
        train_end = int(n_images * self.train_val_test_split[0])
        val_end = train_end + int(n_images * self.train_val_test_split[1])

        if self.split.value == DatasetSplit.TRAIN.value:
            self.metaData = self.metaData[:train_end]
        elif self.split.value == DatasetSplit.VAL.value:
            self.metaData = self.metaData[train_end:val_end]
        elif self.split.value == DatasetSplit.TEST.value:
            self.metaData = self.metaData[val_end:]

    def __len__(self):
        return len(self.metaData)

    def __getitem__(self, idx):
        patient_id = str(self.metaData.iloc[idx, 1])
        image_id = str(self.metaData.iloc[idx, 2])
        img_name = f"{patient_id}_{image_id}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = pad(image, self.get_padding(image), fill=1)
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

    def get_padding(self,image):
        imsize = image.size
        h_padding = (self.resize[0] - imsize[0]) / 2
        v_padding = (self.resize[1] - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        return padding