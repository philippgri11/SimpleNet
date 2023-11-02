import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485]
IMAGENET_STD = [0.229]


class BreastCancerDataset(Dataset):
    def __init__(self, img_dir,
                 meta_data=None,
                 meta_data_csv_path=None,
                 resize=256,
                 imagesize=224,
                 train_val_split=1.0,
                 rotate_degrees=0,
                 translate=0,
                 brightness_factor=0,
                 contrast_factor=0,
                 saturation_factor=0,
                 gray_p=0,
                 h_flip_p=0,
                 v_flip_p=0,
                 scale=0):

        if meta_data is None and meta_data_csv_path is None:
            raise ValueError(
                "Beide Parameter 'meta_data' und 'meta_data_csv_path' sind nicht definiert. Bitte definieren Sie mindestens einen der beiden.")

        if meta_data_csv_path:
            self.metaData = pd.read_csv(meta_data_csv_path)
        else:
            self.metaData = meta_data
        self.img_dir = img_dir
        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def __len__(self):
        return len(self.metaData)

    def __getitem__(self, idx):
        img_name = f"{self.metaData.iloc[idx, 1]}_{self.metaData.iloc[idx, 2]}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        self.transform_img(image)

        cancer = self.metaData.iloc[idx, 6]

        return {
            "image": image,
            "classname": "",
            "anomaly": cancer,
            "is_anomaly": cancer,
            "image_name": os.path.split(img_path)[-1],
            "image_path": img_path,
        }

