import os
from enum import Enum
from typing import Tuple, List, Union
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

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
                 resize=(256, 256),
                 imagesize=224,
                 split=DatasetSplit.TRAIN,
                 train_val_test_split: Union[Tuple[float, float, float], None] = None,
                 num_images: Union[Tuple[int, int, int, int], None] = None,
                 # (num_not_cancer, num_skip_not_cancer, num_cancer, num_skip_cancer)
                 rotate_degrees=0,
                 v_flip_p=0.,
                 h_flip_p=0.,
                 noise_std=0.,
                 brightness_range=(1., 1.),
                 contrast_range=(1., 1.)
                 ):
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
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Resize(self.resize),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range),
            transforms.Lambda(lambda x: x + torch.rand_like(x) * noise_std),
            transforms.Lambda(lambda x: torch.clip(x, 0, 1)),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def __load(self):
        if self.num_images:
            self.__load_num_images()
        else:
            self.__load_train_val_test()

        self.metaData['image_path'] = self.metaData['patient_id'].astype(str) + "_" + self.metaData['image_id'].astype(
            str) + ".png"
        self.metaData['image_path'] = self.metaData['image_path'].apply(lambda img: os.path.join(self.img_dir, img))

    def __load_num_images(self):
        num_not_cancer, num_skip_not_cancer, num_cancer, num_skip_cancer = self.num_images[0]

        not_cancer_df = self.metaData.loc[self.metaData['cancer'] == 0]
        not_cancer_df = not_cancer_df[num_skip_not_cancer:num_skip_not_cancer + num_not_cancer]
        not_cancer_df.sample(frac=1)
        cancer_df = self.metaData.loc[self.metaData['cancer'] == 1]
        cancer_df = cancer_df[num_skip_cancer:num_skip_cancer + num_cancer]
        cancer_df = cancer_df.sample(frac=1)

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
        img_path = self.metaData.iloc[idx]['image_path']
        image = Image.open(img_path)

        # image = pad(image, self.get_padding(image), fill=1)
        image = self.transform_img(image)

        cancer = self.metaData.iloc[idx]['cancer']
        return {
            "image": image,
            "classname": "",
            "anomaly": cancer,
            "is_anomaly": cancer,
            "image_name": os.path.split(img_path)[-1],
            "image_path": img_path,
        }

    def get_padding(self, image) -> List[int]:
        imsize = image.size
        h_padding = (self.resize[0] - imsize[0]) / 2
        v_padding = (self.resize[1] - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

        padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]

        return padding


if __name__ == '__main__':
    train_ds = BreastCancerDataset(
        img_dir="../data/rsna_pp/1024",
        meta_data_csv_path="../train_small.csv",
        num_images=(0, 0, 100, 0),
        rotate_degrees=20,
        noise_std=0.25,
        v_flip_p=0.5,
        h_flip_p=0.25,
        resize=(1024, 1024)
    )
    from matplotlib import pyplot as plt

    for im in train_ds:
        plt.imshow(im['image'].permute(1, 2, 0))
        plt.show()
        input()
