import os
import random
from datetime import datetime

import dotenv
import torch
from torch.utils.data import DataLoader, ConcatDataset

from datasets.mnist import MNISTDataset
from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet
from src import backbones

random.seed(42)


dotenv.load_dotenv()
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

train_ds = MNISTDataset(
    img_dir=img_dir, split=DatasetSplit.TRAIN, classidx_to_remove=0
)

val_ds = MNISTDataset(
    img_dir=img_dir, split=DatasetSplit.VAL
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

backbone = backbones.load("resnet50")

net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['layer1'],
    device=device,
    input_shape=(3, 64, 64),
    pretrain_embed_dimension=256,
    target_embed_dimension=256,
    patchsize=3,
    embedding_size=None,
    meta_epochs=100,
    aed_meta_epochs=1,
    gan_epochs=5,
    noise_std=0.01,
    dsc_layers=2,
    dsc_hidden=128,
    dsc_margin=0.5,
    dsc_lr=0.005,
    pre_proj_lr=0.0001,
    train_backbone=True,
    cos_lr=False,
    lr=0.005,
    pre_proj=1,
    proj_layer_type=1,
    mix_noise=1,
)

models_dir = f'models/{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
dataset_name = "rsna_breast_cancer"
net.set_model_dir(models_dir, dataset_name)
net.save_model_params()

net.train(train_loader, val_loader)
