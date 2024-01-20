import os
from datetime import datetime

import dotenv
from torch.utils.data import DataLoader

from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet
from src import backbones

dotenv.load_dotenv()
device = 'mps'

img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

train_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    num_images=(64, 0, 0, 0),
    rotate_degrees=5
)

val_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(64, 64, 8, 0)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

backbone = backbones.load("resnet50")
net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['layer3'],
    device=device,
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=128,
    target_embed_dimension=128,
    patchsize=3,
    embedding_size=None,
    meta_epochs=1,
    aed_meta_epochs=1,
    gan_epochs=1,
    noise_std=0.01,
    dsc_layers=2,
    dsc_hidden=64,
    dsc_margin=0.5,
    dsc_lr=0.001,
    train_backbone=True,
    cos_lr=False,
    lr=0.001,
    pre_proj=1,
    proj_layer_type=1,
    mix_noise=1,
)
models_dir = f'models/{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
dataset_name = "rsna_breast_cancer"
net.set_model_dir(models_dir, dataset_name)
net.save_model_params()

net.train(train_loader, val_loader)

