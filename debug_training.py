from datetime import datetime

from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from src import backbones
from simplenet import SimpleNet
from torch.utils.data import DataLoader
from time import time
import dotenv
import os

dotenv.load_dotenv()
device = 'mps'

img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

train_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    num_images=(512, 0, 0, 0)
)

val_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(32, 512, 8, 0)
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

backbone = backbones.load("resnet50")
net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['layer2', 'layer3'],
    device=device,
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=64,
    target_embed_dimension=64,
    patchsize=3,
    embedding_size=None,
    meta_epochs=10,
    aed_meta_epochs=5,
    gan_epochs=3,
    noise_std=0.05,
    dsc_layers=2,
    dsc_hidden=64,
    dsc_margin=0.5,
    dsc_lr=0.005,
    auto_noise=0,
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

net.train(train_loader, val_loader)
scores, segmentations, features, labels_gt, masks_gt = net.predict(val_loader)
del scores, segmentations, features, labels_gt, masks_gt
