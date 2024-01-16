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
    train_val_test_split=(0.0001, 0.0001, 0)
)

val_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    train_val_test_split=(0.0001, 0.0001, 0)
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

backbone = backbones.load("efficientnet_b1")
net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['conv_head'],
    device=device,
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=2,
    target_embed_dimension=2,
    patchsize=3,
    embedding_size=None,
    meta_epochs=1,
    aed_meta_epochs=0,
    gan_epochs=1,
    noise_std=0.05,
    dsc_layers=2,
    dsc_hidden=8,
    dsc_margin=0.85,
    dsc_lr=0.001,
    auto_noise=0,
    train_backbone=True,
    cos_lr=False,
    lr=0.001,
    pre_proj=0,
    proj_layer_type=0,
    mix_noise=1,
)
models_dir = f"models/{time()}"
dataset_name = "rsna_breast_cancer"
net.set_model_dir(models_dir, dataset_name)

net.train(train_loader, val_loader)
