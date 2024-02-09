import os
from datetime import datetime

import dotenv
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet
from src import backbones


def pretrain_backbone(backbone, train_loader, val_loader, epochs=10):
    model = nn.Sequential(backbone, torch.nn.Linear(1000, 1, bias=False))
    model.to(device).train()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    lr_lambda = lambda epoch: 0.9 * epoch
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
    for epoch in range(epochs):
        losses = []
        for data in tqdm(train_loader):
            images = data['image'].to(device)
            labels = data['anomaly'].to(device).float()
            optimizer.zero_grad()
            output = model(images)
            output = torch.squeeze(output)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach().cpu().item())

        preds = []
        labels_ = []
        for data in tqdm(val_loader):
            images = data['image'].to(device)
            labels = data['anomaly'].to(device)
            with torch.no_grad():
                output = model(images)
            preds.append(output)
            labels_.append(labels)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        labels = torch.cat(labels_, dim=0).cpu().numpy()
        preds_bin = np.where(preds >= 0.5, 1, 0)
        f1 = f1_score(labels, preds_bin)
        auc = roc_auc_score(labels, preds)
        mse = mean_squared_error(labels, preds)


dotenv.load_dotenv()
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

pretrain_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    num_images=(128, 0, 128, 0),
    resize=(128, 128),
    rotate_degrees=20,
    v_flip_p=0.5,
    h_flip_p=0.25,
    noise_std=0.05,
)

train_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    num_images=(128, 0, 0, 0),
    resize=(128, 128),
    rotate_degrees=20,
    v_flip_p=0.5,
    h_flip_p=0.25,
    noise_std=0.05,
)

val_ds_healthy = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(128, 128, 0, 0),
    resize=(128, 128)
)
val_ds_cancer = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(0, 0, 128, 128),
    resize=(128, 128),
    rotate_degrees=0,
    v_flip_p=0.,
    h_flip_p=0.,
    noise_std=0.5,
    contrast_range=(1.0, 1.0),
    brightness_range=(1.0, 1.0)
)

val_ds = ConcatDataset([val_ds_healthy, val_ds_cancer])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
pretrain_loader = DataLoader(pretrain_ds, batch_size=32, shuffle=True)

backbone = backbones.load("resnet50")

#pretrain_backbone(backbone, pretrain_loader, val_loader)

net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['layer2', 'layer3'],
    device=device,
    input_shape=(3, 128, 128),
    pretrain_embed_dimension=512,
    target_embed_dimension=512,
    patchsize=3,
    embedding_size=None,
    meta_epochs=100,
    aed_meta_epochs=1,
    gan_epochs=5,
    noise_std=0.01,
    dsc_layers=7,
    dsc_hidden=32,
    dsc_margin=0.5,
    dsc_lr=0.005,
    train_backbone=True,
    cos_lr=False,
    lr=0.005,
    pre_proj=1,
    proj_layer_type=1,
    mix_noise=1,
    norm_disc_out=False,
)
models_dir = f'models/{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
dataset_name = "rsna_breast_cancer"
net.set_model_dir(models_dir, dataset_name)
net.save_model_params()

net.train(train_loader, val_loader)
