import os
import random

import dotenv
import numpy as np
import torch.cuda
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from SweepConfig import sweep_configuration
from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet
from src import backbones

random.seed(42)

dotenv.load_dotenv()
device = 'cuda'

CANCER_CNT = 1158


def pretrain_backbone(backbone, run, train_loader, val_loader, pre_epochs_out_layer=5, epochs=10, pos_images=-1, lr=0.001):
    model = nn.Sequential(backbone, torch.nn.Linear(1000, 1, bias=False))
    model.to(device).train()

    if pos_images > 0:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_images / len(train_loader)]).to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=5e-4)
    lr_lambda = lambda epoch: 0.9
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

    for epoch in range(epochs + pre_epochs_out_layer):

        if epoch < pre_epochs_out_layer:
            # Einfrieren aller Backbone-Layer
            for param in model[0].parameters():
                param.requires_grad = False
        else:
            # Entfrieren aller Backbone-Layer
            if epoch == pre_epochs_out_layer:
                # Optimierer neu initialisieren, um die entfrorenen Parameter zu berücksichtigen
                optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.9)
            for param in model[0].parameters():
                param.requires_grad = True

        losses = []
        preds = []
        labels_ = []
        for data in tqdm(train_loader):
            images = data['image'].to(device)
            labels = data['anomaly'].to(device).float()
            optimizer.zero_grad()
            output = model(images)
            preds.append(output.detach())
            labels_.append(labels.detach())
            output = torch.squeeze(output)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach().cpu().item())
        preds = torch.cat(preds, dim=0).cpu().numpy()
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        labels = torch.cat(labels_, dim=0).cpu().numpy()
        preds_bin = np.where(preds >= 0.5, 1, 0)
        f1 = f1_score(labels, preds_bin)
        auc = roc_auc_score(labels, preds)
        mse = mean_squared_error(labels, preds)
        run.log({
            'Loss/Backbone/pretrain_loss': sum(losses) / len(losses),
            'Metrics/Backbone/train/f1-score': f1,
            'Metrics/Backbone/train/auc': auc,
            'Metrics/Backbone/train/mse': mse
        }, commit=False)

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
        run.log({
            'Metrics/Backbone/val/f1-score': f1,
            'Metrics/Backbone/val/auc': auc,
            'Metrics/Backbone/val/mse': mse
        })


def train(config=None):
    with wandb.init(config=config, group='HPO') as run:
        config = wandb.config
        cancer_skip = 0
        if config.pretrain_backbone:
            pretrain_ds = BreastCancerDataset(
                img_dir=img_dir,
                meta_data_csv_path=csv_file,
                num_images=(256, 0, 256, 0),
                resize=config.image_size[1:],
                rotate_degrees=20,
                v_flip_p=0.5,
                h_flip_p=0.25,
                noise_std=0.05,
                brightness_range=(0.7, 1.),
                contrast_range=(0.7, 1.3)
            )
            cancer_skip = 256
            pretrain_loader = DataLoader(pretrain_ds, batch_size=config.batch_size, shuffle=True)

        train_ds = BreastCancerDataset(
            img_dir=img_dir,
            meta_data_csv_path=csv_file,
            num_images=(256, 0, 0, 0),
            resize=config.image_size[1:],
            rotate_degrees=20,
            v_flip_p=0.5,
            h_flip_p=0.25,
            noise_std=0.05,
            brightness_range=(0.7, 1.),
            contrast_range=(0.7, 1.3)
        )

        val_ds = BreastCancerDataset(
            img_dir=img_dir,
            meta_data_csv_path=csv_file,
            split=DatasetSplit.VAL,
            # num_images=(256, 256, CANCER_CNT - cancer_skip, cancer_skip),
            num_images=(256, 256, 256, cancer_skip),
            resize=config.image_size[1:]

        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

        backbone = backbones.load(config.backbone['backbone_name'])

        if config.pretrain_backbone:
            pretrain_backbone(backbone, run, pretrain_loader, val_loader, epochs=config.pretrain_epochs,
                              pos_images=cancer_skip, lr=config.pretrain_lr, pre_epochs_out_layer=config.pretrain_epochs_out_layer)

        net = SimpleNet(device, wandb_run=run)
        net.load(
            backbone=backbone,
            layers_to_extract_from=config.backbone['backbone_layers'],
            device=device,
            input_shape=config.image_size,
            pretrain_embed_dimension=config.pretrain_embed_dimension,
            target_embed_dimension=config.projection_dimension,
            patchsize=config.patch_size,
            meta_epochs=config.meta_epochs,
            aed_meta_epochs=config.aed_meta_epochs,
            gan_epochs=config.gan_epochs,
            noise_std=config.noise_std,
            dsc_layers=config.dsc_layers,
            dsc_hidden=config.dsc_hidden,
            dsc_margin=config.dsc_margin,
            dsc_lr=config.dsc_lr,
            auto_noise=config.auto_noise,
            train_backbone=config.train_backbone,
            cos_lr=config.cos_lr,
            pre_proj=config.pre_proj,
            proj_layer_type=config.proj_layer_type,
            mix_noise=config.mix_noise,
            norm_disc_out=config.norm_disc_out
        )
        models_dir = f'models/{run.name}'
        dataset_name = "rsna_breast_cancer"
        net.set_model_dir(models_dir, dataset_name)
        net.save_model_params()

        run.watch(net)
        net.train(train_loader, val_loader)
    run.unwatch(net)
    del train_loader, val_loader, backbone, net
    torch.cuda.empty_cache()


img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

sweep_id = wandb.sweep(sweep=sweep_configuration, project=os.environ['PROJECT_NAME'])
wandb.agent(sweep_id, function=train)
