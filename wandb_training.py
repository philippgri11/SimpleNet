import os
import random
from datetime import datetime

import dotenv
import torch.cuda
from torch.utils.data import DataLoader

import wandb
from SweepConfig import sweep_configuration
from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet
from src import backbones

random.seed(42)

dotenv.load_dotenv()
device = 'cuda'
image_size = (3, 128, 128)


def train(config=None):
    global train_ds, val_ds
    with wandb.init(config=config, group='SimpleNetSingleRun') as run:
        config = wandb.config

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        backbone = backbones.load(config.backbone['backbone_name'])
        net = SimpleNet(device, wandb_run=run)
        net.load(
            backbone=backbone,
            layers_to_extract_from=config.backbone['backbone_layers'],
            device=device,
            input_shape=image_size,
            pretrain_embed_dimension=config.pretrain_embed_dimension,
            target_embed_dimension=config.projection_dimension,
            patchsize=config.patch_size,
            embedding_size=None,
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
        )
        models_dir = f'models/{run.name}'
        dataset_name = "rsna_breast_cancer"
        net.set_model_dir(models_dir, dataset_name)

        net.train(train_loader, val_loader)
    del train_loader, val_loader, backbone, net
    torch.cuda.empty_cache()


img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

train_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    num_images=(1024, 0, 0, 0),
    resize=image_size[1:]
)

val_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(128, 1024, 128, 0),
    resize=image_size[1:]
)

sweep_id = wandb.sweep(sweep=sweep_configuration, project=os.environ['PROJECT_NAME'])
wandb.agent(sweep_id, function=train)
