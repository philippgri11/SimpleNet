import dataclasses
from datetime import datetime

from src.common import BackboneSetting

sweep_configuration = {
    'method': 'grid',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'parameters': {
        'backbone': {'value': dataclasses.asdict(BackboneSetting('resnet50', ['layer1']))},
        'batch_size': {'value': 8},
        'pretrain_embed_dimension': {'value': 576},
        'projection_dimension': {'value': 576},
        'image_size': {'value': (3, 256, 256)},
        'patch_size': {'value': 3},
        'meta_epochs': {'value': 40},
        'aed_meta_epochs': {'value': 1},  # used for cos_lr scheduler, but needs to be an int allways
        'gan_epochs': {'value': 4},
        'noise_std': {'value': 0.01},
        'dsc_layers': {'value': 3},
        'dsc_hidden': {'value': 128},
        'dsc_margin': {'value': 0.5},
        'dsc_lr': {'value': 0.001},  # LR for Discriminator
        'pre_proj_lr': {'value': 0.0001},
        'auto_noise': {'value': 0},  # scheint ein sinnloser Parameter zu sein
        'train_backbone': {'value': True},
        'cos_lr': {'value': True},
        'lr': {'value': 0.001},  # LR for Projection and Backbone
        'pre_proj': {'value': 3},  # Number of Layers for Projection
        'proj_layer_type': {'value': 9999},  # if > 1 then relu is added to all but the last layer of Projection
        'mix_noise': {'value': 1},
        # 'pretrain_backbone': {'value': False},
        # 'pretrain_epochs': {'value': 25},
        # 'pretrain_lr': {'value': 0.001},
        # 'pretrain_epochs_out_layer': {'value': 10}
    }
}
