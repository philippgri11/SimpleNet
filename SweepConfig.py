import dataclasses
from datetime import datetime

from src.common import BackboneSetting

sweep_configuration = {
    'method': 'grid',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'parameters': {
        'backbone': {'value': dataclasses.asdict(BackboneSetting('wideresnet50', ['layer1', 'layer2']))},
        'batch_size': {'value': 16},
        'pretrain_embed_dimension': {'value': 768},
        'projection_dimension': {'value': 768},
        'image_size': {'value': (3, 224, 224)},
        'patch_size': {'value': 3},
        'meta_epochs': {'value': 50},
        'aed_meta_epochs': {'value': 5},  # used for cos_lr scheduler, but needs to be an int allways
        'gan_epochs': {'value': 5},
        'noise_std': {'value': 0.03},
        'dsc_layers': {'value': 2},
        'dsc_hidden': {'value': 512},
        'dsc_margin': {'value': .3},
        'dsc_lr': {'value': 0.0002},  # LR for Discriminator
        'pre_proj_lr': {'value': 0.0001},
        'auto_noise': {'value': 0},  # scheint ein sinnloser Parameter zu sein
        'train_backbone': {'value': False},
        'cos_lr': {'value': False},
        'lr': {'value': 0.0001},  # LR for Projection and Backbone
        'pre_proj': {'value': 2},  # Number of Layers for Projection
        'proj_layer_type': {'value': 999},  # if > 1 then relu is added to all but the last layer of Projection
        'mix_noise': {'value': 1},
    }
}
