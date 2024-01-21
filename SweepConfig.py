import dataclasses
from datetime import datetime
from src.common import BackboneSetting

sweep_configuration = {
    'method': 'bayes',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'metric': {'goal': 'maximize', 'name': 'auroc'},
    'parameters': {
        'backbone': {'values': [dataclasses.asdict(BackboneSetting('resnet50', ['layer2', 'layer3']))]},
        'pretrain_embed_dimension': {'values': [16, 32, 64, 128, 256, 512]},
        'projection_dimension': {'values': [16, 32, 64, 128, 256, 512]},
        'patch_size': {'values': [3, 5, 7, 9]},
        'meta_epochs': {'value': 25},
        'aed_meta_epochs': {'value': 5},  # used for cos_lr scheduler, but needs to be an int allways
        'gan_epochs': {'value': 2},
        'noise_std': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.5},
        'dsc_layers': {'values': [1, 2, 4]},
        'dsc_hidden': {'values': [8, 16, 32, 64, 128, 256]},
        'dsc_margin': {'values': [0.25, 0.5, 0.75]},
        'dsc_lr': {'min': 0.00005, 'max': 0.1},  # LR for Discriminator
        'auto_noise': {'value': 0},  # scheint ein sinnloser Parameter zu sein
        'train_backbone': {'value': True},
        'cos_lr': {'value': True},
        'lr': {'min': 0.00005, 'max': 0.1},  # LR for Projection and Backbone
        'pre_proj': {'values': [0, 1, 2]},  # Number of Layers for Projection
        'proj_layer_type': {'values': [0, 1]},  # if > 1 then relu is added to all but the last layer of Projection
        'mix_noise': {'value': 1},
    }
}
