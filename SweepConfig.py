import dataclasses
from datetime import datetime
from src.common import BackboneSetting

sweep_configuration = {
    'method': 'grid',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'parameters': {
        'backbone': {'values': [dataclasses.asdict(BackboneSetting('resnet50', ['avgpool']))]},
        'input_shape': {'value': (3, 512, 512)},
        'batch_size': {'value': 256},
        'pretrain_embed_dimension': {'values': [512]},
        'projection_dimension': {'values': [512]},
        'patch_size': {'values': [3]},
        'meta_epochs': {'value': 50},
        'aed_meta_epochs': {'value': 0},
        'gan_epochs': {'values': [2]},
        'noise_std': {'values': [0.05]},
        'dsc_layers': {'values': [2]},
        'dsc_hidden': {'values': [32]},
        'dsc_margin': {'values': [0.5]},
        'dsc_lr': {'values': [0.001]}, # LR for Discriminator
        'auto_noise': {'value': 0},
        'train_backbone': {'value': True},
        'cos_lr': {'value': False},
        'lr': {'value': 0.001}, # LR for Projection and Backbone
        'pre_proj': {'value': 0},
        'proj_layer_type': {'value': 0},
        'mix_noise': {'value': 1},
    }
}
