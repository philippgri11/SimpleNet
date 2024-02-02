import dataclasses
from datetime import datetime
from src.common import BackboneSetting

# sweep_configuration = {
#     'method': 'bayes',
#     'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
#     'metric': {'goal': 'maximize', 'name': 'auroc'},
#     'parameters': {
#         'backbone': {'values': [dataclasses.asdict(BackboneSetting('resnet50', ['layer2', 'layer3']))]},
#         'pretrain_embed_dimension': {'values': [16, 32, 64, 128, 256]},
#         'projection_dimension': {'values': [16, 32, 64, 128, 256]},
#         'patch_size': {'values': [3, 5, 7, 9]},
#         'meta_epochs': {'value': 25},
#         'aed_meta_epochs': {'value': 5},  # used for cos_lr scheduler, but needs to be an int allways
#         'gan_epochs': {'value': 2},
#         'noise_std': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.5},
#         'dsc_layers': {'values': [1, 2, 4]},
#         'dsc_hidden': {'values': [8, 16, 32, 64, 128, 256]},
#         'dsc_margin': {'values': [0.25, 0.5, 0.75]},
#         'dsc_lr': {'values': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},  # LR for Discriminator
#         'auto_noise': {'value': 0},  # scheint ein sinnloser Parameter zu sein
#         'train_backbone': {'value': True},
#         'cos_lr': {'value': True},
#         'lr': {'values': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},  # LR for Projection and Backbone
#         'pre_proj': {'values': [0, 1, 2]},  # Number of Layers for Projection
#         'proj_layer_type': {'values': [0, 1]},  # if > 1 then relu is added to all but the last layer of Projection
#         'mix_noise': {'value': 1},
#     }
# }

sweep_configuration = {
    'method': 'grid',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'parameters': {
        'backbone': {'values': [dataclasses.asdict(BackboneSetting('resnet50', ['layer2', 'layer3']))]},
        'batch_size': {'value': 4},
        'pretrain_embed_dimension': {'value': 1536},
        'projection_dimension': {'value': 1536},
        'image_size': {'value': (3, 1024, 1024)},
        'patch_size': {'value': 3},
        'meta_epochs': {'value': 50},
        'aed_meta_epochs': {'value': 1},  # used for cos_lr scheduler, but needs to be an int allways
        'gan_epochs': {'value': 4},
        'noise_std': {'value': 0.015},
        'dsc_layers': {'value': 2},
        'dsc_hidden': {'value': 1024},
        'dsc_margin': {'value': 0.5},
        'dsc_lr': {'value': 0.0002},  # LR for Discriminator
        'auto_noise': {'value': 0},  # scheint ein sinnloser Parameter zu sein
        'train_backbone': {'value': True},
        'cos_lr': {'value': True},
        'lr': {'value': 0.0001},  # LR for Projection and Backbone
        'pre_proj': {'value': 1},  # Number of Layers for Projection
        'proj_layer_type': {'value': 1},  # if > 1 then relu is added to all but the last layer of Projection
        'mix_noise': {'value': 1},
        'pretrain_backbone': {'value': True},
        'pretrain_epochs': {'value': 20}
    }
}
