from datetime import datetime
from common import BackboneSetting

sweep_configuration = {
    'method': 'grid',
    'name': 'HyperparameterSearch' + datetime.now().strftime("%d/%m/%Y, %H:%M"),
    'parameters': {
        'backbone': {'values': [BackboneSetting('efficientnet_b1', ['conv_head'])]},
        'input_shape': {'value': (3, 224, 224)},
        'pretrain_embed_dimension': {'values': [64]},
        'target_embed_dimension': {'values': [64]},
        'patchsize': {'values': [3]},
        'meta_epochs': {'value': 25},
        'aed_meta_epochs': {'value': 0},
        'gan_epochs': {'values': [2]},
        'noise_std': {'values': [0.05]},
        'dsc_layers': {'values': [2]},
        'dsc_hidden': {'values': [32]},
        'dsc_margin': {'values': [0.5]},
        'dsc_lr': {'values': [0.0001]},
        'auto_noise': {'value': 0},
        'train_backbone': {'value': True},
        'cos_lr': {'value': False},
        'pre_proj': {'value': 0},
        'proj_layer_type': {'value': 0},
        'mix_noise': {'value': 1},
    }
}
