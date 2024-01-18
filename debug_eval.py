import os

import dotenv
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import backbones
import metrics
from datasets.rsna_breast_cancer import BreastCancerDataset, DatasetSplit
from simplenet import SimpleNet

dotenv.load_dotenv()
device = 'mps'

img_dir = os.environ['IMAGE_DIR']
csv_file = os.environ['CSV_PATH']

val_ds = BreastCancerDataset(
    img_dir=img_dir,
    meta_data_csv_path=csv_file,
    split=DatasetSplit.VAL,
    num_images=(128, 64, 32, 16)
)

val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

backbone = backbones.load("resnet50")
net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=['layer2'],
    device=device,
    input_shape=(3, 256, 256),
    pretrain_embed_dimension=256,
    target_embed_dimension=256,
    patchsize=3,
    embedding_size=None,
    meta_epochs=50,
    aed_meta_epochs=5,
    gan_epochs=3,
    noise_std=0.05,
    dsc_layers=2,
    dsc_hidden=64,
    dsc_margin=0.5,
    dsc_lr=0.005,
    auto_noise=0,
    train_backbone=True,
    cos_lr=False,
    lr=0.005,
    pre_proj=1,
    proj_layer_type=1,
    mix_noise=1,
)
ckpt_path = "/Users/ksoll/Documents/git/SimpleNet/models/2024_01_18_19_52/rsna_breast_cancer/ckpt_epoch_10.pth"
net.load_model(ckpt_path)

scores, segmentations, features, labels_gt, masks_gt = net.predict(val_loader)
scores = np.squeeze(np.array(scores))
img_min_scores = scores.min(axis=-1)
img_max_scores = scores.max(axis=-1)
scores = (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-8)

ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

f1_s = [
    metrics.compute_imagewise_retrieval_metrics(
        scores, labels_gt, th=th
    )["f1_score"] for th in ths
]
del scores, segmentations, features, labels_gt, masks_gt

plt.plot(ths, f1_s)
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.show()
