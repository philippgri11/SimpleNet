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
    num_images=(8192, 1152, 1030, 128)
)

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

backbone = backbones.load("resnet50")
net = SimpleNet(device)
ckpt_path = "/Users/ksoll/Documents/git/SimpleNet/models/2024_01_20_02_34/rsna_breast_cancer"
net.load_model(ckpt_path, "ckpt.pth", backbone)

scores, segmentations, labels_gt, masks_gt = net.predict(val_loader)
scores = np.squeeze(np.array(scores))
img_min_scores = scores.min(axis=-1)
img_max_scores = scores.max(axis=-1)
scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)

ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

metrics = [
    metrics.compute_imagewise_retrieval_metrics(
        scores, labels_gt, th=th
    ) for th in ths
]

f1_s = list(map(lambda x: x['f1_score'], metrics))
mses = list(map(lambda x: x['mse'], metrics))
del scores, segmentations, labels_gt, masks_gt

plt.plot(ths, f1_s)
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.show()

plt.plot(ths, mses)
plt.xlabel("Threshold")
plt.ylabel("mse")
plt.show()
