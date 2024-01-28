import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage import io, measure, morphology
from skimage.filters import threshold_otsu
from skimage.transform import resize
from tqdm import tqdm


def get_padding(image, size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    imsize = image.shape
    h_padding = (size[0] - imsize[0]) / 2
    v_padding = (size[1] - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    padding = ((int(l_pad), int(r_pad)), (int(t_pad), int(b_pad)))

    return padding


def process_and_crop_image(image_path, output_folder, margin=10, padding=(1024, 1024), save_resolutions=None):
    try:
        image = io.imread(image_path, as_gray=True)

        thresh = threshold_otsu(image)
        binary = image > thresh
        cleaned = morphology.remove_small_objects(binary, min_size=150)
        filled_image = binary_fill_holes(cleaned)
        label_img = measure.label(filled_image)
        regions = measure.regionprops(label_img)

        if not regions:
            raise Exception("Could not find any regions")

        region_max = max(regions, key=lambda r: r.area)

        minr, minc, maxr, maxc = region_max.bbox
        width, height = image.shape
        minr = max(0, minr - margin)
        minc = max(0, minc - margin)
        maxr = min(width, maxr)
        maxc = min(height, maxc)

        cropped_image = image[minr:maxr, minc:maxc]
        cropped_image = np.pad(cropped_image, get_padding(cropped_image, padding), 'constant')

        filename = os.path.basename(image_path)
        for resultion in save_resolutions:
            if cropped_image.shape != resultion:
                resized_image = resize(cropped_image, resultion) * 255.
            else:
                resized_image = cropped_image
            cropped_image_path = os.path.join(output_folder, str(resultion[0]), f"{filename}")
            io.imsave(cropped_image_path, resized_image.astype(np.uint8))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


output_folder = 'data/rsna_pp/'
input_folder = 'data/rsna_basic_1024'

processed_images_paths = []
df = pd.read_csv('train.csv')
save_resolutions = [(256, 256), (512, 512), (1024, 1024)]

for resolution in save_resolutions:
    os.makedirs(os.path.join(output_folder, str(resolution[0])), exist_ok=True)

for idx, row in tqdm(df.iterrows()):
    image_name = str(row['patient_id']) + "_" + str(row['image_id']) + ".png"
    image_path = os.path.join(input_folder, image_name)
    process_and_crop_image(image_path, output_folder, save_resolutions=save_resolutions)
