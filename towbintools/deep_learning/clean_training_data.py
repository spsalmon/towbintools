from utils.dataset import TilesDatasetFly, TilesDataset
from towbintools.towbintools.deep_learning.augmentation import get_mean_and_std, get_training_augmentation, get_validation_augmentation, grayscale_to_rgb, get_prediction_augmentation
import os
from time import perf_counter

from pytorch_toolbelt.inference.tiles import ImageSlicer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
import pytorch_toolbelt.losses as L
import re


from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryF1Score
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import albumentations as albu
from towbintools.foundation import image_handling, binary_image
from utils.loss import FocalTverskyLoss
from tqdm import tqdm
from tifffile import imwrite
from joblib import Parallel, delayed
import cv2

from architectures import NestedUNet
import pretrained_microscopy_models as pmm

raw_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/raw/"
mask_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/analysis/ch1_seg/"

def extract_seq_number(filename):
    # Define a regular expression pattern to match the number after "Seq"
    pattern = r'Seq(\d+)'

    # Use re.search to find the first occurrence of the pattern in the filename
    match = re.search(pattern, filename)

    # Check if a match was found
    if match:
        # Extract and return the matched number as an integer
        seq_number = int(match.group(1))
        return seq_number
    else:
        # Return None if no match was found
        return 0
    
raw_paths = sorted([os.path.join(raw_dir, file) for file in os.listdir(raw_dir)], key=extract_seq_number)
mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)], key=extract_seq_number)

# Create DataFrames
raw_df = pd.DataFrame({'raw_path': raw_paths})
mask_df = pd.DataFrame({'mask_path': mask_paths})

# Assuming the sequence number is the same for both raw and mask files,
# we can extract it from the paths and create a common column for merging
raw_df['seq_number'] = raw_df['raw_path'].apply(extract_seq_number)
mask_df['seq_number'] = mask_df['mask_path'].apply(extract_seq_number)

# Merge DataFrames based on the seq_number
merged_df = pd.merge(raw_df, mask_df, on='seq_number')
merged_df = merged_df.dropna(how='any')

raw_paths = merged_df['raw_path'].values.tolist()
mask_paths = merged_df['mask_path'].values.tolist()

print(raw_paths[-5:])
print(mask_paths[-5:])


cleaned_raw_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/cleaned/raw/"
cleaned_mask_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/cleaned/ch1_seg/"
os.makedirs(cleaned_raw_dir, exist_ok=True)
os.makedirs(cleaned_mask_dir, exist_ok=True)

def clean_training_zstack(raw_path, mask_path):
    try:
        raw = image_handling.read_tiff_file(raw_path)
        mask = image_handling.read_tiff_file(mask_path)

        # look at all the planes of the mask get indexes of completely black planes
        black_planes = []

        for i, plane in enumerate(mask):
            if np.max(plane) == 0:
                black_planes.append(i)

        # if there are black planes, remove them from the raw and mask

        if len(black_planes) > 0:
            raw = np.delete(raw, black_planes, axis=0)
            mask = np.delete(mask, black_planes, axis=0)
        
        # look at the remaining planes and remove those with more than 2 connected components

        print('remaining planes', mask.shape[0])
        if mask.shape[0] == 0:
            return
        planes_to_remove = []
        for i, plane in enumerate(mask):
            nb_labels, labels = cv2.connectedComponents(plane)
            if nb_labels > 2:
                planes_to_remove.append(i)

        if len(planes_to_remove) > 0:
            raw = np.delete(raw, planes_to_remove, axis=0)
            mask = np.delete(mask, planes_to_remove, axis=0)

        # save all remaining planes into individual tiffs
        if mask.shape[0] > 0:
            for i, plane in enumerate(mask):
                # binary closing fill bright holes and median filter the mask
                plane = cv2.morphologyEx(plane, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
                plane = binary_image.fill_bright_holes(raw, plane, 1)
                plane = cv2.medianBlur(plane, 5)
                imwrite(os.path.join(cleaned_mask_dir, f"{os.path.basename(mask_path)[:-5]}_{i}.tiff"), plane.astype(np.uint8), compression="zlib")
                imwrite(os.path.join(cleaned_raw_dir, f"{os.path.basename(raw_path)[:-5]}_{i}.tiff"), raw[i], compression="zlib")
    except IndexError:
        pass

Parallel(n_jobs=32, prefer='processes')(delayed(clean_training_zstack)(raw_path, mask_path) for raw_path, mask_path in (zip(raw_paths, mask_paths)))