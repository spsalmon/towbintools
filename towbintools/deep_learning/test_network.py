from utils.dataset import TilesDatasetFly, TilesDataset
from towbintools.towbintools.deep_learning.augmentation import get_mean_and_std, get_training_augmentation, get_validation_augmentation, grayscale_to_rgb
import os
from time import perf_counter

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
import pytorch_toolbelt.losses as L


from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryF1Score
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import albumentations as albu
from towbintools.foundation import image_handling
from utils.loss import FocalTverskyLoss

from architectures import NestedUNet
import pretrained_microscopy_models as pmm


database_csv = "/mnt/external.data/TowbinLab/plenart/20221020_Ti2_10x_green_bacteria_wbt150_small_chambers_good/analysis/report/analysis_filemap.csv"

image_column = 'raw'
mask_column = 'analysis/ch2_seg'

database = pd.read_csv(database_csv).dropna(subset=[mask_column])
database = database.dropna(subset=[image_column])

# pick 10000 random images
random_database = database.sample(n=50000, random_state=42)

# get the images that are not in the random database
database = database[~database['raw'].isin(random_database['raw'])]


