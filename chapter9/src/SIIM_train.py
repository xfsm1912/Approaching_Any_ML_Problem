# SIIM_train.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from SIIMDataset import SIIMDataset

# training csv file path
TRAINING_CSV = '../input/train_pneumothorax.csv'

# training and test batch sizes
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4

# number of epochs
EPOCHS = 10

# define the encoder for U-Net  #
# check: https://github.com/qubvel/segmentation_models.pytorch  #
# for all supported encoders
ENCODER = 'resnet18'

# we use imagenet pretrained weights for the encoder
ENCODER_WEIGHTS = 'imagenet'

# train on gpu
DEVICE = 'cuda'

if __name__ == '__main__':
    # read the training csv file
    df = pd.read_csv(TRAINING_CSV)

    # split data into training and validation
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )

    # training and validation images lists/arrays
    training_images = df_train.image_id.values
    validation_images = df_valid.image_id.values

    # fetch unet model from segmentation models
    # with specified encoder architecture
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None,
    )
    # segmentation model provides you with a preprocessing
    # function that can be used for normalizing images
    # normalization is only applied on images and not masks
    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )

    # send model to device
    model.to(DEVICE)

    # init training dataset
    # transform is True for training data
