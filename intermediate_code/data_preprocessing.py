import os
import time
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

import cv2
import scipy
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

import torchvision
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms


# --- DEFINING RAW DATA LOADING ---
def raw_data_to_tensor_dataset(raw_data_path, batch_size):
    if raw_data_path.endswith('.mat'):
        data = scipy.io.loadmat(raw_data_path)                                     # unpacking the .mat dataset into dictionary form for easier use

        images = data['X'].transpose((3, 2, 0, 1))                                 # (height, width, num_channels, N) --> (batch_size, num_channels, height, width)
        labels = data['y'].flatten()
        labels[labels == 10] = 0                                                   # (SVHN uses the label 10 to represent the digit 0, so I'm just rewriting that)

        images_tensor = (torch.tensor(images, dtype=torch.float32) /127.5)-1       # normalizing the images to values [-1,1] to help the model learn better/faster!
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(images_tensor, labels_tensor)                      # creating the dataset (i.e. pairing images + labels together)
    
    # elif you get a set of pngs or jpgs or something

    return dataset