import os
import time
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import scipy
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

import torchvision
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms

from scipy.signal import convolve2d as conv2
from skimage import restoration

from intermediate_code.CNN import CNN
from intermediate_code.data_preprocessing import raw_data_to_tensor_dataset


def apply_model_to_data(model_path, data_path, model_name, batch_size=None, label=None):

    # Loading in the Model ......................................................................
    if model_name == 'original':
        model = CNN() 
    elif model_name == 'vgg16':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 11)


    model.load_state_dict(torch.load(model_path))                                      # loading in the weights of the model!
    model.eval()                                                                       # setting the model to evaluation mode

    # If you have a single image
    if label:
        dataset = raw_data_to_tensor_dataset(data_path, batch_size, label)
        dataloader = DataLoader(dataset, batch_size=1)                                 # batch_size=1 for single image
        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)                             # get the predicted class

        return predictions, labels

    # If you have a set of images
    else: 
        dataset = raw_data_to_tensor_dataset(data_path, batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted)
                all_labels.append(labels)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(labels)
        
        return all_predictions, all_labels