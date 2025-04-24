import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from PIL import Image
import os
import random
import torch.nn.functional as F

from model_training_and_evaluating import grid_search_hyperparameters


def create_plot(error_type):

    data = {
        '0': { 'batch_size': [],
        'training_error': [],
        'validation_error': [],
        'testing_error': [] 
        },
        '0.01': { 'batch_size': [],
        'training_error': [],
        'validation_error': [],
        'testing_error': [] 
        },
        '0.1': { 'batch_size': [],
        'training_error': [],
        'validation_error': [],
        'testing_error': [] 
        },
        '1': { 'batch_size': [],
        'training_error': [],
        'validation_error': [],
        'testing_error': [] 
        }
    }

    for key, values in grid_search_results.items():
        if key[0] == error_type:
            lr = str(key[1])
            data[lr]['batch_size'].append(key[2])
            data[lr]['training_error'].append(values[0])
            data[lr]['validation_error'].append(values[1])
            data[lr]['testing_error'].append(values[2])

    colors = {
        '0': "red", 
        '0.01': "orange", 
        '0.1': "green", 
        '1': "purple"
    }

    # Beginning to Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.figure(figsize=(10, 8))

    for key, value in data.items():
        plt.plot(value['batch_size'], value['training_error'],   color=colors[key], linestyle='-',  label=f"Training, LR: {key}")
        plt.plot(value['batch_size'], value['validation_error'], color=colors[key], linestyle='--', label=f"Validation, LR: {key}")
        plt.plot(value['batch_size'], value['testing_error'],    color=colors[key], linestyle=':',  label=f"Testing, LR: {key}")

    plt.title(f"For {'-'.join(word.capitalize() for word in error_type.split('-'))}: Error Rate as a Function of Batch Size and Learning Rate", fontsize=20, pad=15)
    plt.xlabel(f"Batch Size", fontsize=15, labelpad=10)
    plt.ylabel(f"Error Rate", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', labelsize=15)
    plt.minorticks_on()
    plt.grid(which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.grid(which='minor', axis='both', linestyle='--', linewidth=0.1) 
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()






# --------------------------------------------- for my ORIGINAL DATASET ---------------------------------------------
# Defining the Hyperparameters Whose Combination I'm Optimizing
lf_list = ['cross-entropy', 'mean-squared', 'multi-margin']
lr_list = [0, 0.01, 0.1, 1]
bs_list = [32, 64, 128, 256]

grid_search_results = grid_search_hyperparameters('original', lf_list, lr_list, bs_list)
create_plot('cross-entropy')
create_plot('mean-squared')
create_plot('multi-margin')

# ---------------------------------------------- for the VGG16 DATASET ----------------------------------------------
# Again defining the Hyperparameters Whose Combination I'm Optimizing
loss_function_list = ['cross-entropy']
learning_rate_list = [0.01, 0.1]
batch_size_list = [128, 256]

grid_search_results = grid_search_hyperparameters('vgg16', lf_list, lr_list, bs_list)
create_plot('cross-entropy')
create_plot('multi-margin')