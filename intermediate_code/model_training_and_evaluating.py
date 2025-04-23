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



# --- DEFINING EITHER CNN / VGG16 TRAINING + EVALUATION PIPELINE FOR A GIVEN COMBINATION OF HYPERPARAMETERS ---
def train_and_evaluate_model(base_dataset_path, model_name, loss_function, learning_rate, batch_size):
    
    start_time = time.time()
    best_validation_error = float('inf')
    best_model_state = None # To store the state of the best model

    # Properly Loading in All Data ..............................................................
    training_dataset = None
    testing_dataset = None
    for child in os.listdir(base_dataset_path):
        child_path = os.path.join(base_dataset_path, child)
        if child.startswith('train'):
            if os.path.isfile(child_path):
                if child.endswith('.mat'):
                    training_dataset = raw_data_to_tensor_dataset(child_path, batch_size)
            # elif os.path.isdir(child_path)
        elif child.startswith('test'):
            if os.path.isfile(child_path):
                if child.endswith('.mat'):
                    testing_dataset = raw_data_to_tensor_dataset(child_path, batch_size)
            # elif os.path.isdir(child_path)
            
    if training_dataset is None or testing_dataset is None:
        raise ValueError("Training or testing dataset not found.")

    # Defining the (constant) loss function......................................................
    if loss_function=='cross-entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function=='mean-squared':
        criterion = nn.MSELoss()
    elif loss_function=='multi-margin':
        criterion = nn.MultiMarginLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")
    
    # Model Initialization ......................................................................
    if model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')                                  # downloading a vgg16 with pre-trained weights
        # for param in model.features.parameters():
        #     param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 11)                                      # adjusting for the now 11 classes
    elif model_name == 'original':
        model = CNN()                                                                  # reseting the model + optimizer each fold
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    best_model = model # Keep track of the model instance.

    # Training + Determining Training + Validation Error ........................................
    training_error = 0
    validation_error = 0

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(training_dataset)):
        print(f"Training fold {fold + 1}/{k}...")

        # Separating Full Training Dataset into Training + Validation ...........................
        training_subset = Subset(training_dataset, train_idx)                          # creating subsets + official dataloaders of the full training_dataset for cross validation
        val_subset = Subset(training_dataset, val_idx)
        
        training_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


        # (CNN) Model Initialization ............................................................
        if model_name == 'original':
            model = CNN()                                                              # reseting the model + optimizer each fold.  Create a new instance for each fold.
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # (optimizers adjust model parmeters to minimize the loss function!! it doesn't actually do the backprop, but updates the model using the gradients calculated during that)


        # Training Loop .........................................................................
        model.train()                                                                  # setting the model to training mode!
        for epoch in range(5):                                                         # at a high level... for each testing + validation set combo, you train x epochs amount of times, and then you evaulate the validation accuracy
            for inputs, labels in training_loader:
                optimizer.zero_grad()                                                  # zero-ing out / restarting the gradients ... (BUT THE MODEL WEIGHTS ARE NEVER ZERO-ED!)
                outputs = model(inputs)                                                # running all the input data through to get some predictions
                loss = criterion(outputs, labels)                                      # calculating loss wrt the loss function, comparing the predictions with the actuals
                loss.backward()                                                        # performs backprop
                optimizer.step()                                                       # updates the model parameters using what was found during backprop


        # Training Evaluation ....................................................................
        model.eval()                                                                   # setting the model to evaluation mode!
        correct_train, total_train = 0, 0
        with torch.no_grad():                                                          # turns off gradient checking for memory-saving purposes
            for inputs, labels in training_loader:                                     # loops through all the batches
                outputs = model(inputs)                                                # gets the raw predictions back
                _, predicted = torch.max(outputs, 1)                                   # gets the class with the highest score
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        training_error_fold = (1 - (correct_train/total_train))

        # Validation Evaluation ..................................................................
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        validation_error_fold = (1 - (correct_val/total_val))


        # Saving the Best Model!
        if validation_error_fold < best_validation_error:
            best_validation_error = validation_error_fold
            best_model_state = model.state_dict()                                      # saving the model's weights
        
        print(f"Fold {fold + 1}: Training Error = {training_error_fold*100:.2f}%, Validation Error = {validation_error_fold*100:.2f}%")
        training_error += training_error_fold
        validation_error += validation_error_fold


    training_error /= k
    validation_error /= k


    # Testing Evaluation ........................................................................
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    testing_error = 0
    correct_test, total_test = 0, 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in testing_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    testing_error = (1 - (correct_test/total_test))

    end_time = time.time()
    print(end_time - start_time)

    # Saving the Overall Best Model!
    if best_model_state is not None:
        # best_model.load_state_dict(best_model_state)
        if model_name == 'original':
            path = os.path.join("..", "current-models", "original", f"{loss_function}-{str(learning_rate)}-{str(batch_size)}.pth")
            torch.save(best_model_state, path)
        elif model_name == 'vgg16':
            path = os.path.join("..", "current-models", "vgg16", f"{loss_function}-{str(learning_rate)}-{str(batch_size)}.pth")
            torch.save(best_model_state, path)
    else:
        print("Warning: No best model state found. Saving the last model.")

    return training_error*100, validation_error*100, testing_error*100



# --- DEFINING EASY-USE HYPERPARAMETER GRIDSEARCHING FUNCTION ---
def grid_search_hyperparameters(base_dataset_path, model_name, loss_function_list, learning_rate_list, batch_size_list):

    # {key: value} == {(loss_function, learning_rate, batch_size): 
    #                         (training_error, validation_error, testing_error))}
    hyperparameter_performance_results = {}

    i = 0
    for loss_function in loss_function_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                print("iteration: ", i)
                hyperparameter_performance_results[(loss_function, learning_rate, batch_size)] = train_and_evaluate_model(base_dataset_path, model_name, loss_function, learning_rate, batch_size)
                i += 1

    return hyperparameter_performance_results