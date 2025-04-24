1. GENERAL:

    ALL PARTS of this project was coded entirely in PYTHON, and you must install certain packages/dependencies as well as copies of the given datasets to run my code and reproduce my results!
    I am aware of the required yml for this project; however, since I work on a Mac and not all of those specific packages were available, some versions changed a bit.*

    The version I used was Python 3.10; download this specific version here. 
    ... https://www.python.org/downloads/

    I also used pip3 (pip 24.2) and Homebrew (4.4.20) as package managers. 
    ... https://packaging.python.org/en/latest/tutorials/installing-packages/ 
    ... https://docs.brew.sh/Installation



2. REQUIREMENTS:

    To run the project, you need Python installed along with the following libraries:

    * numpy .......................... https://numpy.org/install/
    * matplotlib ..................... https://matplotlib.org/stable/install/index.html
    * OpenCV ......................... https://opencv.org/releases/
    * Pillow ......................... https://pypi.org/project/pillow/
    * PyTorch ........................ https://pytorch.org/

    These are the additional built-in libraries that I make use of:
    * random
    * os
    * sys
    * time

    These are the import statements used throughout the project for your reference; (aliases remain consistent across all files):
    # ...general...
    * import numpy as np
    * from PIL import Image, ImageDraw

    # ...PyTorch-related...
    * import torch
    * import torch.nn as nn
    * import torch.optim as optim
    * import torchvision
    * import torchvision.transforms as transforms
    * from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
    * from torchvision.models import vgg16
    * import torch.nn.functional as F


    # ...for readability and development and plot-making...
    * import matplotlib.pyplot as plt
    * from matplotlib import rc
    * from matplotlib import colors



3. THE CODE: ITS ORGANIZATION + HOW TO RUN IT

    In order to run this code, you can...
        A. Clone/download this respository
        B. Ensure you have installed the necessary packages/dependencies
        C. Everything related to model creation is contained within 'model_development_and_optimization.py' and 'model_training_and_evaluating.py'
        D. The optimal models themselves have been hand saved as 'original-cross-entropy-0.1-64.pth' and 'vgg16-cross-entropy-0.01-128.pth'
        D. Simply run the code run.py! This will generate the required graded images!