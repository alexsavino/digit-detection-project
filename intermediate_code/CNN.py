import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# --- DEFINING THE MODEL ---
class CNN(nn.Module):
    # Defines Model Layers + General Structure
    def __init__(self):
        # ("CNN" --> made up of convolutional layers, pooling layers, and fully connected layers (i.e. each node from the previous layer factors into / is connected to the fully-connected layer))
        # ("conv layer" --> uses filters to detect features in images)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)        # CONV: 3 input channels (RGB), 6 output filters,  5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)       # CONV: 6 input filters,        16 output filters, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)         # MAX-POOLING: 2x2 pool size
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # FC LAYER: 16 filters * 5x5 image size after pooling = 400 input nodes
        self.fc2 = nn.Linear(120, 84)          # FC LAYER: 120 nodes to 84
        self.fc3 = nn.Linear(84, 11)           # OUTPUT: 84 nodes to 11 classes (digits 0-9 + "no digit" class)

    # Defines How Input Data Flows Through Layers During Prediction, (effectively telling the model by hand how to move data through the layers*)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # applying conv. + max-pooling (using RELU!!)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                # flattening the tensor so it can pass through the FC layers
        x = F.relu(self.fc1(x))                # applying the FC layers (again using RELU!!)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # output prediction
        return x