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
device = torch.device("cpu")


class SimpleDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# --- DEFINING THE ORIGINAL MODEL ---
class CNN(nn.Module):
    # Defines Model Layers + General Structure
    def __init__(self):
        # ("CNN" --> made up of convolutional layers, pooling layers, and fully connected layers (i.e. each node from the previous layer factors into / is connected to the fully-connected layer))
        # ("conv layer" --> uses filters to detect features in images)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)                                                                     # CONV: 3 input channels (RGB), 6 output filters,  5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)                                                                    # CONV: 6 input filters,        16 output filters, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)                                                                      # MAX-POOLING: 2x2 pool size
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                                                               # FC LAYER: 16 filters * 5x5 image size after pooling = 400 input nodes
        self.fc2 = nn.Linear(120, 84)                                                                       # FC LAYER: 120 nodes to 84
        self.fc3 = nn.Linear(84, 11)                                                                        # OUTPUT: 84 nodes to 11 classes (digits 0-9 + "no digit" class)

    # Defines How Input Data Flows Through Layers During Prediction, (effectively telling the model by hand how to move data through the layers*)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                                                # applying conv. + max-pooling (using RELU!!)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                                                                             # flattening the tensor so it can pass through the FC layers
        x = F.relu(self.fc1(x))                                                                             # applying the FC layers (again using RELU!!)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                                                                     # output prediction
        return x

# --- DEFINING THE VGG16 MODEL FOR FINE-TUNING ---
class VGG16(nn.Module):
    def __init__(self, num_classes=11):                                                                     # there will be 0-9 + non-digit class, i.e. 11 classes
        super(VGG16, self).__init__()
        vgg = vgg16(pretrained=True)                                                                        # loading in the pre-trained weights
        
        vgg.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)                                        # CHANGING THE FIRST LAYER SO THAT IT ACCEPTS ONLY 32X32 IMAGES
        self.features = vgg.features
        
        self.classifier = nn.Sequential( # modifying the classifier to match our input size and number of output classes??? COME BACK!!!
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        ) 
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):                                                                                   # defines data flow ... i think?
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- DEFINING A FUNCTION TO GENERATE RANDOM NOISE IMAGES TO TRAIN FOR NON-DIGIT EXAMPLES!!! ---
def generate_non_digit_samples(transform, num_samples):
    SAMPLES = []
    LABELS = []
    
    for _ in range(num_samples//3):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        if transform:
            img = transform(img)
        SAMPLES.append(img)
        LABELS.append(10)
    
    for _ in range(num_samples//3):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                r = (i*8) % 256
                g = (j*8) % 256
                b = ((i+j) * 4) % 256
                img[i, j] = [r, g, b]
        img = Image.fromarray(img)
        if transform:
            img = transform(img)
        SAMPLES.append(img)
        LABELS.append(10)
    
    for _ in range(num_samples // 3):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        pattern_type = random.randint(0, 1)
        
        if pattern_type==0:                                                                               # stripe pattern*
            stripe_width = random.randint(1, 5)
            color1 = [random.randint(0, 255) for _ in range(3)]
            color2 = [random.randint(0, 255) for _ in range(3)]
            for i in range(32):
                for j in range(32):
                    if (i // stripe_width) % 2 == 0:
                        img[i,j] = color1
                    else:
                        img[i,j] = color2
        
        elif pattern_type==1:                                                                             # checkerboard random pattern
            square_size = random.randint(2, 8)
            color1 = [random.randint(0, 255) for _ in range(3)]
            color2 = [random.randint(0, 255) for _ in range(3)]
            for i in range(32):
                for j in range(32):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        img[i,j] = color1
                    else:
                        img[i,j] = color2
        
        img = Image.fromarray(img)
        if transform:
            img = transform(img)
        SAMPLES.append(img)
        LABELS.append(10)
    
    return SAMPLES, LABELS


# --- DEFINING A FUNCTION TO LOAD IN / GET READY DATA AND GENERAL TECH FOR MODEL BUILDING ---
def load_and_prepare_mb_dataset(model_type, lf, lr, bs):
    torch.manual_seed(42)                                                                                    # generating random seeds for reproducibility!
    np.random.seed(42)
    random.seed(42)

    transform = transforms.Compose([                                                                        # defining transformations... i.e. images --> tensors 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Really Loading + Generating + Organizing Data
    training = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)  # loading in the CROPPED DIGIT DATASETS ALWAYS for training!
    testing = torchvision.datasets.SVHN(root='./data',  split='test',  download=True, transform=transform)
    non_digit_train_samples, non_digit_train_labels = generate_non_digit_samples(transform, len(training)//2)          # (generating those non-digit examples as well)
    non_digit_test_samples,  non_digit_test_labels =  generate_non_digit_samples(transform, len(testing)//2)

    svhn_train_samples = []
    svhn_train_labels = []
    svhn_test_samples = []
    svhn_test_labels = []
    for i in range(len(training)):
        img, label = training[i]
        svhn_train_samples.append(img)
        svhn_train_labels.append(int(label))
    for i in range(len(testing)):
        img, label = testing[i]
        svhn_test_samples.append(img)
        svhn_test_labels.append(int(label))

    train_samples = svhn_train_samples + non_digit_train_samples                                            # combining all data to create full digit + non-digit datasets
    train_labels = svhn_train_labels + non_digit_train_labels
    test_samples = svhn_test_samples + non_digit_test_samples
    test_labels = svhn_test_labels + non_digit_test_labels
    train_dataset = SimpleDataset(train_samples, train_labels)
    test_dataset = SimpleDataset(test_samples, test_labels)


    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size                                                              # splitting into single time validation + training
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    batch_size = bs
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)                            # creating the dataloaders, i.e. things that efficiently load + batch during training
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Instantiating Literally Everything (i.e. Model, Loss, Optimizer, etc)
    if model_type == 'original':
        model = CNN()
    elif model_type == 'vgg16':
        model = VGG16()

    device = torch.device("cpu")
    model = model.to(device)
    return model, train_loader, val_loader, test_loader, device


# --- DEFINING A FUNCTION TO ACTUALLY TRAIN A MODEL ---
def train_model(model, model_type, train_loader, val_loader, test_loader, device, lf, lr):
    if lf=='cross-entropy':                                                                                 # declaring all the old model stuff wrt our hyperparam inputs
        criterion = nn.CrossEntropyLoss()
    elif lf=='mean-squared':
        criterion = nn.MSELoss()
    elif lf=='multi-margin':
        criterion = nn.MultiMarginLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_val_acc = 0.0
    associated_best_train_acc = 0.0

    history = {
        'train_acc': [],                                                                                    # stores training + validation accuracies 
        'val_acc': []
    }
    
    for epoch in range(5):
        # print(f'Epoch {epoch+1}/{num_epochs}')
        
        # MODEL IN TRAINING .................................................................................
        model.train()
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        epoch_acc = running_corrects.double() / total_samples
        history['train_acc'].append(epoch_acc.item())
        
        # MODEL IN (VALIDATION) EVALUATION ..................................................................
        model.eval()
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        epoch_acc = running_corrects.double() / total_samples
        history['val_acc'].append(epoch_acc.item())
                
        # SAVING THE BEST MODEL PERFORMANCE .................................................................
        if epoch_acc > best_val_acc:                                                                        # best = best performing on the validation set
            best_val_acc = epoch_acc
            associated_best_train_acc = history['train_acc'][-1]
            
            if model_type == 'original':
                path = os.path.join("..", "current-models", "original", f"{lf}-{str(lr)}-{str(bs)}.pth")
                torch.save(model.state_dict(), path)
            elif model_type == 'vgg16':
                path = os.path.join("..", "current-models", "vgg16", f"{lf}-{str(lr)}-{str(bs)}.pth")
                torch.save(model.state_dict(), path)
            # torch.save(model.state_dict(), 'best_digit_classifier.pth')

    return model, (1-associated_best_train_acc, 1-best_val_acc)                                             # returning ERROR RATE, not accuracy anymore


# --- DEFINING A FUNCTION TO ACTUALLY EVALUATE A MODEL on TESTING DATA EXPLICITLY ---
def evaluate_model(model, dataloader):
    # MODEL IN (TESTING) EVALUATION .........................................................................
    model.eval()
    
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    test_acc = running_corrects.double()/total_samples
    return 1-test_acc.item()


def model_development_and_evaluation_pipeline(model_type, lf, lr, bs):

    model, train_loader, val_loader, test_loader, device = load_and_prepare_mb_dataset(model_type, lf, lr, bs)       # actually TRAINING the model
    model, (train_er, best_val_er) = train_model(model, model_type, train_loader, val_loader, test_loader, device, lf, lr)
    model.load_state_dict(torch.load('best_digit_classifier.pth'))                                          # loading the BEST one
    test_er = evaluate_model(model, test_loader)

    return train_er, best_val_er, test_er

    
def grid_search_hyperparameters(model_type, lf_list, lr_list, bs_list):

    # {key: value} == {(loss_function, learning_rate, batch_size): 
    #                         (training_error, validation_error, testing_error))}
    hyperparameter_performance_results = {}

    i = 0
    for lf in lf_list:
        for lr in lr_list:
            for bs in bs_list:
                print("iteration: ", i)
                hyperparameter_performance_results[(lf, lr, bs)] = model_development_and_evaluation_pipeline(model_type, lf, lr, bs)
                # hyperparameter_performance_results[(loss_function, learning_rate, batch_size)] = train_and_evaluate_model(base_dataset_path, model_name, loss_function, learning_rate, batch_size)
                i += 1

    return hyperparameter_performance_results