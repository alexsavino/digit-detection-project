import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from PIL import Image, ImageDraw
import os
import random
import torch.nn.functional as F
import cv2



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


def predict_single_image(model, image_input):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import cv2
    import numpy as np
    import os

    if isinstance(image_input, str) and os.path.isfile(image_input):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[2] == 3:
            image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            image = image_input
        image = Image.fromarray(image)
    elif isinstance(image_input, Image.Image):
        image = image_input


    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    model.eval()                                                                                            # the prediction step! everything is the same as usual
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = prediction.item()
    confidence_score = confidence.item()
    class_name = 'Non-digit' if predicted_class == 10 else str(predicted_class)

    return {
        'class': class_name,
        'class_index': predicted_class,
        'confidence': confidence_score,
        'probabilities': probabilities[0].cpu().numpy()
    }


def detect_digits_with_sliding_window(model, image_path, stride=8, scales=[1.0, 1.5, 0.75]):
    
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        original_image = image_path.copy()
        if original_image.shape[2] == 3:
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image_rgb = original_image
    elif isinstance(image_path, Image.Image):
        original_image_rgb = np.array(image_path)
        original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

    h, w = original_image.shape[:2]
    window_size = 32                                                                                        # the fixed 32x32 size!
    confidence_threshold = 0.995                                                                            # MIN REQUIRED CONFIDENCE
    
    detections = []
    
    for scale in scales:
        print(f" ... Checking scale {scale} ... ")
        scaled_width = int(w * scale)
        scaled_height = int(h * scale)
        
        if scaled_width <= window_size or scaled_height <= window_size:
            continue
            
        scaled_image = cv2.resize(original_image_rgb, (scaled_width, scaled_height))
        scaled_image_pil = Image.fromarray(scaled_image)
        

        for y in range(0, scaled_height-window_size + 1, stride):
            for x in range(0, scaled_width-window_size + 1, stride):
                window = scaled_image_pil.crop((x, y, x+window_size, y+window_size))
                
                result = predict_single_image(model, window)                                               # performing the prediction
                if result['class_index'] != 10 and result['confidence'] > confidence_threshold:             # if digit with significant confidence
                    orig_x = int(x / scale) 
                    orig_y = int(y / scale)
                    orig_w = int(window_size / scale)
                    orig_h = int(window_size / scale)
                    
                    detections.append({
                        'x': orig_x,
                        'y': orig_y,
                        'width': orig_w,
                        'height': orig_h,
                        'digit': result['class_index'],
                        'confidence': result['confidence'],
                        'scale': scale
                    })
    
    final_detections = detections
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    final_detections.sort(key=lambda det: det['x'])                                                         # sort detections by x value so as to read left-->right
    digit_string = ''.join([str(det['digit']) for det in final_detections])
    
    return final_detections, digit_string

def non_max_suppression(detections, iou_threshold=0.3):
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)                            # (sort by confidence)
    
    keep = []
    while detections:
        best = detections.pop(0)                                                                            # keep the detection with highest confidence
        keep.append(best)
        
        filtered_detections = []
        for det in detections:
            if calculate_iou(best, det) < iou_threshold:
                filtered_detections.append(det)
        detections = filtered_detections
    
    return keep

def calculate_iou(box1, box2):

    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    
    return intersection / float(box1_area + box2_area - intersection)

def visualize_save_detections(image_path, detections):
    
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        if image_path.shape[2] == 3:
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
    else:
        image = np.array(image_path)
    
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    

    for det in detections:                                                                                  # draw the detection boxes ... both bounding box + label
        draw.rectangle(
            [(det['x'], det['y']), (det['x'] + det['width'], det['y'] + det['height'])],
            outline="red", width=2
        )
        
        draw.text(
            (det['x'], det['y'] - 10),
            f"{det['digit']}",
            fill="red"
        )
    print("DETECTIONS: ", detections)
    print("TESTING: ", ''.join([str(det['digit']) for det in detections]))
    os.makedirs("graded_images", exist_ok=True)
    filename = os.path.basename(image_path)
    filename_wo_ext, ext = os.path.splitext(filename)

    save_path = os.path.join("graded_images", f"{filename_wo_ext}.png")

    plt.imshow(np.array(image_pil))
    plt.axis('off')
    plt.title(f"Detected sequence: {''.join([str(det['digit']) for det in detections])}")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return 


# MAIN PART!!
model = VGG16()
device = torch.device("cpu")
model.load_state_dict(torch.load('vgg16-cross-entropy-0.01-128.pth'))
model.to(device)
model.eval()

for image_path in ['1.png', '2.png', '3.png', '4.png', '5.png']:
    detections, digit_string = detect_digits_with_sliding_window(model, image_path)
    visualize_save_detections(image_path, detections)