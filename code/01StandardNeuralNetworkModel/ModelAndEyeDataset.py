import os
from PIL import Image

import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset

# Define the StandardResNeXt class, a custom neural network module
class StandardResNeXt(nn.Module):
    def __init__(self, num_classes=2):
        super(StandardResNeXt, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)  # Use a pre-trained ResNeXt model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace the fully connected layer

    def forward(self, x):
        return self.model(x)  # Define the forward pass

# Define the StandardResNest class, another custom neural network module
class StandardResNest(nn.Module):
    def __init__(self, num_classes=2):
        super(StandardResNest, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Use a pre-trained ResNet model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace the fully connected layer

    def forward(self, x):
        return self.model(x)  # Define the forward pass

# Define the EyeDataset class, a custom dataset for eye images
class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory for the dataset
        self.transform = transform  # Transformations to be applied on images
        # List of image file paths and corresponding labels
        self.images = [os.path.join(root_dir, label, img) for label in os.listdir(root_dir) for img in
                       os.listdir(os.path.join(root_dir, label))]
        self.labels = [int(label) for label in os.listdir(root_dir) for _ in os.listdir(os.path.join(root_dir, label))]

    def __len__(self):
        return len(self.images)  # Return the total number of images

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)  # Open the image
        label = self.labels[idx]  # Get the label

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label  # Return the image and its label
