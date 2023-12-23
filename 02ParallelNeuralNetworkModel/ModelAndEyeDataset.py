import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset

# ParallelResNet
class ParallelResNet(nn.Module):
    def __init__(self):
        super(ParallelResNet, self).__init__()
        self.left_eye_subnet = models.resnet50(pretrained=True)
        self.left_eye_subnet.fc = nn.Linear(self.left_eye_subnet.fc.in_features, 128)
        self.right_eye_subnet = models.resnet50(pretrained=True)
        self.right_eye_subnet.fc = nn.Linear(self.right_eye_subnet.fc.in_features, 128)
        self.classifier = nn.Linear(256, 2)

    def forward(self, left_eye, right_eye):
        left_eye_output = self.left_eye_subnet(left_eye)

        right_eye_output = self.right_eye_subnet(right_eye)
        combined_output = torch.cat((left_eye_output, right_eye_output), dim=1)
        prediction = self.classifier(combined_output)
        return prediction
# ParallelResNeXt
class ParallelResNeXt(nn.Module):
    def __init__(self):
        super(ParallelResNeXt, self).__init__()

        self.left_eye_subnet = models.resnext50_32x4d(pretrained=True)
        self.left_eye_subnet.fc = nn.Linear(self.left_eye_subnet.fc.in_features, 128)
        self.right_eye_subnet = models.resnext50_32x4d(pretrained=True)
        self.right_eye_subnet.fc = nn.Linear(self.right_eye_subnet.fc.in_features, 128)
        self.classifier = nn.Linear(256, 2)  # Fully connected layer, used to merge the output of the left and right eyes

    def forward(self, left_eye, right_eye):
        left_eye_output = self.left_eye_subnet(left_eye)

        right_eye_output = self.right_eye_subnet(right_eye)
        combined_output = torch.cat((left_eye_output, right_eye_output), dim=1)
        prediction = self.classifier(combined_output)
        return prediction


class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = [os.path.join(root_dir, label, pair) for label in os.listdir(root_dir) for pair in os.listdir(os.path.join(root_dir, label))]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_path = self.pairs[idx]
        left_eye_path = os.path.join(pair_path, "左眼.jpg")
        right_eye_path = os.path.join(pair_path, "右眼.jpg")
        label = int(os.path.basename(os.path.dirname(pair_path)))

        left_eye = Image.open(left_eye_path)
        right_eye = Image.open(right_eye_path)

        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)

        return left_eye, right_eye, label