import os
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset



class EyeDataset(Dataset):
    def __init__(self, root_dir, age_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = [os.path.join(root_dir, label, pair) for label in os.listdir(root_dir) for pair in
                      os.listdir(os.path.join(root_dir, label))]
        self.age_df = pd.read_excel(age_path)

        self.age_dict = dict(zip(self.age_df['id'], self.age_df['age']))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_path = self.pairs[idx]
        left_eye_path = os.path.join(pair_path, "左眼.jpg")
        right_eye_path = os.path.join(pair_path, "右眼.jpg")
        label = int(os.path.basename(os.path.dirname(pair_path)))
        id = os.path.basename(pair_path)
        age = self.age_dict.get(id, 0)  # 默认年龄为0，您可以根据实际情况进行修改
        if not isinstance(age, float):
            age = float(age)  # 如果age不是浮点数，将其转换为浮点数
        age = torch.tensor(age, dtype=torch.float32)  # 转换为PyTorch张量
        age = age.view(1)  # 修改形状为[1]

        left_eye = Image.open(left_eye_path)
        right_eye = Image.open(right_eye_path)

        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)

        return left_eye, right_eye, label, age




class ParallenResNextWithAge(nn.Module):
    def __init__(self):
        super(ParallenResNextWithAge, self).__init__()

        # 使用预训练的 ResNeXt50 网络作为基础
        self.left_eye_subnet = models.resnext50_32x4d(pretrained=True)
        # 替换分类器层以适应新的特征尺寸
        self.left_eye_subnet.fc = nn.Linear(self.left_eye_subnet.fc.in_features, 128)

        self.right_eye_subnet = models.resnext50_32x4d(pretrained=True)
        self.right_eye_subnet.fc = nn.Linear(self.right_eye_subnet.fc.in_features, 128)

        # 新增一个处理年龄特征的全连接层
        self.age_fc = nn.Linear(1, 128)

        # 更新分类器的输入维度，包括年龄特征
        self.classifier = nn.Linear(256 + 128, 2)

    def forward(self, left_eye, right_eye, age):
        # 处理左右眼图像
        left_eye_output = self.left_eye_subnet(left_eye)
        right_eye_output = self.right_eye_subnet(right_eye)

        # 处理年龄特征
        age_feature = self.age_fc(age)

        # 将左眼、右眼和年龄特征结合起来
        combined_output = torch.cat((left_eye_output, right_eye_output, age_feature), dim=1)

        # 使用分类器进行预测
        prediction = self.classifier(combined_output)
        return prediction






class ParallenResNetWithAge(nn.Module):
    def __init__(self):
        super(ParallenResNetWithAge, self).__init__()

        # 使用预训练的 ResNeXt50 网络作为基础
        self.left_eye_subnet = models.resnet50(pretrained=True)
        # 替换分类器层以适应新的特征尺寸
        self.left_eye_subnet.fc = nn.Linear(self.left_eye_subnet.fc.in_features, 128)

        self.right_eye_subnet = models.resnet50(pretrained=True)
        self.right_eye_subnet.fc = nn.Linear(self.right_eye_subnet.fc.in_features, 128)

        # 新增一个处理年龄特征的全连接层
        self.age_fc = nn.Linear(1, 128)

        # 更新分类器的输入维度，包括年龄特征
        self.classifier = nn.Linear(256 + 128, 2)

    def forward(self, left_eye, right_eye, age):
        # 处理左右眼图像
        left_eye_output = self.left_eye_subnet(left_eye)
        right_eye_output = self.right_eye_subnet(right_eye)

        # 处理年龄特征
        age_feature = self.age_fc(age)

        # 将左眼、右眼和年龄特征结合起来
        combined_output = torch.cat((left_eye_output, right_eye_output, age_feature), dim=1)

        # 使用分类器进行预测
        prediction = self.classifier(combined_output)
        return prediction



