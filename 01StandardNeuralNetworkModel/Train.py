import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from ModelAndEyeDataset import StandardResNeXt,StandardResNest,EyeDataset
from utilstools.utils import ValidTransform, TrainTransform, calculate_weights

# 检测 GPU 并选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        valid_loss, valid_acc = validate_model(model, valid_loader, criterion)

        if valid_acc > best_acc and epoch_acc > valid_acc:
            best_acc = valid_acc
            filename = f"{modleName}model_state_dict_{best_acc:.4f}.pth"
            torch.save(model.state_dict(), filename)  # 保存最佳模型

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

        scheduler.step()

def validate_model(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)
    return total_loss, total_acc

def main(TRAIN_DATA_PATH, VALID_DATA_PATH,weight_path,model,num_epochs=200,lr=0.001):
    train_dataset = EyeDataset(TRAIN_DATA_PATH, transform=TrainTransform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, prefetch_factor=2)
    valid_dataset = EyeDataset(VALID_DATA_PATH, transform=ValidTransform)
    valid_loader = DataLoader(valid_dataset, batch_size=50, num_workers=4, prefetch_factor=2)
    model=model

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    weights = calculate_weights(TRAIN_DATA_PATH)  #
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if weight_path and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,num_epochs=num_epochs)


if __name__ == '__main__':
    # 全局参数
    TRAIN_DATA_PATH = r'H:\eye-cmit2\train'  # training data path
    VALID_DATA_PATH = r'H:\eye-cmit2\valid'  # valid  data path
    weight_path = r''
    ##Select StandardResNeXt model type

    model = StandardResNest(num_classes=2).to(device)
    modleName=model.__class__.__name__

    ##Select StandardResNeXt model type
    #model = StandardResNest(num_classes=2).to(device)
    #训练轮数
    num_epochs=200
    #学习参数
    lr=0.001

    main(TRAIN_DATA_PATH=TRAIN_DATA_PATH, VALID_DATA_PATH=VALID_DATA_PATH, weight_path=weight_path,model=model,num_epochs=num_epochs,lr=lr)