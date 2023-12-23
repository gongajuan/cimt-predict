# -*- coding: utf-8 -*-
# @Time    : 2023-10-27 17:26
# @Author  : FWJ
# @Software: PyCharm
import csv
import os
import torch
import numpy as np

from tqdm import tqdm
import torch.nn as nn

from utilstools import utils
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from utilstools.utils import ValidTransform
from sklearn.metrics import classification_report
from ModelAndEyeDataset import ParallenResNextWithAge,EyeDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(test_data_loader):
    model.eval()  # 测试模式
    pred = []
    trued = []
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for left_eye, right_eye, labels, age in test_data_loader:
        left_eye = left_eye.to(device)
        right_eye = right_eye.to(device)
        labels = labels.to(device)
        age = age.to(device)



        # 检查输出

        output = model(left_eye, right_eye, age)

        pre = torch.max(F.softmax(output, dim=1), dim=1)[1]
        pred += list(pre.cpu().numpy())
        trued += list(labels.cpu().numpy())

        score_tmp = output
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
    return pred, trued, score_list, label_list




def run_tests(model, title, image_path, classes1, classes2,save_path='./data',  age_path='./age.xls',num_classes=2, batch_size=30):
    model_folder = os.path.join(image_path, title)
    save_path1=os.path.join(save_path,title)
    if not os.path.exists( save_path1):
        os.makedirs( save_path1)



    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for name in ['valid', 'test']:
        path = os.path.join(model_folder, name)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(save_path1, name)
        if not os.path.exists(save_path ):
            os.makedirs(save_path )

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Check if using multiple GPUs
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        # Prepare test data
        test_dataset = EyeDataset(root_dir=os.path.join(image_path, name), transform=ValidTransform,age_path=age_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        pred, trued, score_list, label_list = test(test_data_loader=test_loader)

        true_1 = [classes1[i] for i in trued]
        predictions = [classes1[i] for i in pred]

        # Plot confusion matrix
        conf_mat = confusion_matrix(y_true=true_1, y_pred=predictions)
        utils.plot_confusion_matrix(conf_mat, normalize=True, save_dir=path, target_names=classes2,
                                    title='Confusion_Matrix')

        # Calculate and plot PR and ROC curves
        score_array = np.array(score_list)
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_classes)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        fpr_dict, tpr_dict, roc_auc_dict = utils.calculate_roc_metrics(label_onehot, score_array, num_classes)
        precision, recall, average_precision = utils.calculate_precision_recall_metrics(label_list, score_array,
                                                                                        num_classes)

        # Save PR and ROC curves
        pr_curve_path = os.path.join(save_path, 'PR_curve.pdf')
        utils.plot_pr_curve(precision, recall, average_precision, classes2, pr_curve_path)

        roc_curve_path = os.path.join(save_path, 'ROC.pdf')
        utils.plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, classes2, roc_curve_path, title=title)

        # Save confusion matrix and classification report
        cm_file = os.path.join(save_path, 'confusion_matrix.txt')
        with open(cm_file, 'w') as f:
            f.write(str(conf_mat))

        report = classification_report(true_1, predictions, target_names=classes2, digits=4)
        report_file = os.path.join(save_path, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)

        # Save micro-average curve data to Excel
        curves_excel_path = os.path.join(save_path, 'micro_curves_data.xlsx')
        utils.save_micro_curves_to_excel(precision['micro'], recall['micro'], fpr_dict['micro'], tpr_dict['micro'],
                                         roc_auc_dict['micro'], curves_excel_path)



if __name__ == '__main__':
    # Setup and initialize variables
    save_path = r'./data'
    weight_path = r'ParallenResNextWithAgemodel_state_dict_0.8500.pth'
    image_path ='H:\eye-cmit'  # Replace with actual path
    classes1 = ["0", "1"]
    classes2 = ["CMIT Normal", "CIMIT Thickened"]
    age_csv_path='./age.xls'



    model = ParallenResNextWithAge()
    title = model.__class__.__name__
    #model =  ParallenResNetWithAge()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load model
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # Call the function
    run_tests(model, title,  image_path, classes1, classes2,save_path=save_path,age_path=age_csv_path)



