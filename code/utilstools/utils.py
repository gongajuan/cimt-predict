
import matplotlib.pyplot as plt

from itertools import cycle

import csv

import pandas as pd

from torchvision import transforms

TrainTransform= transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224,224)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


ValidTransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




def save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'FPR', 'TPR', 'AUC'])

        # 保存micro-average ROC曲线数据
        for fpr, tpr in zip(fpr_dict["micro"], tpr_dict["micro"]):
            writer.writerow(['Micro-average', fpr, tpr, roc_auc_dict["micro"]])

        # 保存其他类别的ROC曲线数据
        for i, class_name in enumerate(class_names):
            for fpr, tpr in zip(fpr_dict[i], tpr_dict[i]):
                writer.writerow([class_name, fpr, tpr, roc_auc_dict[i]])


def plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict, class_names, save_path,title):
    plt.figure()
    lw = 2
    roc_csv = save_path + 'roc_data.csv'
    save_roc_data_to_csv(fpr_dict, tpr_dict, roc_auc_dict, class_names, roc_csv)

    # 仅绘制一次 Micro-average ROC curve
    plt.plot(fpr_dict["micro"] * 100, tpr_dict["micro"] * 100,
             label=f'Micro-average ROC Curve (AUC = {roc_auc_dict["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=4)

    # 绘制其他类别的 ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow', 'lime', 'blue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr_dict[i] * 100, tpr_dict[i] * 100, color=color, lw=lw,
                 label=f'{class_names[i]} ROC Curve (AUC = {roc_auc_dict[i]:0.4f})')

    titlename=f'The ROC Curve of the {title} Model Predicting CMIT Thickness'

    plt.plot([0, 1] * 100, [0, 1] * 100, 'k--', lw=lw)
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 100.0])
    plt.xlabel('False Positive Rate (%)')
    plt.ylabel('True Positive Rate (%)')
    plt.title(titlename)
    plt.legend(loc="lower right")
    plt.savefig(save_path, format='pdf')
    plt.show()


def save_micro_curves_to_excel(pr_precision_micro, pr_recall_micro, roc_fpr_micro, roc_tpr_micro, roc_auc_micro, excel_path):
    """
    保存micro-average PR曲线和ROC曲线数据到Excel文件中。
    :param pr_precision_micro: micro-average PR曲线的精确率数据。
    :param pr_recall_micro: micro-average PR曲线的召回率数据。
    :param roc_fpr_micro: micro-average ROC曲线的FPR数据。
    :param roc_tpr_micro: micro-average ROC曲线的TPR数据。
    :param roc_auc_micro: micro-average ROC曲线的AUC数据。
    :param excel_path: 保存Excel文件的路径。
    """
    # 创建PR曲线的DataFrame
    pr_data = {'Precision': pr_precision_micro, 'Recall': pr_recall_micro}
    pr_df = pd.DataFrame(pr_data)

    # 创建ROC曲线的DataFrame
    roc_data = {'FPR': roc_fpr_micro, 'TPR': roc_tpr_micro, 'AUC': [roc_auc_micro] * len(roc_fpr_micro)}
    roc_df = pd.DataFrame(roc_data)

    # 保存到Excel
    with pd.ExcelWriter(excel_path) as writer:
        pr_df.to_excel(writer, sheet_name='Micro-Average PR Curve', index=False)
        roc_df.to_excel(writer, sheet_name='Micro-Average ROC Curve', index=False)