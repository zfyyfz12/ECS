import pandas as pd
import os
import torch
from torch.utils import data
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
import random
from matplotlib import pyplot as plt


def softmax_cal_acc(feature, target_label, prediction_output=0):
    confidence = nn.functional.softmax(feature, dim=1)
    confidence[:, 0] = confidence[:, 0]
    confidence[:, 1] = confidence[:, 1]
    prediction = torch.max(confidence, 1)[1]
    predict_label = torch.squeeze(prediction, 0)
    acc, sen, spe, ppv, npv = cal_acc(predict_label, target_label)

    if prediction_output == 1:
        confidence = confidence[:, 1] - confidence[:, 0]
        confidence = torch.squeeze(confidence, 0)
        return acc, sen, spe, ppv, npv, predict_label, confidence
    else:
        return acc, sen, spe, ppv, npv


def cal_acc(predict_label, target_label):
    if target_label.shape == torch.Size([]):
        tn = 1 if predict_label + target_label == 2 else 0  # ASD: label=1  HC: label=0
        fn = 1 if predict_label - target_label == 1 else 0
        tp = 1 if predict_label + target_label == 0 else 0
        fp = 1 if predict_label - target_label == -1 else 0
    else:
        tn = sum(predict_label + target_label == 2).item()
        fn = sum(predict_label - target_label == 1).item()
        tp = sum(predict_label + target_label == 0).item()
        fp = sum(predict_label - target_label == -1).item()
    acc = (tp + tn) / (tp + fp + tn + fn)
    sen = tp / (tp + fn) if (tp + fn) != 0 else -1
    spe = tn / (tn + fp) if (tn + fp) != 0 else -1
    ppv = tp / (tp + fp) if (tp + fp) != 0 else -1
    npv = tn / (tn + fn) if (tn + fn) != 0 else -1
    return acc, sen, spe, ppv, npv


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def select_data(cv_fold, id_label, feat_matrix):
    ids = id_label[:, 0]

    train_id = np.load(f'./10_fold_cv_id/train_id_{cv_fold}.npy')
    valid_id = np.load(f'./10_fold_cv_id/valid_id_{cv_fold}.npy')
    test_id = np.load(f'./10_fold_cv_id/test_id_{cv_fold}.npy')
    train_label = np.load(f'./10_fold_cv_id/train_label_{cv_fold}.npy')
    valid_label = np.load(f'./10_fold_cv_id/valid_label_{cv_fold}.npy')
    test_label = np.load(f'./10_fold_cv_id/test_label_{cv_fold}.npy')

    train_idx = np.zeros((len(train_id),))
    valid_idx = np.zeros((len(valid_id),))
    test_idx = np.zeros((len(test_id),))
    i = 0
    for id in train_id:
        train_idx[i] = np.where(ids == id)[0]
        i += 1
    i = 0
    for id in valid_id:
        valid_idx[i] = np.where(ids == id)[0]
        i += 1
    i = 0
    for id in test_id:
        test_idx[i] = np.where(ids == id)[0]
        i += 1
    train_feat = feat_matrix[train_idx.astype(int), :]
    valid_feat = feat_matrix[valid_idx.astype(int), :]
    test_feat = feat_matrix[test_idx.astype(int), :]
    return train_feat, valid_feat, test_feat, train_label, valid_label, test_label


def show_curve(save_path, epoch_list, train_acc_list, valid_acc_list, test_acc_list, save_point_list):
    max_epoch = epoch_list.shape[0]
    plt.plot(epoch_list, train_acc_list, 'y', label='train_acc', linewidth=0.3)
    plt.plot(epoch_list, valid_acc_list, 'g', label='valid_acc', linewidth=0.3)
    plt.plot(epoch_list, test_acc_list, 'b', label='test_acc', linewidth=0.3)
    plt.plot(save_point_list[:, 0], save_point_list[:, 1], 'rx', label=f'saved_{"%.4f" % save_point_list[-1, 1]}',
             linewidth=5)
    plt.xticks(np.arange(0, max_epoch + 1, int(max_epoch / 10)))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    plt.close()


def cal_sc_percent(mask):
    sc_percent = np.zeros((len(mask.keys()),))
    for i in range(len(mask.keys())):
        sc_percent[i] = round(len(np.where(mask[i] == 1)[0]) / (mask[i].shape[0] * mask[i].shape[1]) * 100, 2)
    return sc_percent


if __name__ == '__main__':
    pass
