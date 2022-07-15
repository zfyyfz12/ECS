import numpy as np
from torch import nn
from utils import *
import torch
import copy
from sparselinear import SparseLinear
from sc_linear import *


class DNN(nn.Module):

    def __init__(self, conn_type, mask, in_dim, lyr_1, lyr_2, lyr_3=2):
        super(DNN, self).__init__()
        self.hidden_lyr_num = 2 if lyr_3 == 2 else 3
        if conn_type == 'sc':
            self.lyr_1 = CustomizedLinear(in_dim, lyr_1, mask=mask[0])
            self.lyr_2 = CustomizedLinear(lyr_1, lyr_2, mask=mask[1])
            self.lyr_3 = CustomizedLinear(lyr_2, lyr_3, mask=mask[2])
            if self.hidden_lyr_num == 3:
                self.lyr_4 = CustomizedLinear(lyr_3, 2, mask=mask[3])
        else:
            self.lyr_1 = nn.Linear(in_dim, lyr_1)
            self.lyr_2 = nn.Linear(lyr_1, lyr_2)
            self.lyr_3 = nn.Linear(lyr_2, lyr_3)
            if self.hidden_lyr_num == 3:
                self.lyr_4 = nn.Linear(lyr_3, 2)

    def forward(self, x):
        x = self.lyr_1(x)
        x = x.tanh()
        x = self.lyr_2(x)
        x = x.tanh()
        x = self.lyr_3(x)
        if self.hidden_lyr_num == 3:
            x = x.tanh()
            x = self.lyr_4(x)
        return x

    def train_mlp(self, cv_fold, max_epoch, train_feature, train_label, valid_feature, valid_label,
                  test_feature, test_label, valid_start, valid_end, train_end, LR, alpha, save_path, learning_curve=0,
                  save_dnn=0):

        optimizer = torch.optim.RMSprop(self.parameters(), lr=LR, alpha=alpha)
        loss_func = torch.nn.CrossEntropyLoss()

        max_valid_acc = 0
        max_valid_sen = 0
        max_valid_spe = 0
        max_test_acc = 0
        max_test_sen = 0
        max_test_spe = 0

        flag = 0
        valid_cnt_flag = 0
        epoch_list = np.arange(0, max_epoch)
        train_acc_list = np.zeros(max_epoch)
        valid_acc_list = np.zeros(max_epoch)
        test_acc_list = np.zeros(max_epoch)
        save_point_list = np.zeros((1, 2))
        final_classifier = 'none'

        for epoch in range(max_epoch):
            self.train()
            train_nn_out = self.forward(train_feature)
            train_loss = loss_func(train_nn_out, train_label)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_acc, train_sen, train_spe, train_ppv, train_npv = softmax_cal_acc(train_nn_out, train_label)[0:5]
            if flag == 0:
                valid_end_threshold = 1
            else:
                valid_end_threshold = valid_end

            if learning_curve == 1:
                self.eval()
                valid_nn_output = self.forward(valid_feature)
                test_nn_output = self.forward(test_feature)

                valid_acc, valid_sen, valid_spe, valid_ppv, valid_npv = \
                    softmax_cal_acc(valid_nn_output, valid_label)[0:5]

                test_acc, test_sen, test_spe, test_ppv, test_npv, test_predict_label, test_confidence = \
                    softmax_cal_acc(test_nn_output, test_label, prediction_output=1)

                train_acc_list[epoch] = train_acc
                valid_acc_list[epoch] = valid_acc
                test_acc_list[epoch] = test_acc

            if train_acc >= valid_start and train_acc <= valid_end_threshold:
                flag = 1
                valid_cnt_flag += 1

                if learning_curve == 0:
                    self.eval()
                    valid_nn_output = self.forward(valid_feature)
                    test_nn_output = self.forward(test_feature)

                    valid_acc, valid_sen, valid_spe, valid_ppv, valid_npv = \
                        softmax_cal_acc(valid_nn_output, valid_label)[0:5]

                    test_acc, test_sen, test_spe, test_ppv, test_npv, test_predict_label, test_confidence = \
                        softmax_cal_acc(test_nn_output, test_label, prediction_output=1)

                if valid_acc >= max_valid_acc:
                    valid_loss = loss_func(valid_nn_output, valid_label)

                    max_valid_acc = valid_acc
                    max_valid_sen = valid_sen
                    max_valid_spe = valid_spe
                    max_test_acc = test_acc
                    max_test_sen = test_sen
                    max_test_spe = test_spe
                    save_point_list = np.concatenate((save_point_list, np.array([[epoch, test_acc]])), axis=0)
                    if save_dnn != 0:
                        final_classifier = copy.deepcopy(self)

            if learning_curve == 1:
                if epoch % 50 == 0 or epoch == max_epoch - 1:
                    show_curve(save_path, epoch_list, train_acc_list, valid_acc_list, test_acc_list,
                               save_point_list)

            if train_acc > train_end:
                break

            if valid_cnt_flag >= 200:
                break
        if save_dnn != 0:
            print(final_classifier)
            torch.save(final_classifier, f'./scdnn_model_fold_{cv_fold}.pkl')

        return max_valid_acc, max_valid_sen, max_valid_spe, max_test_acc, max_test_sen, max_test_spe, train_loss, valid_loss
