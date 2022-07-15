import time
from utils import *
import os
import numpy as np
from torch.utils import data
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter setting
seed = 10000
feat_num = 9000

noise_percent_1 = 0
ae_LR = 0.0001
nn_LR = 0.0001
ae_alpha = 0.9
nn_alpha = 0.9
valid_start = 0.99
valid_end = 1
train_end = 1
epoch_1 = 160

# Load the selected functional connectivity vectors of 505 ASD and 530 HC subjects.
# Feature vector of each subject inclued 9000 top features of the DSDC-based feature ranking.
# The DSDC-based feature selection method refer to "Identification of Autism spectrum disorder based on a novel feature
# selection method and Variational Autoencoder" Fangyu Zhang, Yanjie Wei, Jin Liu, Yanlin Wang, Wenhui Xi, Yi Pan
feat_mat = np.load(f'./cc400_1035_pcc_matrix_top_{feat_num}.npy')
id_label = np.load(f'./id_label_1035.npy')

def train_and_eval(conn_type, mask, lyr_1_out, lyr_2_out, lyr_3_out, max_epoch, save_dnn):
    fit_eval_time_s = time.time()
    if not os.path.exists(f'./{conn_type}_log'):
        os.mkdir(f'./{conn_type}_log')
    log_path = f'./{conn_type}_log'

    valid_acc_array = np.zeros((10,))
    test_acc_array = np.zeros(10)
    test_sen_array = np.zeros(10)
    test_spe_array = np.zeros(10)
    train_loss_array = np.zeros(10)
    valid_loss_array = np.zeros(10)

    # 10-fold cross validation
    for cv_fold in range(0, 10):

        train_feat, valid_feat, test_feat, \
        train_label, valid_label, test_label = select_data(cv_fold, id_label, feat_mat)

        train_feat = torch.Tensor(train_feat).to(device)
        valid_feat = torch.Tensor(valid_feat).to(device)
        test_feat = torch.Tensor(test_feat).to(device)
        train_label = torch.Tensor(train_label).type(torch.LongTensor).to(device)
        valid_label = torch.Tensor(valid_label).type(torch.LongTensor).to(device)
        test_label = torch.Tensor(test_label).type(torch.LongTensor).to(device)

        # train dnn ----------------------------------------------------------------------------------------------------
        set_random_seed(seed)
        mlp = DNN(conn_type, mask, train_feat.shape[1], lyr_1_out, lyr_2_out, lyr_3_out).to(
            device)

        save_path = f'{log_path}/learning_curve_{feat_num}_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}_{cv_fold}'
        valid_acc, valid_sen, valid_spe, test_acc, test_sen, test_spe, train_loss, valid_loss = \
            mlp.train_mlp(cv_fold, max_epoch, train_feat, train_label, valid_feat, valid_label,
                          test_feat, test_label, valid_start, valid_end, train_end, nn_LR, nn_alpha,
                          save_path, learning_curve=0, save_dnn=save_dnn)

        valid_acc_array[cv_fold] = round(valid_acc, 4)
        test_acc_array[cv_fold] = round(test_acc, 4)
        test_sen_array[cv_fold] = round(test_sen, 4)
        test_spe_array[cv_fold] = round(test_spe, 4)
        train_loss_array[cv_fold] = train_loss
        valid_loss_array[cv_fold] = valid_loss

    # cal avg value of 10 folds runs------------------------------------------------------------------------------------
    avg_vacc = np.average(valid_acc_array[np.where(valid_acc_array != 0)[0]])
    avg_tacc = np.average(test_acc_array[np.where(test_acc_array != 0)[0]])
    avg_tsen = np.average(test_sen_array[np.where(test_sen_array != 0)[0]])
    avg_tspe = np.average(test_spe_array[np.where(test_spe_array != 0)[0]])
    avg_trloss = np.average(train_loss_array[np.where(train_loss_array != 0)[0]])
    avg_vloss = np.average(valid_loss_array[np.where(valid_loss_array != 0)[0]])

    if conn_type == 'sc':
        sc_percent = cal_sc_percent(mask)
    else:
        sc_percent = 100

    np.save(f'{log_path}/{feat_num}_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}_' + \
            f'{"%.5f" % (avg_vacc - 0.05 * avg_trloss)}_{"%.5f" % avg_trloss}_{"%.4f" % avg_vacc}_{"%.4f" % avg_tacc}_{sc_percent}%',
            np.concatenate((valid_acc_array.reshape(1, -1),
                            test_acc_array.reshape(1, -1),
                            test_sen_array.reshape(1, -1),
                            test_spe_array.reshape(1, -1),
                            train_loss_array.reshape(1, -1),
                            valid_loss_array.reshape(1, -1))))
    print(f'sparse degree: {sc_percent}')
    print('avg_vacc: ', "%.4f" % avg_vacc)
    print('avg_tacc: ', "%.4f" % avg_tacc)
    print('avg_tsen: ', "%.4f" % avg_tsen)
    print('avg_tspe: ', "%.4f" % avg_tspe)

    fit_eval_time_e = time.time()
    print('fitness eval time:', fit_eval_time_e - fit_eval_time_s)

    return avg_vacc, avg_tacc, avg_trloss, avg_vloss
