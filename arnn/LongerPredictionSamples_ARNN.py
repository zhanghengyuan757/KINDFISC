"""
Date: 2021-08-09 16:44:14
LastEditors: GodK
"""
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg

from arnn.NN_F2 import NN_F2
# Y = mylorenz(30)  # coupled lorenz system
from tools.PathTools import LogPathGetter


def set_random():
    seed = 666
    random.seed(seed)
    np.random.seed(seed)


def run_arnn(sorted_rdata, jd, plot=False, cancer='', gene=''):
    set_random()
    Y = sorted_rdata.T.to_numpy()
    noisestrength = 0

    X = Y + noisestrength * np.random.rand(*Y.shape)  # noise could be added

    ii = 0
    ii_set = [10]  # sample init, can be changed;
    result = []
    for ii in ii_set:
        # print(ii)

        trainlength = 30  # length of training data (observed data), m

        selected_variables_idx = list(
            range(sorted_rdata.index.size))  # selected the most correlated variables, can be changed
        xx = X[int(X.shape[0] / 5) + ii: X.shape[0], selected_variables_idx].T  # after transient dynamics
        noisestrength = 0  # strength of noise
        xx_noise = xx + noisestrength * np.random.rand(*xx.shape)

        traindata = xx_noise[:, :trainlength]
        k = 50  # randomly selected variables of matrix B
        predict_len = 11  # L

        # jd = 2  # the index of target variable

        D = xx_noise.shape[0]  # number of variables in the system
        real_y = xx[jd, :]
        real_y_noise = real_y + noisestrength * np.random.rand(*real_y.shape)
        traindata_y = real_y_noise[:trainlength]

        """
        ****************************** ARNN start **************************
        """

        # Given a set of fixed weights for F for each time points: A*F(X^t)=Y^t, F(X^t)=B*(Y^t)
        traindata_x_NN = NN_F2(traindata)
        # traindata_x_NN=sio.loadmat("traindata_x_NN.mat")["traindata_x_NN"]

        w_flag = np.zeros((traindata_x_NN.shape[0], traindata_x_NN.shape[0]))
        A = np.zeros((predict_len, traindata_x_NN.shape[0]))  # matrix A
        B = np.zeros((traindata_x_NN.shape[0], predict_len))  # matrix B

        predict_pred = np.zeros(predict_len - 1)

        # End of ITERATION 1:  sufficient iterations
        for iter in range(1000):  # cal coeffcient B

            temp_set = list(set(range(traindata_x_NN.shape[0])) - set([jd]))
            np.random.shuffle(temp_set)

            random_idx = sorted([jd, *temp_set[:k - 1]])
            traindata_x = traindata_x_NN[random_idx, :trainlength]  # random chose k variables from F(D)

            for i in range(traindata_x.shape[0]):
                # Ax=b,  1: x=pinv(A)*b,    2: x=A\b,    3: x=lsqnonneg(A,b)
                b = traindata_x[i, :trainlength - predict_len + 1].T  # 1*(m-L+1)

                B_w = np.zeros((trainlength - predict_len + 1, predict_len))
                for j in range(trainlength - predict_len + 1):
                    B_w[j, :] = traindata_y[j:j + predict_len]
                B_para = np.transpose(np.linalg.pinv(B_w).dot(b))
                B[random_idx[i], :] = (B[random_idx[i], :] + B_para + B_para * (1 - w_flag[random_idx[i], 0])) / 2
                w_flag[random_idx[i], 0] = 1

            ####################### tmp predict based on B ############################
            super_bb = np.zeros((predict_len - 1) * traindata_x_NN.shape[0])
            super_AA = np.zeros(((predict_len - 1) * traindata_x_NN.shape[0], predict_len - 1))
            for i in range(traindata_x_NN.shape[0]):
                kt = -1
                bb = np.zeros(predict_len - 1)
                AA = np.zeros((predict_len - 1, predict_len - 1))
                for j in range(trainlength - (predict_len - 1), trainlength):
                    kt = kt + 1
                    bb[kt] = traindata_x_NN[i, j]
                    col_known_y_num = trainlength - j
                    for r in range(col_known_y_num):
                        bb[kt] = bb[kt] - B[i, r] * traindata_y[trainlength - col_known_y_num + r]
                    AA[kt, :predict_len - col_known_y_num] = B[i, col_known_y_num:predict_len]

                super_bb[(predict_len - 1) * i:(predict_len - 1) * i + predict_len - 1] = bb[:]
                super_AA[(predict_len - 1) * i:(predict_len - 1) * i + predict_len - 1, :] = AA

            pred_y_tmp = (np.linalg.pinv(super_AA).dot(super_bb.T)).T

            ####################### update the values of matrix A and Y ############################
            tmp_y = np.concatenate([real_y[:trainlength], pred_y_tmp])
            Ym = np.zeros((predict_len, trainlength))
            for j in range(predict_len):
                Ym[j, :] = tmp_y[j: j + trainlength]

            BX = np.concatenate([B, traindata_x_NN], axis=1)
            IY = np.concatenate([np.eye(predict_len), Ym], axis=1)
            A = IY @ np.linalg.pinv(BX)
            union_predict_y_ARNN = np.zeros(predict_len - 1)
            for j1 in range(predict_len - 1):
                tmp_y = np.zeros((predict_len - j1 - 1, 1))
                kt = -1
                for j2 in range(j1, predict_len - 1):
                    kt = kt + 1
                    row = j2 + 1
                    col = trainlength - j2 + j1 - 1
                    tmp_y[kt] = A[row, :] @ traindata_x_NN[:, col]
                union_predict_y_ARNN[j1] = np.mean(tmp_y)

            # End of ITERATION 2: the predicting result converges.

            eof_error = np.sqrt(np.square(union_predict_y_ARNN - predict_pred).mean())
            if eof_error < 0.0001:
                break

            predict_pred = union_predict_y_ARNN

        ######################### result display #################################
        my_real = real_y[trainlength:trainlength + predict_len - 1]
        RMSE = np.sqrt(np.square(union_predict_y_ARNN - my_real).mean())
        RMSE = RMSE / (
                np.std(real_y[trainlength - 2 * predict_len:trainlength + predict_len - 1]) + 0.001)  # normalize RMSE
        if plot:
            refx = X[int(X.shape[0] / 5 * 2) + ii:X.shape[0], :].T  # Lorenz reference

            # plt.subplot(2, 1, 1)
            # plt.plot(refx[jd, :int(X.shape[0] / 5 * 2)], 'c-*', linewidth=2, markersize=4)
            # plt.plot(list(range(int(X.shape[0] / 5 * 2), int(X.shape[0] / 5 * 2) + trainlength)), real_y[:trainlength],
            #          '-*', linewidth=2, markersize=4)
            # plt.title(f'original attractor. Init: {ii}, Noise strength: {noisestrength}')
            plt.figure(figsize=(8, 4))
            # plt.subplot(2, 1, 2)
            real_y = [list(range(trainlength + predict_len - 1)), real_y_noise[:trainlength + predict_len - 1]]
            lpg = LogPathGetter()
            pd.DataFrame(real_y).T.to_csv(
                lpg.get_path(cancer, 'arnn', gene.split("|")[0], f'RMSE{round(RMSE, 2)}_real.csv'))
            plt.plot(real_y[0], real_y[1], '-*',
                     linewidth=2, markersize=4, label='real trends')
            pred_y = [list(range(trainlength, trainlength + predict_len - 1)), union_predict_y_ARNN]
            pd.DataFrame(pred_y).T.to_csv(
                lpg.get_path(cancer, 'arnn', gene.split("|")[0], f'RMSE{round(RMSE, 2)}_pred.csv'))
            plt.plot(pred_y[0], pred_y[1], 'r-o', linewidth=2,
                     markersize=5, label='predicted trends')
            plt.title(
                f'ARNN Prediction of {cancer.upper()} dataset of gene {gene.split("|")[0]} (m={trainlength}, L={predict_len}, RMSE={round(RMSE, 2)})')
            plt.legend()
            plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
            plt.draw()
            plt.pause(1)
            plt.clf()
        result.append(RMSE)
    return result


def get_arnn(cancer, rdata, title):
    rdata = rdata.apply(np.exp2) - 1
    result = []
    rdata_index_size = rdata.index.size
    if rdata_index_size > 150:
        rdata_index_size = 50
    for jd in range(rdata_index_size):
        result.append(run_arnn(rdata, jd, plot=False, cancer=cancer, gene=rdata.index[jd]))
    a = pd.DataFrame(result)
    a['sum'] = a.sum(axis=1)
    print('ARNN Result avg:', a.mean().iloc[0])
    a = a.sort_values(by=['sum'])
    indexs = a.index[:3]
    print(rdata.index[indexs[0]], rdata.index[indexs[1]])
    run_arnn(rdata, indexs[0], plot=True, cancer=cancer, gene=rdata.index[indexs[0]])
    run_arnn(rdata, indexs[1], plot=True, cancer=cancer, gene=rdata.index[indexs[1]])
    run_arnn(rdata, indexs[2], plot=True, cancer=cancer, gene=rdata.index[indexs[1]])
    lpg = LogPathGetter()
    # a.to_csv(lpg.get_path(title, 'arnn.csv'))
