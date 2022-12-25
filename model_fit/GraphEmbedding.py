import time
from typing import List

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from model_fit.Models import GravityGAE
from processing.DataPrepare import *
from tools.ReportTools import reports

learning_rate = 0.05
hid_n = 64
z_n = 32
epochs = 200
cuda = True
gpu_id = 0
verbose = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 用作模型评估
def get_pred_from_emb(edge_pos: pd.DataFrame, emb: pd.DataFrame, epsilon=0.01):
    assert type(edge_pos) == pd.DataFrame and edge_pos.columns.tolist() == ['i', 'j']
    emb_index = emb.index.tolist()
    emb_index = dict(zip(emb_index, range(len(emb_index))))
    emb = emb.to_numpy()
    edges_pos = edge_pos.to_numpy()
    edges_neg = edge_pos[['j', 'i']].to_numpy()
    preds = []
    p = []
    for e0, e1 in np.vstack([edges_pos, edges_neg]):
        e0, e1 = emb_index[e0], emb_index[e1]
        dist = np.square(epsilon + np.linalg.norm(emb[e0, 0:-1] - emb[e1, 0:-1], ord=2))
        pred_score = sigmoid(emb[e1, -1] - np.log(dist))
        preds.append(pred_score)
        if pred_score > 0.5:
            p.append(1)
        else:
            p.append(0)
    preds_all = np.array(p)
    labels_all = np.hstack([np.ones(edges_pos.shape[0]), np.zeros(edges_neg.shape[0])])
    return labels_all, preds_all


def cal_dist(n0, n1, emb, epsilon=0.01):
    dist = np.square(epsilon + np.linalg.norm(emb[n0, 0:-1] - emb[n1, 0:-1], ord=2))
    return sigmoid(emb[n1, -1] - np.log(dist))


def ggae_predict(cancer, feat_select_method, threshold, expr, stage, frac_I_III, frac_IA_IVC):
    edge_I_IV, edge_IA_IVC, edge_I = get_directed_edges(cancer, feat_select_method, threshold)
    train_y, test_y_I_IV, test_y_IA_IVC = split_edges(edge_I_IV, edge_IA_IVC, frac_I_III, frac_IA_IVC)
    adj, gk = prepare_gae_data(train_y, pd.concat([test_y_I_IV, test_y_IA_IVC]), edge_I)
    features = np.eye(adj.shape[0])
    # features = expr.iloc[:, gk].to_numpy().T
    emb = ggae_get_emb(adj, features, gk)
    print("IA-II\II-III测试集", int(frac_I_III * 100), "%用来训练，", int(100 - frac_I_III * 100), '%用来测试:')
    test_y_I_IV = test_y_I_IV[test_y_I_IV['y'] == 1]
    y, pred = get_pred_from_emb(test_y_I_IV[['i', 'j']], emb)
    reports(y, pred)
    if not test_y_IA_IVC.empty:
        print('IA-IB\IIA-IIB测试集')
        test_y_IA_IVC = test_y_IA_IVC[test_y_IA_IVC['y'] == 1]
        y, pred = get_pred_from_emb(test_y_IA_IVC[['i', 'j']], emb)
        reports(y, pred)
    preds = []
    preds_neg = []
    mass_sub = []
    i = []
    j = []
    emb_index = emb.index.tolist()
    emb_index = dict(zip(emb_index, range(len(emb_index))))
    emb = emb.to_numpy()
    edges_to_pred = pd.concat([edge_I_IV, edge_IA_IVC, edge_I])
    for _, l in edges_to_pred.iterrows():
        if l['i'] not in emb_index.keys() or l['j'] not in emb_index.keys():
            continue
        l1, l2 = emb_index[l['i']], emb_index[l['j']]
        preds.append(cal_dist(l1, l2, emb))  # i->j
        preds_neg.append(cal_dist(l2, l1, emb))  # j->i
        p = sigmoid(emb[l2, -1] - emb[l1, -1])  # i->j i默认为较早的样本
        if p > 0.5:
            mass_sub.append(1)
        else:
            mass_sub.append(0)
        i.append(l['i'])
        j.append(l['j'])
    emb_prediction = list(zip(preds, preds_neg, mass_sub, i, j))
    emb_prediction = pd.DataFrame(emb_prediction, columns=['i->j', 'i<-j', 'i<j', 'i', 'j'])
    return emb_prediction


def ggae_get_emb(adj, x, gk):
    input_dim = x.shape[1]
    tensor_x = torch.FloatTensor(x)
    edge_sum = adj.sum()
    pos_weight = torch.FloatTensor([(input_dim ** 2 - edge_sum) / edge_sum])
    cost_norm = input_dim ** 2 / float((input_dim ** 2 - edge_sum) * 2)
    adj_norm = normalize_adj(adj)
    tensor_adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    tensor_adj_eye = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))

    model = GravityGAE(input_dim, hid_n, z_n)
    print('Gravity_AE Training...')
    if cuda:
        tensor_x = tensor_x.cuda(gpu_id)
        tensor_adj_norm = tensor_adj_norm.cuda(gpu_id)
        tensor_adj_eye = tensor_adj_eye.cuda(gpu_id)
        pos_weight = pos_weight.cuda(gpu_id)
        model.cuda(gpu_id)
    cost = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    print("")
    for epoch in range(epochs):
        t = time.time()
        model.train()
        x2 = model(tensor_x, tensor_adj_norm)
        optimizer.zero_grad()
        loss = cost_norm * cost(x2, tensor_adj_eye.to_dense())
        loss.backward()
        optimizer.step()
        model.eval()
        if np.isnan(loss.item()):
            raise Exception('GAE Gradient disappear! Try to low "learning_rate" in GraphEmbedding.py')
        if verbose and (epoch + 1) % int(epochs / 20) == 0:
            print("\rEpoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  "time=", "{:.5f}".format(time.time() - t), end='', flush=True)
    print("")
    print('Gravity_AE Trained.')
    model.eval()
    emb = model.encode(tensor_x, tensor_adj_norm)
    embedding = pd.DataFrame(emb.cpu().detach().numpy(), index=gk).sort_index()
    return embedding
