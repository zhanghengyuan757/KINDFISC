import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from model_fit.GraphEmbedding import ggae_get_emb
from processing.DataPrepare import get_directed_edges, split_edges, prepare_gae_data, kfold_split
from tools.PathTools import TempPathGetter
from tools.ReportTools import reports

per_I_IV, per_IA_IVC = 0.8, 0.8
k_fold = 5


class BaseCancerCL:
    def __init__(self, cancer, feat_select_method, threshold, expr, stage):
        self.expr = expr
        self.stage = stage
        self.cancer = cancer
        self.feat_select_method = feat_select_method
        self.embedding = None
        edge_I_IV, edge_IA_IVC, edge_I = get_directed_edges(cancer, feat_select_method, threshold)
        self.edge_I_IV, self.edge_IA_IVC, self.edge_I = edge_I_IV, edge_IA_IVC, edge_I
        self.k_fold_train_y = []
        self.k_fold_test_y_I_IV = []
        self.k_fold_test_y_IA_IVC = []
        if k_fold <= 1:
            train_y, test_y_I_IV, test_y_IA_IVC = split_edges(edge_I_IV, edge_IA_IVC, per_I_IV, per_IA_IVC)
            self.k_fold_train_y.append(train_y)
            self.k_fold_test_y_I_IV.append(test_y_I_IV)
            self.k_fold_test_y_IA_IVC.append(test_y_IA_IVC)
        if k_fold > 1:
            folds = kfold_split(k_fold, edge_I_IV)  # 返回k_fold个数组[(train_index,test_index),(),...]
            folds2 = kfold_split(k_fold, edge_IA_IVC)
            for k in range(k_fold):
                train, test = folds[k]
                train2, test2 = folds2[k]
                self.k_fold_train_y.append(pd.concat([edge_I_IV.iloc[train, :], edge_IA_IVC.iloc[train2, :]], axis=0))
                self.k_fold_test_y_I_IV.append(edge_I_IV.iloc[test, :])
                self.k_fold_test_y_IA_IVC.append(edge_IA_IVC.iloc[test2, :])

    def fit_pred(self, k=0):
        train_y = self.k_fold_train_y[k]
        test_y_I_IV = self.k_fold_test_y_I_IV[k]
        test_y_IA_IVC = self.k_fold_test_y_IA_IVC[k]
        if k_fold > 1:
            print('=' * 20, '第', k + 1, '折', '=' * 20)
            self.reset_model()
        adj, gk = prepare_gae_data(train_y, pd.concat([test_y_I_IV, test_y_I_IV]), self.edge_I)
        embedding = ggae_get_emb(adj, np.eye(adj.shape[0]), gk).T
        embedding.columns = self.expr.columns[embedding.columns]
        embedding.index = ['emb_' + str(e) for e in embedding.index]
        temp = pd.concat([self.expr, embedding], axis=0).fillna(0)
        embedding = temp.filter(items=embedding.index, axis=0).fillna(0)
        self.embedding = embedding
        self.fit(self.expr, self.embedding, self.stage, train_y)

        if not test_y_I_IV.empty:
            print("I-II\II-III测试集:")
            y, pred = test_y_I_IV['y'], self.predict(self.expr, self.embedding, self.stage,
                                                     test_y_I_IV[['i', 'j']])
            report1 = reports(y, pred)
        if not test_y_IA_IVC.empty:
            print('IA-IB\IIA-IIB测试集:')
            y, pred = test_y_IA_IVC['y'], self.predict(self.expr, self.embedding, self.stage,
                                                       test_y_IA_IVC[['i', 'j']])
            report2 = reports(y, pred)
        if k_fold > 1:
            return report1, report2
        else:
            return

    def pred_all(self):
        if self.edge_IA_IVC.empty:
            all_edge = pd.concat(
                [self.edge_I_IV[['i', 'j']], self.edge_I[['i', 'j']]]).astype(int)
        else:
            all_edge = pd.concat(
                [self.edge_I_IV[['i', 'j']], self.edge_IA_IVC[['i', 'j']], self.edge_I[['i', 'j']]]).astype(int)
        all_edge = all_edge.reset_index(drop=True)
        pred = self.predict(self.expr, self.embedding, self.stage, all_edge)
        pred_prob = self.predict_proba(self.expr, self.embedding, self.stage, all_edge)
        result = np.hstack([pred_prob, pred.reshape(-1, 1), all_edge[['i', 'j']].to_numpy()])
        return pd.DataFrame(result, columns=['i->j', 'i<-j', 'i<j', 'i', 'j'])

    def fit(self, expr, embedding, stage, edges):
        pass

    def predict(self, expr, embedding, stage, edges):
        pass

    def predict_proba(self, expr, embedding, stage, edges):
        pass

    def reset_model(self):
        pass
