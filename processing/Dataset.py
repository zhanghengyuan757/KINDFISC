import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class Seq2Dataset(Dataset):
    def __init__(self, expr: pd.DataFrame, embedding: pd.DataFrame, stage: pd.DataFrame, edges: pd.DataFrame,
                 concat=True):
        if expr is None:
            t_matrix = embedding
        elif embedding is None:
            t_matrix = expr
        else:
            t_matrix = pd.concat([expr, embedding], axis=1)
        self.expr = expr.to_numpy()
        if embedding is not None:
            self.embedding = embedding.to_numpy()
        else:
            self.embedding = None
        self.t_matrix = t_matrix.to_numpy()
        self.edge_index = edges.to_numpy()
        self.predict = edges.shape[1] == 2
        self.concat = concat
        if stage is None:
            self.stage = None
        else:
            self.stage = LabelEncoder().fit_transform(stage['stage'])
            self.flag = stage['flag'].tolist()

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        i, j = self.edge_index[idx, 0:2]
        if self.stage is None:
            if self.concat:
                x = np.hstack([self.t_matrix[:, i], self.t_matrix[:, j]])
            else:
                x = self.t_matrix[:, i] - self.t_matrix[:, j]
            x = torch.from_numpy(x).type(torch.FloatTensor)
            if not self.predict:
                y = self.edge_index[idx, -1]
                y = torch.Tensor([y]).type(torch.LongTensor)
                return x, y
            else:
                return x
        else:
            expr1 = torch.from_numpy(self.expr[:, i]).type(torch.FloatTensor)
            expr2 = torch.from_numpy(self.expr[:, j]).type(torch.FloatTensor)
            embedding1 = torch.from_numpy(self.embedding[:, i]).type(torch.FloatTensor)
            embedding2 = torch.from_numpy(self.embedding[:, j]).type(torch.FloatTensor)
            s1, s2 = self.stage[i], self.stage[j]
            f1, f2 = self.flag[i], self.flag[j]
            if not self.predict:
                y = self.edge_index[idx, -1]
                y = torch.Tensor([y]).type(torch.LongTensor)
                return expr1, expr2, embedding1, embedding2, s1, s2, f1, f2, y
            else:
                return expr1, expr2, embedding1, embedding2
