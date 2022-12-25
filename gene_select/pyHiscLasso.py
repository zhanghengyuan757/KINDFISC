import os
import warnings
from typing import List

import pandas as pd
from pyHSICLasso import HSICLasso
from sklearn.preprocessing import LabelEncoder

from tools.PathTools import DataPathGetter, TempPathGetter


def hsic_lasso(cancer: str, pre_select_genes) -> List[str]:
    warnings.filterwarnings("ignore")
    types = [0, 1, 2, 3]
    dpg = DataPathGetter(cancer)
    path = TempPathGetter().get_path(cancer, 'hsic_lasso_genes.csv')
    if os.path.exists(path):
        return pd.read_csv(path, header=0, index_col=0).iloc[:, 0].tolist()
    X = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
    X = X.filter(items=pre_select_genes, axis=0)
    y = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
    y = y[y['flag'].isin(types)]['flag']
    X = X.filter(items=y.index, axis=1).T
    le = LabelEncoder()
    hsic = HSICLasso()
    y = le.fit_transform(y)
    x, _l = X.to_numpy(), X.columns.tolist()
    hsic.input(x, y, featname=_l)
    hsic.classification(1000, n_jobs=20)
    genes = hsic.get_features()
    score = hsic.get_index_score()
    res = pd.DataFrame([genes, score]).T
    res.to_csv(path)
    return genes
