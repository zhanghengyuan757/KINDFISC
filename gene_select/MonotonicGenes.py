import os
from typing import List

import numpy as np
import pandas as pd

from tools.PathTools import DataPathGetter, TempPathGetter
from scipy import stats

p = 0.1
fc = 1.1


def find_mono_genes(cancer: str,pre_select_genes) -> List[str]:
    types = [0, 1, 2, 3]
    dpg = DataPathGetter(cancer)
    path = TempPathGetter().get_path(cancer, 'dge', fc, p, 'genes.csv')
    if os.path.exists(path):
        return pd.read_csv(path, header=0, index_col=0).iloc[:, 0].tolist()
    gene_expr = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
    stages = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
    gene_expr = gene_expr.filter(items=[i for i in gene_expr.index if not i.startswith('?')], axis=0)
    gene_expr = gene_expr.filter(items=pre_select_genes, axis=0)
    stage_mean = []
    for f in stages['flag'].drop_duplicates():
        temp = gene_expr.filter(items=stages[stages['flag'] == f].index, axis=1).mean(axis=1)
        stage_mean.append(temp)
    stage_mean_df = pd.concat(stage_mean, axis=1)
    gene_expr = gene_expr[stage_mean_df.min(axis=1) >= 10]
    genes = gene_expr.index.tolist()
    change_marks = []
    for f in range(len(types) - 1):
        itemsa = stages[stages['flag'] == types[f]].index
        itemsb = stages[stages['flag'] == types[f + 1]].index
        a = gene_expr.filter(items=itemsa, axis=1).to_numpy()
        b = gene_expr.filter(items=itemsb, axis=1).to_numpy()
        change_marks.append(DGE(a, b, len(genes)))
    mark_pd = pd.DataFrame(change_marks, columns=genes,
                           index=[str(types[t]) + 'v' + str(types[t + 1]) for t in range(len(types) - 1)]).T
    _m = len(types) - 1
    genes = mark_pd[
        (mark_pd.sum(axis=1) == (_m - 1))
        | (mark_pd.sum(axis=1) == -(_m - 1))
        | (mark_pd.sum(axis=1) == -_m)
        | (mark_pd.sum(axis=1) == _m)
        ].index.tolist()
    res = pd.DataFrame(genes)
    res.to_csv(path)
    return genes


def DGE(a: np.ndarray, b: np.ndarray, genesize):
    mark = []
    for g in range(genesize):
        l = 0
        _a, _b = a[g, :], b[g, :]
        s1, p1 = stats.ranksums(_a, _b)
        ma, mb = np.mean(_a), np.mean(_b)
        _fc = ma / mb
        if p1 < p and s1 > 0 and np.log2(_fc) > np.log2(fc):
            l = 1
        if p1 < p and s1 < 0 and np.log2(_fc) < -np.log2(fc):
            l = -1
        mark.append(l)
    return mark
