import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder

from gene_select.MonotonicGenes import find_mono_genes
from gene_select.pyHiscLasso import hsic_lasso
from tools.PathTools import DataPathGetter, TempPathGetter
import matplotlib.pyplot as plt
from sklearn import manifold

random_state = 666


def gene_select(cancer: str, feature_selection_method, verbose=True, stages=[0, 1, 2, 3]):
    dpg = DataPathGetter(cancer)
    expr = pd.read_csv(dpg.get_path('gene_expr_fpkm.csv'), index_col=0)
    stage = pd.read_csv(dpg.get_path('cancer_stage.csv'), index_col=0)
    stage = stage[stage['flag'].isin(stages)]
    expr = expr.filter(items=map(str, stage.index), axis=1)
    if verbose:
        print(cancer, "基因总共有：", expr.index.size)
    expr = expr.filter(items=expr[expr.mean(axis=1) > 10].index, axis=0)  # 去掉样本（exclude Stage IV）均值小于10的特征
    # a=pd.DataFrame(expr.index)
    if verbose:
        print(cancer, "去掉样本均值小于10的基因总共有：", expr.index.size)

    hl_genes = hsic_lasso(cancer, expr.index)
    if verbose:
        print(cancer, "经过HSIC-Lasso筛选后的基因总共有：", len(hl_genes))
    genes = find_mono_genes(cancer, expr.index)
    if verbose:
        print(cancer, "符合癌症发展规律的基因总共有：", len(genes))
    if feature_selection_method == 'dge':
        expr = expr.filter(items=genes, axis=0)
    elif feature_selection_method == 'hsic':
        expr = expr.filter(items=hl_genes, axis=0)
    elif feature_selection_method in ['hsic&dge', 'dge&hsic']:
        expr = expr.filter(items=set(hl_genes) & set(genes), axis=0)
        if verbose:
            path = TempPathGetter().get_path(cancer, 'hsic_lasso_genes.csv')
            df = pd.read_csv(path, header=0, index_col=0)
            df = df[df['0'].isin(set(hl_genes) & set(genes))]
            df = df.sort_values(by=['1'], ascending=False)
            df['0'] = list(map(lambda s: s.split('|')[0], df['0']))
            df.to_csv(dpg.get_path('cancer_related_genes.csv'))
            print(df.head(10))
    elif feature_selection_method == 'dge-hsic':
        expr = expr.filter(items=set(genes) - set(hl_genes), axis=0)
    elif feature_selection_method == 'hsic-dge':
        expr = expr.filter(items=set(hl_genes) - set(genes), axis=0)
    elif feature_selection_method == 'hsic+dge':
        expr = expr.filter(items=set(hl_genes) | set(genes), axis=0)
    elif feature_selection_method is None:
        pass
    else:
        raise Exception('this method has no implement:' + feature_selection_method)
    fn = expr.index.size
    if verbose:
        print(cancer, "最后特征总共有(", feature_selection_method, ")：", fn)
    summerize = stage.groupby(['stage']).count()
    if verbose:
        print(summerize)
    expr: pd.DataFrame = (expr + 1).apply(np.log2)
    expr = expr.filter(items=stage.index, axis=1)
    return expr, stage


def cal_similarity_net(cancer: str, feature_selection_method):
    expr, label = gene_select(cancer, feature_selection_method, verbose=False)
    dpg = TempPathGetter()
    path1 = dpg.get_path(cancer, feature_selection_method, 'nodes.csv')
    path2 = dpg.get_path(cancer, feature_selection_method, 'edges.csv')
    if os.path.exists(path1) and os.path.exists(path2):
        return
    size = expr.columns.size
    edges = []
    import time
    print(cancer, "similarity network calculating...")
    time_start = time.time()
    spearman = expr.corr(method='spearman').to_numpy()
    pearson = expr.corr(method='pearson').to_numpy()
    kendall = expr.corr(method='kendall').to_numpy()
    euclidean = expr.corr(method=lambda a, b: np.sqrt(np.sum(np.square(a - b)))).to_numpy()
    manhattan = expr.corr(method=lambda a, b: np.sum(np.abs(a - b))).to_numpy()
    chebyshev = expr.corr(method=lambda a, b: np.max(np.abs(a - b))).to_numpy()
    for i in range(1, size):
        for j in range(0, i):
            edges.append({'i': i, 'j': j,
                          'spearman': spearman[i][j], 'pearson': pearson[i][j], 'kendall': kendall[i][j],
                          'euclidean': euclidean[i][j], 'manhattan': manhattan[i][j], 'chebyshev': chebyshev[i][j]})
    time_end = time.time()
    calculate_time = time_end - time_start
    print(cancer, "similarity network calculate_time:", round(calculate_time, 2), 's')
    edges = pd.DataFrame(edges)
    edges['euclidean'] = edges['euclidean'] / (edges['euclidean'].max())
    edges['manhattan'] = edges['manhattan'] / (edges['manhattan'].max())
    edges['chebyshev'] = edges['chebyshev'] / (edges['chebyshev'].max())
    label.to_csv(path1)
    edges.to_csv(path2)


def get_directed_edges(cancer, feature_selection_method, threshold):
    tpg = TempPathGetter()
    node_path = tpg.get_path(cancer, feature_selection_method, 'nodes.csv')
    edge_path = tpg.get_path(cancer, feature_selection_method, 'edges.csv')
    if not (os.path.exists(node_path) and os.path.exists(edge_path)):
        cal_similarity_net(cancer, feature_selection_method)
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(node_path, index_col=0)
    stages = nodes['stage'].tolist()  # IA-IVC stage
    flags = nodes['flag'].tolist()  # I-IV stage
    edges = edges[edges['spearman'] >= threshold][['i', 'j']]
    edge_I_IV, edge_IA_IVC, edge_I = [], [], []
    for _, l in edges.iterrows():
        i, j = int(l['i']), int(l['j'])
        fi, fj = flags[i], flags[j]
        if fi > fj:  # edge direct using I-IV stage
            edge_I_IV.append({'i': i, 'j': j, 'y': 1})
            edge_I_IV.append({'i': j, 'j': i, 'y': 0})
        elif fi < fj:
            edge_I_IV.append({'i': i, 'j': j, 'y': 0})
            edge_I_IV.append({'i': j, 'j': i, 'y': 1})
        else:
            si, sj = stages[i], stages[j]
            if si > sj:  # edge direct using IA-IVC stage
                edge_IA_IVC.append({'i': i, 'j': j, 'y': 1})
                edge_IA_IVC.append({'i': j, 'j': i, 'y': 0})
            elif si < sj:
                edge_IA_IVC.append({'i': i, 'j': j, 'y': 0})
                edge_IA_IVC.append({'i': j, 'j': i, 'y': 1})
            else:
                edge_I.append({'i': i, 'j': j})
                edge_I.append({'i': j, 'j': i})
    edge_I_IV = pd.DataFrame(edge_I_IV, dtype=int)
    edge_IA_IVC = pd.DataFrame(edge_IA_IVC, dtype=int, columns=['i', 'j', 'y'])
    edge_I = pd.DataFrame(edge_I, dtype=int)
    return edge_I_IV, edge_IA_IVC, edge_I


def split_edges(edge_I_IV: pd.DataFrame, edge_IA_IVC: pd.DataFrame,
                frac_I_IV: float, frac_IA_IVC: float):
    train_y = edge_I_IV.sample(frac=frac_I_IV, random_state=random_state)

    test_y_I_IV = edge_I_IV.iloc[~edge_I_IV.index.isin(train_y.index), :]
    if frac_IA_IVC == 0 or edge_IA_IVC.empty or edge_IA_IVC is None:
        return train_y, test_y_I_IV, edge_IA_IVC

    train_y_IA_IVC = edge_IA_IVC.sample(frac=frac_IA_IVC, random_state=random_state)
    train_y = pd.concat([train_y, train_y_IA_IVC])
    test_y_IA_IVC = edge_IA_IVC.iloc[~edge_IA_IVC.index.isin(train_y_IA_IVC.index), :]
    return train_y, test_y_I_IV, test_y_IA_IVC


def kfold_split(k, y):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True)
    folds = []
    for train, test in kf.split(y):
        folds.append((train, test))
    return folds


def prepare_gae_data(train, test, unknown):
    assert type(train) == pd.DataFrame and type(test) == pd.DataFrame and type(unknown) == pd.DataFrame
    unknown = unknown[['i', 'j']]
    train = pd.concat([train, test], axis=0)
    train = train[train['y'] == 1][['i', 'j']]
    g = nx.DiGraph(pd.concat([train, unknown]).to_numpy().tolist())
    warnings.filterwarnings('ignore')
    g.remove_edges_from(unknown.to_numpy())
    g.remove_edges_from(test[['i', 'j']].to_numpy())
    adj = nx.adjacency_matrix(g)
    gk = list(g.nodes.keys())
    return adj, gk


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    coords, values, shape = sparse_to_tuple(sparse_mx)
    indices = torch.from_numpy(coords.transpose().astype(np.int64))
    values = torch.from_numpy(values.astype(np.float32))
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return adj_normalized
