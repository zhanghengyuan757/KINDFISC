import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cupy as cp
from tools.PathTools import LogPathGetter, TempPathGetter


def get_graph_longest_path(cancer, fs, threshold, stage, expr, title, dag_type='small_c'):
    lpg = LogPathGetter()
    r_path = lpg.get_path(title, 'seq.csv')
    rdata_path = lpg.get_path(title, 'rdata.csv')
    if os.path.exists(r_path):
        seq = pd.read_csv(r_path, index_col=0)
        rdata = expr.filter(items=seq.index, axis=1)
        return seq, rdata
    print(cancer, '去掉闭环')
    if dag_type == 'score':
        g = get_dag(cancer, fs, threshold, title)
    if dag_type == 'small_c':
        g = get_dag_1(cancer, fs, threshold, title)
    if dag_type == 'big_c':
        g = get_dag_2(cancer, fs, threshold, title)
    seq = nx.algorithms.dag_longest_path(g)
    seq = stage.filter(items=seq, axis=0)
    # seq = seq[seq['flag'] != 0]
    rdata = expr.filter(items=seq.index, axis=1)
    seq.to_csv(r_path)
    rdata.to_csv(rdata_path)
    print(cancer, '去掉闭环运行完毕')
    return seq, rdata


def get_dag_by_predit_prob(cancer):
    lpg = LogPathGetter()
    matrix: pd.DataFrame = pd.read_excel(lpg.get_path(cancer, "vote_matrix.xlsx"), index_col=0)
    proba_matrix: pd.DataFrame = pd.read_excel(lpg.get_path(cancer, "probability_matrix.xlsx"), index_col=0)
    matrix[matrix == 2] = 1
    g = nx.DiGraph(matrix)
    for n in g.copy().nodes():
        if g.degree[n] == 0:
            g.remove_node(n)
    loops = 0

    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        cycle_prob = [abs(proba_matrix.loc[c[0], c[1]] - 0.5) for c in cycle]
        cycle = list(zip(cycle, cycle_prob))
        cycle.sort(key=lambda a: a[1])
        if len(cycle) > 2:
            cycle = cycle[:1]
        elif abs(cycle_prob[0] - cycle_prob[1]) < 0.01:
            cycle = cycle[:1]
        for c in cycle:
            g.remove_edge(*c[0])
        loops += 1
    print(cancer, '总计去环迭代', loops, '次')
    return g


def get_dag_by_spearman(cancer, fs, threshold):
    lpg = LogPathGetter()
    tpg = TempPathGetter()
    edge_path = tpg.get_path(cancer, fs, 'edges.csv')
    nodes_path = tpg.get_path(cancer, fs, 'nodes.csv')
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(nodes_path, index_col=0)
    matrix: pd.DataFrame = pd.read_excel(lpg.get_path(cancer, "vote_matrix.xlsx"), index_col=0)
    matrix = matrix.filter(items=nodes.index, axis=0).filter(items=nodes.index, axis=1)
    edges = edges[edges['spearman'] >= threshold][['i', 'j', 'spearman']]
    spearman_map = {}
    for _, l in edges.iterrows():
        i, j, spearman = int(l['i']), int(l['j']), l['spearman']
        i, j = nodes.index[i], nodes.index[j]
        spearman_map[str(i) + '-' + str(j)] = spearman
    matrix[matrix == 2] = 1
    g = nx.DiGraph(matrix)
    for n in g.copy().nodes():
        if g.degree[n] == 0:
            g.remove_node(n)
    flow_hierarchy = nx.flow_hierarchy(g)
    print(cancer, 'flow_hierarchy', flow_hierarchy)
    loops = 0
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        cycle_spear = []
        for c in cycle:
            ij = str(c[0]) + '-' + str(c[1])
            ji = str(c[1]) + '-' + str(c[0])
            if ij in spearman_map.keys():
                cycle_spear.append(spearman_map[ij])
            else:
                cycle_spear.append(spearman_map[ji])
        cycle = list(zip(cycle, cycle_spear))
        cycle.sort(key=lambda a: a[1])
        if len(cycle) > 2:
            cycle = cycle[:1]
        elif abs(cycle_spear[0] - cycle_spear[1]) < 0.01:
            cycle = cycle[:1]
        for c in cycle:
            g.remove_edge(*c[0])
        loops += 1
    print(cancer, '总计去环迭代', loops, '次')
    return g


def get_dag(cancer, fs, threshold, title, show_plot=False):
    lpg = LogPathGetter()
    tpg = TempPathGetter()
    edge_path = tpg.get_path(cancer, fs, 'edges.csv')
    nodes_path = tpg.get_path(cancer, fs, 'nodes.csv')
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(nodes_path, index_col=0)
    edges = edges[edges['spearman'] >= threshold][['i', 'j', 'spearman']]
    matrix: pd.DataFrame = pd.read_excel(lpg.get_path(title, "vote_matrix.xlsx"), index_col=0)
    matrix = matrix.filter(items=nodes.index, axis=0).filter(items=nodes.index, axis=1)
    spearman_map = {}
    max_spear, min_spear = -1, 2
    dout = np.sum(matrix, axis=1)  # D_out
    din = np.sum(matrix, axis=0)
    dmax = np.max(dout + din)
    for _, l in edges.iterrows():
        i, j, spearman = int(l['i']), int(l['j']), l['spearman']
        spearman_map[str(i) + '-' + str(j)] = spearman
        if spearman > max_spear:
            max_spear = spearman
        if spearman < min_spear:
            min_spear = spearman
    matrix[matrix == 2] = 1
    A = cp.asarray(matrix.to_numpy())
    # A = np.array([[0, 1, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0],
    #      ])
    loops = 0
    cyc_A = A
    cyc_A_index = range(A.shape[0])
    while True:
        An_i = cp.identity(cyc_A.shape[0], int)
        _A_r = cp.zeros(cyc_A.shape, int)
        for i in range(1, cyc_A.shape[0]):
            An_i = cp.matmul(An_i, cyc_A)
            An_i[An_i > 0] = 1
            _A_r += An_i * cyc_A.T
            _A_r[_A_r > 0] = 1
            # print(i, _A_r.sum())
        _A_r = _A_r.T
        if cp.sum(_A_r) == 0:
            break
        else:
            cyc = cp.where(_A_r == 1)
            links_in_cyc = list(zip(cyc[0].tolist(), cyc[1].tolist()))
            scores = []
            dout = cp.sum(_A_r, axis=1)  # D_out
            din = cp.sum(_A_r, axis=0)
            dout_l = dout.tolist()  # D_in
            din_l = din.tolist()  # D_in
            for i, j in links_in_cyc:
                oi, oj = cyc_A_index[i], cyc_A_index[j]
                ij = str(oi) + '-' + str(oj)
                ji = str(oj) + '-' + str(oi)
                if ij in spearman_map.keys():
                    spearman = spearman_map[ij]
                else:
                    spearman = spearman_map[ji]
                di = dout_l[i] + din_l[i]
                dj = dout_l[j] + din_l[j]
                score = (1 + (spearman - min_spear) / (max_spear - min_spear))
                score = score * (1 + (di + dj - 2) / (dmax - 2))
                # score = score * (di + dj)
                # similarity higher more possible got wrong prediction
                scores.append({'score': float(score)})

            score_df = pd.DataFrame(scores)
            score_df = score_df.sort_values(by='score', ascending=True)
            # del_links_index = score_df.iloc[:20, :].index
            del_links_index = score_df.iloc[:100, :].index
            if score_df.index.size < 10000:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.005), :].index
            if score_df.index.size < 1000:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.01), :].index
            if score_df.index.size < 100:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.1), :].index
            if score_df.index.size < 10:
                del_links_index = score_df.iloc[0:1, :].index
            loops += 1
            del_links = [links_in_cyc[int(i)] for i in del_links_index]
            if show_plot and score_df.index.size < 10:
                g = nx.DiGraph(pd.DataFrame(cp.asnumpy(_A_r)))
                for n in g.copy().nodes():
                    if g.degree[n] == 0:
                        g.remove_node(n)
                nx.draw(g, node_size=100, node_color='b', edge_color=['r' if e in del_links else 'b' for e in g.edges])
                plt.show()  # 显示
                g.clear()
            for del_link in del_links:
                cyc_A[del_link] = 0
                i, j = del_link
                i, j = cyc_A_index[i], cyc_A_index[j]
                A[i, j] = 0
            d = dout + din
            cyc_A = cyc_A[d != 0, :][:, d != 0]
            cyc_A_index = np.array(cyc_A_index)[(d != 0).get()].tolist()
    print(cancer, '总计去环迭代', loops, '次')
    g = nx.DiGraph(pd.DataFrame(cp.asnumpy(A), index=nodes.index, columns=nodes.index))
    assert nx.is_directed_acyclic_graph(g)
    return g


def get_dag_1(cancer, fs, threshold, title):
    lpg = LogPathGetter()
    tpg = TempPathGetter()
    edge_path = tpg.get_path(cancer, fs, 'edges.csv')
    nodes_path = tpg.get_path(cancer, fs, 'nodes.csv')
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(nodes_path, index_col=0)
    edges = edges[edges['spearman'] >= threshold][['i', 'j', 'spearman']]
    matrix: pd.DataFrame = pd.read_excel(lpg.get_path(title, "vote_matrix.xlsx"), index_col=0)
    matrix = matrix.filter(items=nodes.index, axis=0).filter(items=nodes.index, axis=1)
    spearman_map = {}
    max_spear, min_spear = -1, 2
    dout = np.sum(matrix, axis=1)  # D_out
    din = np.sum(matrix, axis=0)
    dmax = np.max(dout + din)
    din, dout = din.tolist(), dout.tolist()
    for _, l in edges.iterrows():
        i, j, spearman = int(l['i']), int(l['j']), l['spearman']
        spearman_map[str(i) + '-' + str(j)] = spearman
        if spearman > max_spear:
            max_spear = spearman
        if spearman < min_spear:
            min_spear = spearman
    matrix[matrix == 2] = 1
    A = cp.asarray(matrix.to_numpy())
    loops = 0
    for ai in range(1, A.shape[0]):
        if nx.is_directed_acyclic_graph(nx.DiGraph(A.get())):
            break
        while True:
            _A_r = cp.zeros(A.shape, int)
            An_i = cp.identity(A.shape[0], int)
            for _ in range(ai):
                An_i = cp.matmul(An_i, A)
            An_i[An_i > 0] = 1
            _A_r += An_i * A.T
            _A_r[_A_r > 0] = 1
            _A_r = _A_r.T
            cyc = cp.where(_A_r == 1)
            links_in_cyc = list(zip(cyc[0].tolist(), cyc[1].tolist()))
            if len(links_in_cyc) == 0:
                break

            scores = []
            for i, j in links_in_cyc:
                ij = str(i) + '-' + str(j)
                ji = str(j) + '-' + str(i)
                if ij in spearman_map.keys():
                    spearman = spearman_map[ij]
                else:
                    spearman = spearman_map[ji]
                di = dout[i] + din[i]
                dj = dout[j] + din[j]
                score = (1 + (spearman - min_spear) / (max_spear - min_spear))
                score = score * (1 + (di + dj - 2) / (dmax - 2))
                scores.append({'score': float(score)})
            score_df = pd.DataFrame(scores)
            score_df = score_df.sort_values(by='score', ascending=True)
            del_links_index = score_df.iloc[:50, :].index
            if score_df.index.size < 10000:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.005), :].index
            if score_df.index.size < 1000:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.01), :].index
            if score_df.index.size < 100:
                del_links_index = score_df.iloc[:int(score_df.index.size * 0.1), :].index
            if score_df.index.size < 10:
                del_links_index = score_df.iloc[0:1, :].index
            loops += 1
            del_links = [links_in_cyc[int(i)] for i in del_links_index]
            for del_link in del_links:
                A[del_link] = 0

    print(cancer, '总计去环迭代', loops, '次')
    g = nx.DiGraph(pd.DataFrame(cp.asnumpy(A), index=nodes.index, columns=nodes.index))
    assert nx.is_directed_acyclic_graph(g)
    return g


def get_dag_2(cancer, fs, threshold, title):
    lpg = LogPathGetter()
    tpg = TempPathGetter()
    edge_path = tpg.get_path(cancer, fs, 'edges.csv')
    nodes_path = tpg.get_path(cancer, fs, 'nodes.csv')
    edges = pd.read_csv(edge_path, index_col=0)
    nodes = pd.read_csv(nodes_path, index_col=0)
    edges = edges[edges['spearman'] >= threshold][['i', 'j', 'spearman']]
    matrix: pd.DataFrame = pd.read_excel(lpg.get_path(title, "vote_matrix.xlsx"), index_col=0)
    matrix = matrix.filter(items=nodes.index, axis=0).filter(items=nodes.index, axis=1)
    spearman_map = {}
    max_spear, min_spear = -1, 2
    dout = np.sum(matrix, axis=1)  # D_out
    din = np.sum(matrix, axis=0)
    dmax = np.max(dout + din)
    din, dout = din.tolist(), dout.tolist()
    for _, l in edges.iterrows():
        i, j, spearman = int(l['i']), int(l['j']), l['spearman']
        spearman_map[str(i) + '-' + str(j)] = spearman
        if spearman > max_spear:
            max_spear = spearman
        if spearman < min_spear:
            min_spear = spearman
    matrix[matrix == 2] = 1
    A = cp.asarray(matrix.to_numpy())
    loops = 0
    if nx.is_directed_acyclic_graph(nx.DiGraph(cp.asnumpy(A))):
        return nx.DiGraph(pd.DataFrame(cp.asnumpy(A), index=nodes.index, columns=nodes.index))
    max_loop_len = list(getcycA(A).keys())[-1]
    while True:
        cyc_A = getcycA(A, max_loop_len=max_loop_len)
        if cyc_A == {}:
            break
        _A_r = cyc_A[list(cyc_A.keys())[-1]]
        cyc = cp.where(_A_r == 1)
        links_in_cyc = list(zip(cyc[0].tolist(), cyc[1].tolist()))
        scores = []
        for i, j in links_in_cyc:
            ij = str(i) + '-' + str(j)
            ji = str(j) + '-' + str(i)
            if ij in spearman_map.keys():
                spearman = spearman_map[ij]
            else:
                spearman = spearman_map[ji]
            di = dout[i] + din[i]
            dj = dout[j] + din[j]
            score = (1 + (spearman - min_spear) / (max_spear - min_spear))
            score = score * (1 + (di + dj - 2) / (dmax - 2))
            scores.append({'score': float(score)})
        score_df = pd.DataFrame(scores)
        score_df = score_df.sort_values(by='score', ascending=True)
        del_links_index = score_df.iloc[:50, :].index
        if score_df.index.size < 10000:
            del_links_index = score_df.iloc[:int(score_df.index.size * 0.005), :].index
        if score_df.index.size < 1000:
            del_links_index = score_df.iloc[:int(score_df.index.size * 0.01), :].index
        if score_df.index.size < 100:
            del_links_index = score_df.iloc[:int(score_df.index.size * 0.1), :].index
        if score_df.index.size < 10:
            del_links_index = score_df.iloc[0:1, :].index
        loops += 1
        del_links = [links_in_cyc[int(i)] for i in del_links_index]
        for del_link in del_links:
            A[del_link] = 0
    print(cancer, '总计去环迭代', loops, '次')
    g = nx.DiGraph(pd.DataFrame(cp.asnumpy(A), index=nodes.index, columns=nodes.index))
    assert nx.is_directed_acyclic_graph(g)
    return g


def get_del_cyc(score_df):
    del_links_index = score_df.iloc[:50, :].index
    if score_df.index.size < 10000:
        del_links_index = score_df.iloc[:int(score_df.index.size * 0.005), :].index
    if score_df.index.size < 1000:
        del_links_index = score_df.iloc[:int(score_df.index.size * 0.01), :].index
    if score_df.index.size < 100:
        del_links_index = score_df.iloc[:int(score_df.index.size * 0.1), :].index
    if score_df.index.size < 10:
        del_links_index = score_df.iloc[0:1, :].index
    return del_links_index


# sum([cp.sum(a) for a in list(cycA.values())])
def getcycA(A, max_loop_len=-1):
    if max_loop_len == -1:  # max_loop_len unknown
        _A_r = cp.zeros(A.shape, int)
        An_i = cp.identity(A.shape[0], int)
        for _ in range(A.shape[0]):
            An_i = cp.matmul(An_i, A)
            An_i[An_i > 0] = 1
            _A_r += An_i * A.T
        _A_r[_A_r > 0] = 1
        cyc_link_sum = _A_r.sum()
        max_loop_len = A.shape[0]
    else:
        cyc_link_sum = -1

    _A_r = cp.zeros(A.shape, int)
    An_i = cp.identity(A.shape[0], int)
    cycA = {}
    for i in range(max_loop_len + 1):
        An_i = cp.matmul(An_i, A)
        An_i[An_i > 0] = 1
        _A_r += An_i * A.T
        _A_r[_A_r > 0] = 1
        if _A_r.sum() != 0:
            if i - 1 in cycA.keys():
                sums = sum(list(cycA.values()))
                sums[sums > 0] = 1
                cycA[i] = _A_r.T - sums
                if cp.sum(cycA[i]) == 0:
                    cycA.pop(i)
            else:
                cycA[i] = _A_r.copy().T
        if cyc_link_sum != -1 and _A_r.sum() == cyc_link_sum:
            break
    return cycA
