import warnings

import numpy as np
import pandas as pd
import sklearn.metrics.cluster
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ldnb.lDNB_v1 import tipping_point
from processing.DataPrepare import gene_select
from tools.PathTools import ChartPathGetter, DataPathGetter, LogPathGetter, PathGetter
import seaborn as sns
import rpy2.robjects as robjects
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.packages import importr


def get_right_percent(df: pd.DataFrame):
    import itertools
    barcodes = df.index.tolist()
    perms = list(itertools.combinations(barcodes, 2))  # C(n，2) n为样本个数
    sum_n = len(perms)
    right_pair_num = 0
    stages = df['stage'].tolist()
    for p in perms:
        i, j = p[0], p[1]
        index_i, index_j = barcodes.index(i), barcodes.index(j)
        sort_boolean = index_i < index_j
        stage_boolean = stages[index_i] <= stages[index_j]
        if sort_boolean == stage_boolean:
            right_pair_num += 1
    return round(100. * right_pair_num / sum_n, 2)


# 按最原始的标签来排序
def get_bucket_top_k(df: pd.DataFrame):
    stages = set(df['stage'].tolist())
    df = df.reset_index(drop=True)
    patient_sorted_by_stage = df.sort_values('stage').reset_index(drop=True)
    thick_hit = 0
    # 每个分期内样本分桶，查看桶内命中个数
    for stage in stages:
        sorted_index = patient_sorted_by_stage[patient_sorted_by_stage['stage'] == stage].index
        picked_df = df.filter(items=sorted_index, axis=0)
        thick_hit += picked_df[picked_df['stage'] == stage].index.size
    return round(100. * thick_hit / df.index.size, 2)


def spearman(df: pd.DataFrame):
    le = LabelEncoder()
    stages = sorted(list(df['stage'].tolist()))
    le.fit(stages)
    r = pd.Series(le.transform(stages)).corr(pd.Series(le.transform(df['stage'])), method='spearman')
    return round(r, 2)


def get_dyno_spear(sorted_list, dyno_list):
    stage_spearman = spearman(dyno_list.replace('N', 'A'))
    _sorted_list = sorted_list
    _sorted_list['score'] = dyno_list['score']
    # 以dyno的打分为时间scale 评估spearman相关性
    dyno_spear = _sorted_list['score'].reset_index(drop=True).corr(dyno_list['score'].reset_index(drop=True),
                                                                   method='spearman')
    return stage_spearman, dyno_spear


def gene_spearman(rdata: pd.DataFrame):
    x2 = pd.Series(range(rdata.columns.size))
    result = []
    for i in range(rdata.index.size):
        x1 = rdata.iloc[i, :].reset_index(drop=True)
        spearman_r = round(x1.corr(x2, method='spearman'), 2)
        result.append(spearman_r)
    result = pd.DataFrame(result, index=rdata.index, columns=['spearman'])
    return result


def get_dag_longest_path_rank(_sorted_list: pd.DataFrame, sorted_rdata, file_name):
    sorted_list = _sorted_list
    lpg = LogPathGetter()
    # print(file_name, 'sample_pairs_right_percent:', get_right_percent(sorted_list))
    print(file_name, 'sample_in_bucket_percent:', get_bucket_top_k(sorted_list))
    print(file_name, 'sample_label_spearman:', spearman(sorted_list))
    gene_spearman(sorted_rdata).to_excel(lpg.get_path(file_name, 'gene_spearman', '.xlsx'))
    sorted_list['stage2'] = LabelEncoder().fit_transform(sorted_list['stage'])
    ax = plt.scatter(range(1, sorted_list.index.size + 1), sorted_list['stage2'], c=np.array(sorted_list['stage2']),
                     cmap='Dark2')
    stages = sorted(sorted_list['stage'].drop_duplicates())
    # stages[0] = 'Normal'
    plt.yticks(sorted(sorted_list['stage2'].drop_duplicates()), stages)
    plt.title(file_name)
    plt.tight_layout()
    plt.show()


def mutation_plot(cancer, sorted_list, by_stage=False):
    mutation = pd.read_csv(DataPathGetter(cancer).get_path(cancer.upper(), 'mutation_num.csv'), index_col=0)
    _sorted_list = sorted_list
    print(cancer, ":", _sorted_list.index.size)
    _sorted_list.index = [i[:12] for i in _sorted_list.index]
    mutation = mutation.filter(items=_sorted_list.index, axis=0)
    print(cancer, "(samples in mutation result):", mutation.index.size)
    plt.figure(figsize=(8, 4))
    stages = _sorted_list.filter(items=mutation.index, axis=0)['stage']
    mutation['stage'] = stages
    lpg = LogPathGetter()
    if by_stage:
        mutation = mutation.groupby(['stage']).mean()
        mutation.to_csv(lpg.get_path(cancer, 'mutation_stage.csv'))
        plt.xticks(range(mutation.index.size), mutation.index)
        plt.plot(mutation['x'])
        # mutation = mutation['x'].rolling(5, min_periods=5).mean()
        # plt.xticks(range(mutation.index.size), range(mutation.index.size))
        # plt.plot(mutation)
        plt.title(cancer + ' mutation index')
        plt.show()
    else:
        colors = LabelEncoder().fit_transform(stages)
        scatter = plt.scatter(range(1, mutation.index.size + 1), mutation['x'], c=np.array(colors),
                              cmap='Dark2')
        mutation.to_csv(lpg.get_path(cancer, 'mutation_granular.csv'))
        plt.legend(handles=scatter.legend_elements()[0], labels=stages.drop_duplicates().tolist())
        plt.title(cancer + ' mutation index')
        plt.show()


def tipping_point_plot(cancer, cancer_list: pd.DataFrame, cancer_data: pd.DataFrame, fs, by_stage=False):
    expr, stage = gene_select(cancer, fs, verbose=False, stages=[0])
    _sorted_list = pd.concat([stage, cancer_list], axis=0)
    _sorted_data = pd.concat([expr, cancer_data], axis=1)
    title = None
    if not by_stage:
        title = cancer
    tp: pd.Series = tipping_point(_sorted_list, _sorted_data, title)
    plt.figure(figsize=(10, 5))
    lpg = LogPathGetter()
    if by_stage:
        tp = tp.reset_index().groupby(['stage']).mean().iloc[:, 0]
        plt.xticks(range(tp.index.size), tp.index)
        tp.to_csv(lpg.get_path(cancer, 'tp_stage.csv'))
        plt.plot(tp.tolist())
        plt.title(cancer + ' tipping point(L-DNB)')
        plt.show()
    else:
        tp.to_csv(lpg.get_path(cancer, 'tp_granular.csv'))
        colors = LabelEncoder().fit_transform(tp.index)
        scatter = plt.scatter(range(1, tp.index.size + 1), tp, c=np.array(colors),
                              cmap='Dark2')
        plt.legend(handles=scatter.legend_elements()[0], labels=tp.index.drop_duplicates().tolist())
        plt.title(cancer + ' tipping point(L-DNB)')
        plt.show()


def cluster_rank(cancer, sorted_list: pd.DataFrame, sorted_data: pd.DataFrame):
    y_true = LabelEncoder().fit_transform(sorted_list['stage'])
    model = AgglomerativeClustering(linkage='ward', n_clusters=sorted_list['stage'].drop_duplicates().size)
    y = model.fit_predict(sorted_data.T, y_true)
    score = adjusted_rand_score(y_true, y)
    print(cancer, '子序列adjusted_rand_score:', score)


def classifier_rank(cancer, sorted_list: pd.DataFrame, sorted_data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(sorted_data.T,
                                                        LabelEncoder().fit_transform(sorted_list['stage']),
                                                        test_size=0.1,
                                                        random_state=0)  # 交叉验证划分训练集和测试集.test_size为测试集所占的比例
    print(cancer, '子序列训练集大小：', X_train.shape, y_train.shape)  # 训练集样本大小
    print(cancer, '子序列测试集大小：', X_test.shape, y_test.shape)  # 测试集样本大小
    clf = RandomForestClassifier().fit(X_train, y_train)  # 使用训练集训练模型
    print(cancer, '子序列准确率：', clf.score(X_test, y_test))  # 计算测试集的度量值（准确率）


def single_cell(expr, stage):
    cancer_stage = stage[stage['flag'] != 0]
    importr('dyno')
    importr('tidyverse')
    for single_cell_method in ['slingshot', 'mst', 'slice']:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv['expr'] = robjects.conversion.rpy2py(expr)
            robjects.globalenv['stage'] = robjects.conversion.rpy2py(stage)
        try:
            robjects.r("method = '%s'" % single_cell_method)
            robjects.r.source('r/dyno.R')
        except Exception as e:
            print(single_cell_method, '报错')
            continue

        seq1 = list(robjects.r('result').names)
        seq1 = list(map(lambda a: a.replace('.', '-'), seq1))
        new_list = stage.filter(items=seq1, axis=0)
        new_list['score'] = list(robjects.r('result'))
        stage_spearman, dyno_spears = get_dyno_spear(stage, new_list)
        print(single_cell_method, '分期label的spearman:', stage_spearman)
        print(single_cell_method, '与框架计算score的spearman:', dyno_spears)
        # stage_spearman, dyno_spears = get_dyno_spear(cancer_stage, new_list[new_list['flag'] != 0])
        # print(single_cell_method, '剔除正常样本后，分期label的spearman:', stage_spearman)
        # print(single_cell_method, '剔除正常样本后，与框架计算score的spearman:', dyno_spears)


def single_cell_raw(expr, stage):
    importr('dyno')
    importr('tidyverse')
    for single_cell_method in ['slingshot', 'mst', 'slice']:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv['expr'] = robjects.conversion.rpy2py(expr)
            robjects.globalenv['stage'] = robjects.conversion.rpy2py(stage)
        try:
            robjects.r("method = '%s'" % single_cell_method)
            robjects.r.source('r/dyno.R')
        except Exception as e:
            print(single_cell_method, '报错')
            continue
        seq1 = list(robjects.r('result').names)
        seq1 = list(map(lambda a: a.replace('.', '-'), seq1))
        new_list = stage.filter(items=seq1, axis=0)
        new_list['score'] = list(robjects.r('result'))
        print(single_cell_method, '原始数据集 分期label的spearman:', spearman(new_list.replace('N', 'A')))
        # print(single_cell_method, '剔除正常样本后，原始数据集 分期label的spearman:', spearman(new_list[new_list['flag'] != 0]))


def prepare_prob_rank(cancer, stage, label='stage'):
    raw_matrix = gene_select(cancer, None, verbose=False)[0]
    raw_matrix_seq = raw_matrix.filter(items=stage.index, axis=1)
    stage = stage[stage['stage'] != 'N']
    le = LabelEncoder()
    le.fit(sorted(stage[label].drop_duplicates()))
    raw_matrix_seq = raw_matrix_seq.filter(items=stage.index, axis=1)
    labels = le.transform(stage[label])
    raw_matrix_seq: pd.DataFrame = raw_matrix_seq.append(pd.Series(labels, name='stage', index=stage.index) + 1)
    raw_matrix_seq.to_csv('../files/prob/' + cancer + '_matrix.csv', index=False)
    stage.to_csv('../files/prob/' + cancer + '_stage.csv')


def prob_rank(cancer):
    bpg = PathGetter('prob')
    indexs = pd.read_csv(bpg.get_path(cancer, 'result.csv'), header=None)
    indexs = (indexs - 1).iloc[:, 0].tolist()
    stages = pd.read_csv(bpg.get_path(cancer, 'stage.csv'))
    stages = stages.iloc[indexs, :]
    stages['stage'].to_csv(bpg.get_path(cancer, 'prob_stage.csv'))
    print(cancer, '_prob_sample_in_bucket_percent:', get_bucket_top_k(stages))
    print(cancer, '_prob_sample_label_spearman:', spearman(stages))
    r = pd.Series(indexs).corr(
        pd.Series(range(1, len(indexs) + 1)), method='spearman')
    print(cancer, '_prob_dist_spearman:', r)


if __name__ == '__main__':
    lpg = LogPathGetter()
    luad_seq = pd.read_csv(lpg.get_path('luad_hsic&dge_0.81_seq.csv'),
                           index_col=0)
    brca_seq = pd.read_csv(lpg.get_path('brca_hsic&dge_0.83_seq.csv'),
                           index_col=0)

    # prepare_prob_rank('luad', luad_seq)
    # prepare_prob_rank('brca', brca_seq)

    prob_rank('luad')
    prob_rank('brca')
