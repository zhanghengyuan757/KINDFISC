import pandas as pd

import model_fit
from arnn.LongerPredictionSamples_ARNN import get_arnn
from model_fit.DLClassifier import BaseTorchDl
from processing.DataPrepare import gene_select
from ti.PatientSeqRank import get_dag_longest_path_rank, tipping_point_plot, single_cell, single_cell_raw
from ti.PatientSort import SortResultMatrix
from tools.GraphTools import get_graph_longest_path
from tools.PathTools import get_title


def piplines(cancer, fs, threshold):
    model_fit.BaseCL.k_fold = 5

    # 通过癌症类型获得基因表达矩阵和分期信息
    expr, stage = gene_select(cancer, fs, verbose=True)
    # 模型预测
    classifier = BaseTorchDl(cancer, fs, threshold, expr, stage)
    title = get_title(cancer, fs, threshold)
    reports1 = []
    reports2 = []
    for k in range(model_fit.BaseCL.k_fold):
        report1, report2 = classifier.fit_pred(k)
        reports1.append(report1)
        reports2.append(report2)
        if model_fit.BaseCL.k_fold > 1:
            title = get_title(cancer, fs, threshold, k + 1, 'fold')
        all_pred = classifier.pred_all()
        # # 保存投票矩阵
        srm = SortResultMatrix(all_pred, stage)
        srm.save_result(srm.sorted_index(), title)
        sorted_list, sorted_rdata = get_graph_longest_path(cancer, fs, threshold, stage, expr, title)
        cancer_list = sorted_list[sorted_list['flag'] != 0]
        assert sorted_list.index.size != 0
        cancer_rdata = sorted_rdata.loc[:, cancer_list.index]
        # srm.draw_matrix(title, cancer_list)
        get_dag_longest_path_rank(cancer_list, cancer_rdata,
                                  'Time series distribution of ' + cancer.upper() + " patients" + str(
                                      k + 1) + 'fold')
        print('结果序列长度：', len(cancer_list))
    reports1 = pd.concat(reports1,axis=1).mean(axis=1)
    reports2 = pd.concat(reports2,axis=1).mean(axis=1)
    print('I-II\II-III ', model_fit.BaseCL.k_fold, '折macro avg 均值')
    print(reports1)
    print('IA-IB\IIA-IIB', model_fit.BaseCL.k_fold, '折macro avg 均值')
    print(reports2)

    title = get_title(cancer, fs, threshold)
    print('=' * 20, '使用全部样本训练', '=' * 20)
    model_fit.BaseCL.k_fold = 1
    model_fit.BaseCL.per_I_IV = 1
    model_fit.BaseCL.per_IA_IVC = 1
    classifier = BaseTorchDl(cancer, fs, threshold, expr, stage)
    classifier.fit_pred()
    all_pred = classifier.pred_all()
    # # 保存投票矩阵
    srm = SortResultMatrix(all_pred, stage)
    srm.save_result(srm.sorted_index(), title)
    sorted_list, sorted_rdata = get_graph_longest_path(cancer, fs, threshold, stage, expr, title)
    cancer_list = sorted_list[sorted_list['flag'] != 0]
    assert sorted_list.index.size != 0
    cancer_rdata = sorted_rdata.loc[:, cancer_list.index]
    # srm.draw_matrix(title, cancer_list)
    get_dag_longest_path_rank(cancer_list, cancer_rdata,
                              'Time series distribution of ' + cancer.upper() + " patients")
    print('结果序列长度：', len(cancer_list))

    raw_matrix = gene_select(cancer, None, verbose=False)[0]
    raw_matrix_seq = raw_matrix.filter(items=sorted_list.index, axis=1)
    #
    print(cancer, 'E-SGAP：')
    single_cell_raw(expr, stage) # need R environment
    #
    # print(cancer, 'E-AGAP：')
    # single_cell_raw(raw_matrix, stage) # need R environment with Dynoverse
    #
    print(cancer, 'E-AGSP：')
    single_cell_raw(raw_matrix_seq, sorted_list)  # need R environment with Dynoverse

    print('spearman threshold:', threshold, cancer, 'E-SGSP：')
    single_cell(sorted_rdata, sorted_list) # need R environment with Dynoverse

    assert sorted_list.index.size != 0
    cancer_rdata = sorted_rdata.loc[:, cancer_list.index]
    print('=' * 20, 'ARNN', '=' * 20)
    get_arnn(cancer, cancer_rdata, title)
    print('=' * 20, 'Tipping point', '=' * 20)
    tipping_point_plot(cancer, cancer_list, cancer_rdata, fs, by_stage=True)
    tipping_point_plot(cancer, cancer_list, cancer_rdata, fs, by_stage=False)


if __name__ == '__main__':
    piplines('luad', 'hsic&dge', 0.81)
    print('+' * 50)
    piplines('brca', 'hsic&dge', 0.83)

