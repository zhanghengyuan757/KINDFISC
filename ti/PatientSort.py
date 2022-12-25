# 滑动窗口对基因组数据进行排序
import random
from functools import cmp_to_key
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from tools.PathTools import ChartPathGetter, LogPathGetter

cpg = ChartPathGetter()
lpg = LogPathGetter()


class SortResultMatrix:
    matrix: np.matrix  # 投票矩阵
    prob_matrix: np.matrix  # 投票矩阵
    barcodes: List[str]  # 患者的barcode
    summarize_matrix: pd.DataFrame

    def __add__(self, other):
        if len(self.barcodes) != len(other.barcodes):
            return None
        for i in range(len(self.barcodes)):
            for j in range(len(self.barcodes)):
                self.__add_result2matrix(i, j, other.matrix[i][j],
                                         [other.prob_matrix[i][j], other.prob_matrix[j][i]])

    # !!!!这里定义行索引A列索引B 为1,代表A的样本比B大
    def __init__(self, edge_pred: pd.DataFrame, patient_label: pd.DataFrame):
        self.edge_pred = edge_pred
        size = len(patient_label)
        self.patient_label = patient_label
        self.barcodes = patient_label.index.tolist()  # 样本的Barcode对应到matrix上
        self.matrix = np.zeros((size, size), dtype=int)
        self.prob_matrix = np.zeros((size, size), dtype=float)
        sub_stages = patient_label['stage'].drop_duplicates().tolist()
        s = len(sub_stages)
        self.summarize_matrix = pd.DataFrame(np.zeros((s, s)), sub_stages, sub_stages)
        self.__get_model_result()
        print('打分矩阵构建完毕')

    def __get_model_result(self):
        # 将结果[0-23]=>perm=>转换成样本间的大小比较

        for index, y in self.edge_pred.iterrows():
            i, j, _d, p0, p1 = int(y['i']), int(y['j']), int(y['i<j']), y['i->j'], y['i<-j']
            direct = True
            assert _d in [0, 1]
            if _d == 1:
                direct = False
            self.__add_result2matrix(i, j, direct, (p0, p1))

    def __add_result2matrix(self, i, j, result: bool, p):
        stages = self.patient_label['stage']
        si, sj = stages[i], stages[j]
        temp_prob_a, temp_prob_b = self.prob_matrix[i][j], self.prob_matrix[j][i]
        temp_a, temp_b = self.matrix[i][j], self.matrix[j][i]
        if temp_b == 1 and result:  # 第二次出现了矛盾的预测 去掉双向边
            if temp_prob_b > p[0]:
                return
            else:
                self.matrix[i][j] = 1
                self.matrix[j][i] = 0

        if temp_a == 1 and not result:
            if temp_prob_a > p[1]:
                return
            else:
                self.matrix[i][j] = 0
                self.matrix[j][i] = 1
        self.prob_matrix[i][j] += p[0]
        self.prob_matrix[j][i] += p[1]
        self.summarize_matrix.loc[si, sj] += p[0]
        self.summarize_matrix.loc[sj, si] += p[1]
        if result:
            self.matrix[i][j] += 1
        else:
            self.matrix[j][i] += 1
        assert not (self.matrix[j][i] > 0 and self.matrix[i][j] > 0)

    def sorted_index(self):

        index_s = self.barcodes.copy()
        random.shuffle(index_s)
        # index_s.sort(key=cmp_to_key(lambda x, y: self.matrix[self.barcodes.index(x)][self.barcodes.index(y)] -
        #                                          self.matrix[self.barcodes.index(y)][self.barcodes.index(x)]))
        index_s.sort(key=cmp_to_key(lambda x, y: self.prob_matrix[self.barcodes.index(x)][self.barcodes.index(y)] -
                                                 self.prob_matrix[self.barcodes.index(y)][self.barcodes.index(x)]))
        return index_s

    def save_result(self, sorted_barcodes: list, cancer):
        self.save_vote_matrix(self.matrix, sorted_barcodes, cancer, 'vote_matrix')
        data = pd.DataFrame(self.prob_matrix, columns=self.barcodes, index=self.barcodes)
        self.save_vote_matrix(data, sorted_barcodes, cancer, 'probability_matrix')
        self.save_vote_matrix(self.summarize_matrix, None, cancer, 'stages_summarize')

    def save_vote_matrix(self, data, sorted_barcodes, cancer, title):
        if type(data) is np.ndarray:
            data = pd.DataFrame(data, columns=self.barcodes, index=self.barcodes)
        if sorted_barcodes is not None:
            data = data.filter(items=sorted_barcodes, axis=1).filter(items=sorted_barcodes)
            data = data.reindex(sorted_barcodes, axis=1).reindex(sorted_barcodes)
        data.to_excel(lpg.get_path(cancer, title, ".xlsx"))

    def draw_matrix(self, title, sorted_barcodes):
        fig = plt.figure(figsize=(16, 10))
        plt.title(title)
        data = pd.DataFrame(self.prob_matrix, columns=self.barcodes, index=self.barcodes)
        sns.heatmap(data=data.filter(items=sorted_barcodes.index, axis=0).filter(items=sorted_barcodes.index, axis=1),
                    cmap=LinearSegmentedColormap.from_list("", ["white", "red"]))
        fig.savefig(cpg.get_path(title, '.png'), bbox_inches='tight')
        plt.show()
