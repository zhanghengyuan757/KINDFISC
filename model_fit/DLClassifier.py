import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model_fit.BaseCL import BaseCancerCL
from model_fit.Models import PairDistLoss, LinkPredictModel
from processing.Dataset import Seq2Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
batch_size = 1000
epochs = 10
learning_rate = 0.0001
verbose = True
parallel = False
optimizer = optim.Adam
LAMBDA = 1


class BaseTorchDl(BaseCancerCL):
    def __init__(self, cancer, feat_select_method, threshold, expr, stage,
                 parallel=False):
        super(BaseTorchDl, self).__init__(cancer, feat_select_method, threshold, expr, stage)
        set_random()
        model = LinkPredictModel()
        self.train_tool = TrainTool(model, parallel, verbose)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, expr, embedding, stage, edges):
        print('CNN-Based model Training...')
        set_random()
        dataset = Seq2Dataset(expr, embedding, stage, edges)
        train_loss, train_acc = self.train_tool.train(self.optimizer, dataset,
                                                      self.batch_size, self.epochs, self.learning_rate)
        get_line_plot(self.epochs, train_loss, train_acc)
        print('CNN-Based model Trained.')

    def predict(self, expr, embedding, stage, edges):
        set_random()
        dataset = Seq2Dataset(expr, embedding, stage, edges)
        return np.array(self.train_tool.predict(dataset))

    def predict_proba(self, expr, embedding, stage, edges):
        set_random()
        dataset = Seq2Dataset(expr, embedding, stage, edges)
        return np.array(self.train_tool.predict_proba(dataset))

    def reset_model(self):
        set_random()
        self.train_tool = TrainTool(LinkPredictModel(), parallel, verbose)


def set_random():
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


class TrainTool:
    verbose = True
    path = ""
    """
    model 指定模型结构(需要实例化)
    path 模型参数缓存路径
    load 是否载入先前训练好的模型
    parallel 是否单机多卡(默认多卡，如果在多卡的时候保存了参数，那么单卡不能使用)
    """

    def __init__(self, model, parallel=True, verbose=True):
        _net = model
        _net.training = True
        _net.cuda()
        self.verbose = verbose
        self.parallel = parallel
        if not isinstance(_net, nn.DataParallel) and parallel:
            _net = nn.DataParallel(_net)
        self.net = _net

    """
    criterion 指定优化目标
    optimizer 优化器(SGD ADAM)
    train_d 训练数据集
    valid_d 验证数据集 模型的训练会根据验证数据集的loss来决定是否 保存当前模型参数
    jian_du 有监督学习和无监督学习，监督学习的train_d需要返回数据和标签，而无监督学习只需要返回数据
    batch_size 每次交给模型训练的一个批次的数据量
    epochs 训练轮数 一个数据集需要训练多轮来让模型来拟合
    learning_rate 学习率 不应该设置的过大
    """

    def train(self, optimizer, train_d,
              batch_size, epochs,
              learning_rate):
        train_loss, train_acc = [], []
        batch_train_loss, batch_train_acc, batch_test_acc = [], [], []
        net = self.net
        optimizer = optimizer(net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train_d, batch_size, shuffle=True)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = PairDistLoss()
        for epoch in range(epochs):
            net.train()
            correct = 0
            for i, data in enumerate(train_loader):
                expr1, expr2, graph_z_1, graph_z_2, s1, s2, f1, f2, labels = data
                expr1 = Variable(expr1.cuda())
                expr2 = Variable(expr2.cuda())
                graph_z_1 = Variable(graph_z_1.cuda())
                graph_z_2 = Variable(graph_z_2.cuda())
                labels = Variable(labels.cuda())
                labels = labels.squeeze()  # must be Tensor of dim BATCH_SIZE, MUST BE 1Dimensional!
                """
                当网络参量进行反馈时，梯度是累积计算而不是被替换，
                但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
                因此需要对每个batch调用一遍zero_grad（）将参数梯度置0
                """
                optimizer.zero_grad()
                if self.parallel:
                    z1, z2 = net.module.get_z(expr1, expr2)
                    onehot = net.module.fc(z1, z2, graph_z_1, graph_z_2)
                else:
                    z1, z2 = net.get_z(expr1, expr2)
                    onehot = net.fc(z1, z2, graph_z_1, graph_z_2)

                loss1 = criterion1(onehot, labels)
                loss2 = criterion2(z1, z2, s1, s2, f1, f2)
                loss = loss1 + LAMBDA * loss2
                loss.backward()
                optimizer.step()

                if (i + 1) % (len(train_d) // batch_size // 10) == 0:
                    if self.verbose:
                        print('\rEpoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, epochs,
                                                                              i + 1, len(train_d) // batch_size,
                                                                              loss.item()), end='', flush=True)

                if self.parallel:
                    z1, z2 = net.module.get_z(expr1, expr2)
                    y_hat = net.module.fc(z1, z2, graph_z_1, graph_z_2)
                else:
                    z1, z2 = net.get_z(expr1, expr2)
                    y_hat = net.fc(z1, z2, graph_z_1, graph_z_2)
                pred = y_hat.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
                train_loss.append(loss.item())
                train_acc.append(100. * correct / len(train_d))
            if self.verbose:
                batch_train_loss.append(loss.item())
                batch_train_acc.append(train_acc[-1])
                print('\rEpoch: {}, Training Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
                    epoch + 1, loss.item(), correct, len(train_d), train_acc[-1]), end='', flush=True)
        if self.verbose:
            print('')

        return batch_train_loss, batch_train_acc

    def predict(self, dataset):
        self.net = self.net.eval()
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        r = []

        for _, data in enumerate(data_loader):
            expr1, expr2, graph_z_1, graph_z_2 = data
            expr1 = Variable(expr1.cuda())
            expr2 = Variable(expr2.cuda())
            graph_z_1 = Variable(graph_z_1.cuda())
            graph_z_2 = Variable(graph_z_2.cuda())
            net = self.net
            if self.parallel:
                z1, z2 = net.module.get_z(expr1, expr2)
                onehot = net.module.fc(z1, z2, graph_z_1, graph_z_2)
            else:
                z1, z2 = net.get_z(expr1, expr2)
                onehot = net.fc(z1, z2, graph_z_1, graph_z_2)
            _, predicted = torch.max(onehot.data, 1)
            r.append(predicted.cpu().tolist())
        return sum(r, [])

    def predict_proba(self, dataset):
        self.net = self.net.eval()
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        r = []
        for _, data in enumerate(data_loader):
            expr1, expr2, embedding1, embedding2 = data
            expr1 = Variable(expr1.cuda())
            expr2 = Variable(expr2.cuda())
            graph_z_1 = Variable(embedding1.cuda())
            graph_z_2 = Variable(embedding2.cuda())
            net = self.net
            if self.parallel:
                z1, z2 = net.module.get_z(expr1, expr2)
                onehot = net.module.fc(z1, z2, graph_z_1, graph_z_2)
            else:
                z1, z2 = net.get_z(expr1, expr2)
                onehot = net.fc(z1, z2, graph_z_1, graph_z_2)
            r.append(onehot.cpu().tolist())
        return sum(r, [])


def get_line_plot(epochs_n, train_loss, train_acc, jian_du=True):
    # iter_n = len(train_loss) / epochs_n
    # iter_n = epochs_n
    fig, ax1 = plt.subplots(figsize=(8, 4))
    if jian_du:
        ax2 = ax1.twinx()  # 共享x轴
    # x = [i / iter_n for i in range(len(train_acc))]
    x = range(1, len(train_loss) + 1)
    # ax1.plot(x, train_loss, 'r', label=u'Training loss')
    ax1.plot(x, train_loss, 'r', label=u'Training loss')
    ax1.legend()
    if jian_du:
        # ax2.plot(x, train_acc, 'g', label=u'Training accuracy')
        # ax2.plot(x, [train_acc[epochs_n * (i + 1) - 1] for i in range(epochs_n)], 'g',
        #          label=u'Training accuracy')
        ax2.plot(x, train_acc, 'g', label=u'Training accuracy')
        ax2.tick_params(axis='y')
        ax2.legend()
        ax2.set_ylabel(u'accuracy')
    plt.xlabel(u'epoch')
    ax1.set_ylabel(u'loss')
    plt.show()
