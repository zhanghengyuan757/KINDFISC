import math

import torch
from torch import nn

from model_fit.Layers import GraphConvolution, BasicBlock


class GravityGAE(nn.Module):
    def __init__(self, input_dim, hid_dim, z_dim, normalize=False, epsilon=0.01):
        super(GravityGAE, self).__init__()
        self.z_dim = z_dim
        self.normalize = normalize
        self.encode_act_fn = nn.ReLU()
        self.decode_act_fn = nn.Sigmoid()
        self.gc1 = GraphConvolution(input_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, z_dim)
        self.epsilon = epsilon

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z)

    def encode(self, x, adj):
        x = self.gc1(x, adj)
        x = self.encode_act_fn(x)
        x = self.gc2(x, adj)
        return x

    def decode(self, z):
        if self.normalize:
            _z = nn.functional.normalize(z[:, 0:-1], p=2, dim=1)
        else:
            _z = z[:, 0:-1]
        x1 = torch.sum(_z ** 2, dim=1, keepdim=True)
        x2 = torch.matmul(_z, torch.t(_z))
        dist = x1 - 2 * x2 + torch.t(x1) + self.epsilon
        m = z[:, -1:]
        mass = torch.matmul(torch.ones([m.shape[0], 1]).cuda(), torch.t(m))
        out = mass - torch.log(dist)
        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(256 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        z = x.view(x.size(0), -1)
        return z


class LinkPredictModel(nn.Module):
    def __init__(self, resnet_z=256, graph_z=32):
        super(LinkPredictModel, self).__init__()
        self.act_fn = nn.ReLU(inplace=True)
        self.resnet1 = ResNet()
        self.resnet2 = ResNet()
        self.resnet_z = resnet_z
        self.graph_z = graph_z
        self.fc1 = nn.Sequential(nn.Linear(self.resnet_z * 2, self.resnet_z * 2), nn.Linear(self.resnet_z * 2, 2))
        self.fc2 = nn.Sequential(nn.Linear(self.graph_z * 2, self.graph_z * 2),
                                 nn.Linear(self.graph_z * 2, 2))
        # self.fc3 = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def get_z(self, expr1, expr2):
        z1 = self.resnet1(expr1)
        z2 = self.resnet2(expr2)
        return z1, z2

    def fc(self, z1, z2, g1, g2):
        z = torch.cat((z1, z2), 1)
        g = torch.cat((g1, g2), 1)
        expr_onehot = self.fc1(z)
        graph_onehot = self.fc2(g)
        onehot = torch.cat((expr_onehot, graph_onehot), 1)
        onehot = expr_onehot + graph_onehot
        # onehot = self.fc3(onehot)
        onehot = self.sigmoid(onehot)
        return onehot


class PairDistLoss(nn.Module):
    alpha, beta, scale = (0.5, 1.5, 1)

    def __init__(self):
        super(PairDistLoss, self).__init__()
        self.alpha = PairDistLoss.alpha
        self.beta = PairDistLoss.beta
        self.scale = PairDistLoss.scale

    def forward(self, z1, z2, s1, s2, f1, f2):
        dist = torch.pairwise_distance(z1, z2)
        f = torch.eq(f1, f2)
        a = torch.Tensor([self.beta, self.alpha])
        f = a[f.type(torch.long)]
        s = torch.abs_(torch.sub(s1, s2))
        f = f.cuda()
        s = s.cuda()
        loss = torch.mean(torch.log(1 + torch.exp(-self.scale * (dist - s * f)))) / self.scale
        return loss
