# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:54
# @Author       : Jue Wang and Yufan Liu
# @Description  : losses implementation

import torch
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
import numpy as np


# loss functions
# distance and loss
def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()
    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return (1 - F.cosine_similarity(p, z.detach(), dim=-1)).mean()  # scale to 0-2

    elif version == 'backprop':
        return 1 - F.cosine_similarity(p, z, dim=-1)  # with backpropagation

    else:
        raise Exception


def soft_margin_loss(anchor, pos, neg, y, margin=None):
    # innately with a distance comparison in Triplet loss
    # see https://github.com/Rostlab/EAT/, see EAT paper equation 2
    dist_cal = partial(D, version='backprop')
    pos_dist = dist_cal(anchor, pos)
    neg_dist = dist_cal(anchor, neg)

    if margin is not None:
        return nn.MarginRankingLoss(margin)(neg_dist, pos_dist, y)
    else:
        return nn.SoftMarginLoss()(neg_dist - pos_dist, y)


# implementation of triplet center loss
# see https://github.com/xlliu7/Shrec2018_TripletCenterLoss.pytorch/blob/master/misc/custom_loss.py  # [batch,dim]
class TripletCenterLoss(nn.Module):
    # note the device
    def __init__(self, margin=0, num_classes=2, num_dim=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim))  # random initialize as parameters

    def forward(self, inputs, targets):
        # resize inputs, delete labels with -1
        inputs = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
        targets = targets.reshape(targets.size(0) * targets.size(1))
        ignore_idx = targets != -1
        inputs = inputs[ignore_idx]
        targets = targets[ignore_idx]


        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))  # [batch, dim]
        centers_batch = self.centers.gather(0, targets_expand)  # 取出相应index的embedding

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)  # [batch, batch, dim]
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)  # as above
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative (the furthest positive and nearest negative)
        # hard mining
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will causes loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # inputs: [size, dim]
        batch_size = inputs.size(0)
        dist = torch.cdist(inputs, inputs)  # [batch, batch]

        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will causes loss be zero
        loss = self.loss(dist_an, dist_ap, y)

        return loss


# implementation of Focal Loss
# see https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, samples_per_class=None, beta=0.999):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta  # for class balanced
        self.samples_per_class = samples_per_class

        if not samples_per_class:
            raise NotImplementedError("samples_per_class is None.")

    def forward(self, feature, label):
        feature = feature.reshape(feature.size(0) * feature.size(1), feature.size(2))
        label = label.reshape(label.size(0) * label.size(1))
        ignore_idx = label != -1
        feature = feature[ignore_idx]
        label = label[ignore_idx]



        batch_size, num_class = feature.size(0), feature.size(1)
        label = F.one_hot(label, num_class).float()

        # alpha setting
        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * 2
        weights = torch.tensor(weights, device=feature.device).float()

        weights = weights.unsqueeze(0)
        weights = weights.repeat(batch_size, 1) * label
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, num_class)
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = weights

        # focal loss, alpha is weights above
        bc_loss = F.binary_cross_entropy(input=feature, target=label, reduction='none')
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * label * feature - self.gamma * torch.log(1 + torch.exp(-1.0 * feature)))
        loss = modulator * bc_loss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        return focal_loss / torch.sum(label)

class CrossEntropy(nn.Module):
    def forward(self, feature, label):
        feature = feature.reshape(feature.size(0) * feature.size(1), feature.size(2))
        label = label.reshape(label.size(0) * label.size(1))
        ignore_idx = label != -1
        feature = feature[ignore_idx]
        label = label[ignore_idx]
        num_class = feature.size(1)
        label = F.one_hot(label, num_class).float()

        # focal loss, alpha is weights above
        bc_loss = F.binary_cross_entropy(input=feature, target=label, reduction='mean')
        return bc_loss      

if __name__ == '__main__':
    # # debug
    torch.manual_seed(42)
    loss = FocalLoss(alpha=0.5, gamma=0)
    inputs = torch.rand(16, 2)
    targets = torch.tensor([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=torch.long)
    # print(loss(inputs, targets))

    print(loss(inputs, targets))

    loss2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1]))
    print(loss2(inputs, targets))
