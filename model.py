# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:54
# @Author       : Jue Wang and Yufan Liu
# @Description  : Dataset and model

import torch
import torch.nn as nn

# models
class Encoder(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=2048, out_dim=1024):
        super(Encoder, self).__init__()
        # Layers from Patric Hua
        # forward size: 1024 -> hidden -> 1024
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ClassifierModel(nn.Module):
    def __init__(self, in_dim=1024):
        super(ClassifierModel, self).__init__()
        # self.norm0 = nn.BatchNorm1d(1024)
        self.mlp1 = nn.Linear(in_dim, 512)

        self.mlp2 = nn.Linear(512, 256)

        self.mlp3 = nn.Linear(256, 128)

        self.mlp4 = nn.Linear(128, 2)
        self.act = nn.ReLU()
        self.head = nn.Softmax(-1)

    def forward(self, x):
        # x = self.norm0(x)
        x = self.act(self.mlp1(x))
        x = self.act(self.mlp2(x))
        x = self.act(self.mlp3(x))
        return self.head(self.mlp4(x))


class StageMLP(nn.Module):
    def __init__(self):
        super(StageMLP, self).__init__()
        self.encoder = Encoder()
        self.classifier = ClassifierModel()

    def forward(self, x):
        embedding = self.encoder(x)
        score = self.classifer(embedding)
        return score, embedding


class ContinueModel(nn.Module):
    # A full model
    def __init__(self):
        super(ContinueModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU()
        )


        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x  # dim 1024

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.layer5(x)), inter

class SimpleModel(nn.Module):
    # ESM only
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(1280, 2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x  # dim 1024

        return self.head(x), inter

class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1280, 1280, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1280)

        self.conv2 = nn.Conv1d(1280, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)

        self.head = nn.Softmax(-1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, dim, length
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        inter = x.permute(0, 2, 1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)


        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return self.head(x), inter


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layer1 = nn.GRU(1024, 1024, 1, batch_first=True)
        self.layer2 = nn.GRU(1024, 128, 1, batch_first=True)
        self.layer3 = nn.GRU(128, 64, 1, batch_first=True)

        self.project = nn.Linear(64, 2)

        self.head = nn.Softmax(-1)

    def forward(self, x):
        device = x.device
        bz = x.size(0)
        h0 = torch.zeros(1, bz, 1024).to(device)
        c0 = torch.zeros(1, bz, 1024).to(device)
        out, _ = self.layer1(x, (h0, c0))
        inter = out

        h1 = torch.zeros(1, bz, 128).to(device)
        c1 = torch.zeros(1, bz, 128).to(device)
        out, _ = self.layer2(out, (h1, c1))

        h2 = torch.zeros(1, bz, 64).to(device)
        c2 = torch.zeros(1, bz, 64).to(device)
        out, _ = self.layer3(out, (h2, c2))

        x = self.project(out)
        return self.head(x), inter


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.begin = nn.Linear(1280, 1024)

        self.attention = nn.MultiheadAttention(1024, 8, 0.3, batch_first=True)

        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.3)

        self.ffn = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1024)
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.begin(x)
        x, attention = self.attention(x, x, x)  # [b, length, length]
        x = self.dropout(self.norm(x + x))
        inter = x
        x = self.dropout(self.norm(self.ffn(x) + x))  # [length, 1024]
        return self.class_head(x), inter
