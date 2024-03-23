# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:54
# @Author       : Jue Wang and Yufan Liu
# @Description  : Data loading

import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

def label2tensor(labels):
    labels = list(map(lambda x: int(x), list(labels)))
    return torch.tensor(labels).long()

# Dataloader
class LigandData(Dataset):
    def __init__(self, data_dict_root):
        data_dict = pickle.load(open(data_dict_root, 'rb'))
        self.features = [t[0] for t in data_dict.values()]
        self.labels = [t[1] for t in data_dict.values()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BatchCollate(object):
    def __call__(self, data):
        # ([bz, length, dim], label)
        # (feature, label)
        tensors = pad_sequence([t[0] for t in data], batch_first=True)  # ([bz, length, dim])   
        labels = pad_sequence([label2tensor(t[1]) for t in data], batch_first=True, padding_value=-1)  # [bz, length] 
        return tensors, labels


class ProteinLigandData(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_root, val_data_root, workers=1, pin_memory=False):
        super(ProteinLigandData, self).__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

        self.collate_fn = BatchCollate()
        self.train_data = LigandData(train_data_root)
        self.val_data = LigandData(val_data_root)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)  # shuffle here.

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=1,  # does not matter, no-interaction between sequences, not like attention
                          collate_fn=self.collate_fn,
                          num_workers=self.workers,
                          pin_memory=self.pin_memory)
    