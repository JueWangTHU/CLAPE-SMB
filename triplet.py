# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:49
# @Author       : Jue Wang and Yufan Liu
# @Description  : Classification with class-balanced focal loss and TCL

## train 

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import count
from data import ProteinLigandData
from model import StageMLP, ContinueModel, CNNOD, RNN, TransformerModel, SimpleModel
from losses import TripletCenterLoss, FocalLoss, CrossEntropy
import numpy as np
import pytorch_lightning as pl
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

pl.seed_everything(42)


class TripletClassificationModel(pl.LightningModule):
    def __init__(self,
                alpha,
                margin,
                clw,  # clw: contrastive learning weight
                clf_lr,
                loss_lr,
                gamma,
                loss,
                samples_per_class,
                batch_size,
                backbone
                ):
        # print("clw is short for Contrastive Learning Weight")
        super(TripletClassificationModel, self).__init__()
        self.save_hyperparameters()
        assert backbone in ['stage', 'full', 'cnn', 'rnn', 'attention', 'simple'], 'Not support.'
        self.backbone = backbone
        self.triplet_criterion = TripletCenterLoss(margin=margin)

        if loss == 'focal':
            self.clf_criterion = FocalLoss(alpha=alpha, gamma=gamma, samples_per_class=samples_per_class)
        elif loss == 'ce':
            self.clf_criterion = CrossEntropy()
        else:
            raise Exception

        self.clw = clw
        self.clf_lr = clf_lr
        self.loss_lr = loss_lr
        self.automatic_optimization = False  # pause auto optimizer

        # model definitions
        if backbone == 'stage':
            self.full_model = StageMLP()
        elif backbone == 'full':
            self.full_model = ContinueModel()
        elif backbone == 'cnn':
            self.full_model = CNNOD()
        elif backbone == 'rnn':
            self.full_model = RNN()
        elif backbone == 'attention':
            self.full_model = TransformerModel()
        elif backbone == 'simple':
            self.full_model = SimpleModel()

        # scheduler
        # self.warmup_ratio = warmup_ratio
        # self.epochs = epochs
        # self.n_batch = n_batch

    def training_step(self, batch, batch_idx):
        model_opt, loss_opt = self.optimizers()

        model_opt.zero_grad()
        loss_opt.zero_grad()

        feature, label = batch
        score, embedding = self.full_model(feature)

        clf_loss = self.clf_criterion(score, label)
        triplet_loss = self.triplet_criterion(score, label)
        self.log('classification loss', clf_loss)
        self.log('triplet loss', triplet_loss)

   
        loss = clf_loss + self.clw * triplet_loss
        self.log('loss', loss)
        self.manual_backward(loss)

        model_opt.step()
        if self.clw != 0:
            #self.clip_gradients(loss_opt, gradient_clip_val=0.5)
            loss_opt.step()
        # model_scheduler.step()
        # loss_scheduler.step()

        # self.log('model lr', model_scheduler.get_lr()[0])
        # self.log('loss lr', loss_scheduler.get_lr()[0])

        return {'embedding': embedding.reshape(embedding.size(0) * embedding.size(1), -1), 'label': label.reshape(label.size(0) * label.size(1)), 'score': score, 'loss': loss}

    def training_epoch_end(self, outputs):
        # training embedding is saved as batch, which is not the same
        embedding_list = [out['embedding'].detach().cpu().numpy() for out in outputs]
        label_list = [out['label'].cpu().numpy() for out in outputs]
        score_list = [out['score'].detach().cpu().numpy() for out in outputs]

        self.train_embedding = embedding_list
        self.train_label = label_list
        self.train_score = score_list

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        #embedding = self.encoder(feature)
        #score = self.classifier(embedding)
        score, embedding = self.full_model(feature)
        print(score.shape)
        return {'embedding': embedding.squeeze(0), 'label': label.squeeze(0), 'score': score.squeeze(0)}

    def validation_epoch_end(self, outputs):
        # stack encoded features and labels
        embedding_list = [out['embedding'].detach().cpu().numpy() for out in outputs]
        label_list = [out['label'].cpu().numpy() for out in outputs]
        score_list = [out['score'].detach().cpu().numpy() for out in outputs]

        self.val_embedding = embedding_list
        self.val_label = label_list
        self.val_score = score_list

        # metrics
        score = np.concatenate(score_list)
        label = np.concatenate(label_list)
        auc = roc_auc_score(label, score[:, 1])
        mcc = matthews_corrcoef(label, score.argmax(1))
        self.log("AUC", auc)
        self.log("MCC", mcc)

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(self.full_model.parameters(), lr=self.clf_lr)
        loss_optimizer = torch.optim.Adam(self.triplet_criterion.parameters(), lr=self.loss_lr)
        return [model_optimizer, loss_optimizer]
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_embedding'] = self.train_embedding
        checkpoint['train_label'] = self.train_label
        checkpoint['train_score'] = self.train_score
        checkpoint['val_embedding'] = self.val_embedding
        checkpoint['val_label'] = self.val_label
        checkpoint['val_score'] = self.val_score

    def on_load_checkpoint(self, checkpoint):
        self.train_embedding = checkpoint['train_embedding']
        self.train_label = checkpoint['train_label']
        self.train_score = checkpoint['train_score']
        self.val_embedding = checkpoint['val_embedding']
        self.val_label = checkpoint['val_label']
        self.val_score = checkpoint['val_score']


if __name__ == '__main__':
    ligand = 'SM' + '/'
    batch_size = 8
    
    data_params = {'batch_size': batch_size, 
                    'train_data_root': './Dataset/SM/esm_train_UniProtSMB.pkl',
                    'val_data_root': './Dataset/SM/esm_valid_UniProtSMB.pkl'}
    samples_per_class = count.count('./Raw_data/train_UniProtSMB.txt')

    data = ProteinLigandData(**data_params)
    
    epochs = 20
    gpus = [0]
    model_params = {'alpha': None,
                    'margin': 5,
                    'clw': 0.2,
                    'clf_lr': 1e-4,
                    'loss_lr': 0.01,
                    'gamma': 5,
                    'loss': 'focal',
                    'samples_per_class': samples_per_class,
                    'batch_size' : batch_size,
                    'backbone': 'simple'
                    }
    model_backbone = model_params['backbone'] + '/'
    training_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
   
    checkpoint = ModelCheckpoint(dirpath='triplet_classification/' + ligand + model_backbone + training_time,
                                 save_top_k=1, monitor='MCC', mode='max')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='triplet_classification/' + ligand + model_backbone + training_time, name='pl_logs',
                                             version='', default_hp_metric=False)
    trainer = pl.Trainer(logger=tb_logger, callbacks=checkpoint, max_epochs=epochs,
                         gpus=gpus, log_every_n_steps=1)
    
    model = TripletClassificationModel(**model_params)
    # model = TripletClassificationModel.load_from_checkpoint('./triplet_classification/SM/cnn/09-18-10-42-48/epoch=12-step=26000.ckpt')

    trainer.fit(model, datamodule=data)







