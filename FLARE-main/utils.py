import os
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,precision_recall_curve,auc
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix,roc_curve,auc
from torch.utils.data import Dataset
import time
from typing import Optional, Union
from argparse import Namespace
import os
import pickle
from scipy.io import loadmat
import numpy as np
from collections import Counter,defaultdict
import torch
import random
from collections import namedtuple
import copy
import shutil
import json
import gc
import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix,roc_curve,auc
from torch.utils.data import Dataset
import time
from typing import Optional, Union
from argparse import Namespace

class EarlyStopper:
    def __init__(self, dir_path, patience=30):
        self.save_path = os.path.join(dir_path,f"early_stop_checkpoint.pth")
        self.patience = patience
        self.counter = 0
        self.best_ep = -1
        self.best_score = np.inf
        self.early_stop = False
        self.thres=None

    def step(self, score, epoch, model,thres):
        if self.best_score is None:
            self.best_score = score
            self.best_ep = epoch
            self.thres=thres
            self.save_checkpoint(model)
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} in epoch {epoch}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_ep = epoch
            self.thres=thres
            self.save_checkpoint(model)
            self.counter = 0
            self.early_stop = False
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.save_path))


class Summary:
    def __init__(self,dir_path,task_name="train",flush_secs=60):
        self.task_name = task_name
        self.log_dir = os.path.join(dir_path,"logging",self.task_name)
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)
        self.metric_step = 0
        self.loss_step=0

    def update_loss(self,loss):
        self.writer.add_scalar(tag="loss", scalar_value=loss, global_step=self.loss_step)
        self.loss_step += 1
    def update_metrics(self,metrics,names):
        for name,metric in zip(names,metrics):
            self.writer.add_scalar(tag=f"metrics/{name}", scalar_value=metric,
                                   global_step=self.metric_step)
        self.metric_step += 1

    def close(self):
        self.writer.close()



# Adapted from: https://github.com/moment-timeseries-foundation-model/moment-research
# Copyright (c) 2024 Auton Lab, Carnegie Mellon University
# License: MIT
def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric

def mae(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    delta_y = np.abs(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    delta_y = np.square(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


def rmse(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    return np.sqrt(mse(y, y_hat, reduction, axis))

def _divide_no_nan(a: float, b: float) -> float:
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div

def mape(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    error = _divide_no_nan(delta_y, scale)
    return 100 * _reduce(error, reduction=reduction, axis=axis)

def smape(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    error = _divide_no_nan(delta_y, scale)
    error = 200 * _reduce(error, reduction=reduction, axis=axis)
    if isinstance(error, float):
        assert error <= 200, "SMAPE should be lower than 200"
    else:
        assert all(error <= 200), "SMAPE should be lower than 200"
    return error

def get_classification_metrics(y, y_hat, reduction='mean',thres=0.5):
    probs=np.reshape(y_hat,(-1,))
    labels=np.reshape(y,(-1,))
    preds=probs>thres
    tn, fp, fn, tp =confusion_matrix(labels,preds).ravel()

    # precision @ k 
    argsort_probs=np.argsort(probs).flatten()
    tpp=np.logical_and(preds,labels.astype(np.bool_))
    precision_at_50=np.sum(tpp[argsort_probs[-50:]])/50
    precision_at_100=np.sum(tpp[argsort_probs[-100:]])/100
    precision_at_500=np.sum(tpp[argsort_probs[-500:]])/500
    
    
    recall_1 = recall_score(labels, preds, pos_label=1)
    precision_1= precision_score(labels, preds, pos_label=1)
    f1_score_1 = f1_score(labels, preds, pos_label=1)
    f1_score_macro = f1_score(labels, preds, average="macro")
    f1_score_micro = f1_score(labels, preds, average="micro")

    
    ap = average_precision_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)
    precisions,recalls,threses=precision_recall_curve(labels,probs)
    auprc=auc(recalls,precisions)
    TSS=(tp/(tp+fn))-(fp/(fp+tn))
    HSS2=2*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tn+fp)*(tp+fp))
    # IOU=(np.sum(np.logical_and(labels,preds)))/(np.sum(np.logical_or(labels,preds)))
    # dice=2*np.sum(np.logical_and(labels,preds))/(np.sum(labels)+np.sum(preds))

    metrics=Namespace()
    metrics.recall_1=recall_1
    metrics.precision_1=precision_1
    metrics.f1_score_1=f1_score_1
    metrics.ap=ap
    metrics.auc=auroc
    metrics.TSS=TSS
    metrics.HSS2=HSS2
    # metrics.IOU=IOU
    # metrics.dice=dice
    metrics.tn=tn
    metrics.fp=fp
    metrics.fn=fn
    metrics.tp=tp
    metrics.thres=thres
    metrics.acc=(tp+tn)/(tp+tn+fp+fn)
    metrics.auprc=auprc
    metrics.f1_score_macro=f1_score_macro
    metrics.f1_score_micro=f1_score_micro
    metrics.precision_at_50=precision_at_50
    metrics.precision_at_100=precision_at_100
    metrics.precision_at_500=precision_at_500
    
    return metrics
    
def get_classification_metrics2(y,probs_1):
    # probs_1=y_probs[:,1]
    labels=y
    precision,recall,thresholds = precision_recall_curve(labels,probs_1)
    f1= 2 * (precision * recall) / (precision + recall)
    thres=thresholds[np.nanargmax(f1).flatten()[0]]
    return thres


def get_classification_metrics3(y, y_hat, thres=0.5):
    B,L=y.shape
    yrow=np.sum(y,axis=1)    
    yhatrow=np.sum(y_hat,axis=1)    
    tp=np.sum(np.logical_and(yrow>0,yhatrow>0))
    fp=np.sum(np.logical_and(yhatrow>0,yrow==0))
    fn=np.sum(np.logical_and(yhatrow==0,yrow>0))    
    tn=np.sum(np.logical_and(yhatrow==0,yrow==0))    
    
    recall_1=tp/(tp+fn)
    precision_1=tp/(tp+fp)
    f1_score_1=2*recall_1*precision_1/(recall_1+precision_1)
    exist=2*tp/(2*tp+fp+fn)
    density=1-np.mean(np.abs(np.sum(y_hat-y,axis=1)))
    
    TSS=(tp/(tp+fn))-(fp/(fp+tn))
    HSS2=2*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tn+fp)*(tp+fp))
    # IOU=(np.sum(np.logical_and(labels,preds)))/(np.sum(np.logical_or(labels,preds)))
    # dice=2*np.sum(np.logical_and(labels,preds))/(np.sum(labels)+np.sum(preds))

    metrics=Namespace()
    metrics.recall_1=recall_1
    metrics.precision_1=precision_1
    metrics.f1_score_1=f1_score_1
    metrics.TSS=TSS
    metrics.HSS2=HSS2
    metrics.tn=tn
    metrics.fp=fp
    metrics.fn=fn
    metrics.tp=tp
    metrics.exist=exist
    metrics.density=density
    return metrics
    