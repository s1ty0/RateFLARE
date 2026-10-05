from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
import pandas as pd
import torch
from typing import Callable
from sklearn.preprocessing import StandardScaler

class MyDataset1_noscaler(Dataset):
    def __init__(self, npy_dir_path,meta_feats_path,
                 kids_path, p2n_rate_train,p2n_rate_test,
                 data_split="train",if_only_metavec=False):
        super(MyDataset1_noscaler,self).__init__()
        self.npy_dir_path=npy_dir_path
        self.kids=np.load(kids_path)
        self.if_only_metavec=if_only_metavec
        if data_split=="train":
            self.ifTrain=True
        else:
            self.ifTrain=False
        self.load_meta_idxs_feats(meta_feats_path)
        self.p2n_rate_train=p2n_rate_train
        self.p2n_rate_test=p2n_rate_test
        self.data_split=data_split
        self._generate_train_()

    def load_meta_idxs_feats(self,meta_feats_path):
        data=np.load(meta_feats_path) 
        meta_kids=data[:,0].flatten()
        meta_feats=data[:,1:].astype(np.float_)
        mask=np.isin(self.kids,meta_kids)
        if self.if_only_metavec:
            mean_=np.nanmean(meta_feats,axis=0)
            std_=np.nanstd(meta_feats,axis=0)
            meta_feats = (meta_feats-mean_)/std_
            meta_feats[np.isnan(meta_feats)]=0
        mask=np.isin(meta_kids,self.kids)
        self.meta_feats=meta_feats[mask]

    def _generate_train_(self):
        if self.ifTrain:
            # np.memmap(file_path, dtype="float32", mode="r")
            all_datas=np.load(os.path.join(self.npy_dir_path,"all_datas_Tr.npy"),mmap_mode="r+")
            # all_datas=np.load(os.path.join(self.npy_dir_path,"all_datas_Tr.npy"))
            all_labels=np.load(os.path.join(self.npy_dir_path,"all_labels_Tr.npy"))
            all_kids=np.load(os.path.join(self.npy_dir_path,"all_kids_Tr.npy"))
            p2n_rate_data=self.p2n_rate_train
        else:
            all_datas=np.load(os.path.join(self.npy_dir_path,"all_datas_t.npy"),mmap_mode="r+")
            # all_datas=np.load(os.path.join(self.npy_dir_path,"all_datas_t.npy"))
            all_labels=np.load(os.path.join(self.npy_dir_path,"all_labels_t.npy"))
            all_kids=np.load(os.path.join(self.npy_dir_path,"all_kids_t.npy"))
            p2n_rate_data=self.p2n_rate_test
        self.num_wo_add=len(all_kids)

        if p2n_rate_data>0:
            mask=all_labels==1
            num_1=np.sum(mask)
            num_keep_0=int(num_1/p2n_rate_data)
            idxs_0=np.argwhere(~mask).flatten()
            np.random.seed(123)
            idxs_0=np.random.permutation(idxs_0)[:num_keep_0]
            mask[idxs_0]=True
            self.use_iiis=np.argwhere(mask).flatten()
        else:
            self.use_iiis=np.arange(len(all_labels))

        self.all_datas=all_datas
        self.all_labels=all_labels
        self.all_kids=all_kids

    def __len__(self):
        return len(self.use_iiis)

    def __getitem__(self, index):
        index=self.use_iiis[index]
        label=self.all_labels[index]
        lc_time,lc_flux,lc_label=self.all_datas[index]
        lc_id=self.all_kids[index]

        lc_mask=np.isnan(lc_flux)
        lc_flux[lc_mask]=1
        lc_mask=lc_mask.astype(np.float_)

        meta_feats=self.meta_feats[lc_id]
        kid_name=self.kids[lc_id]

        idx_noNan=np.argwhere(~np.isnan(lc_time)).flatten()[0]
        value_noNan=lc_time[idx_noNan]
        idxs_Nan=np.argwhere(np.isnan(lc_time)).flatten()
        values_Nan=np.ones_like(idxs_Nan)*value_noNan+(idxs_Nan-idx_noNan)*(1/48)
        lc_time[idxs_Nan]=values_Nan

        return lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id



# Adapted from: https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
# Copyright (c) 2018 Ming
# License: MIT
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.Dataset):
            # return dataset.use_label[dataset.use_iiis]
            return dataset.all_labels[dataset.use_iiis]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
