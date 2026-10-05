# utils/early_stopping.py 或直接放在 train.py 顶部
import torch
import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf if mode == 'max' else np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, val_score, model):
        score = val_score

        if self.mode == 'max':
            if score > self.val_score_max + self.delta:
                self.save_checkpoint(val_score, model)
                self.val_score_max = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:  # min mode
            if score < self.val_score_max - self.delta:
                self.save_checkpoint(val_score, model)
                self.val_score_max = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation score improves.'''
        if self.verbose:
            print(f'Validation score improved ({self.val_score_max:.6f} --> {val_score:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)