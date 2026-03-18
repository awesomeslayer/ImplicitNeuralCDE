# FILE: src_torch/data.py
import numpy as np
import torch
from aeon.datasets import load_classification
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class DataModule(LightningDataModule):
    def __init__(self, dsname, batch_size):
        super().__init__()
        self.dsname = dsname
        self.batch_size = batch_size
        X, Y = load_classification(dsname, extract_path="/app/data/")
        Y = Y.astype(int) - 1
        X = X.transpose(0, 2, 1).astype(np.float32)
        self.inp_dim = int(X.shape[-1])
        self.out_dim = int(Y.max() + 1)

        Xtrain, Xvaltest, Ytrain, Yvaltest = train_test_split(X, Y, test_size=0.2, random_state=42)
        Xval, Xtest, Yval, Ytest = train_test_split(Xvaltest, Yvaltest, test_size=0.5, random_state=42)

        self.train_ds = TensorDataset(torch.from_numpy(Xtrain), torch.from_numpy(Ytrain))
        self.val_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(Yval))
        self.test_ds = TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4, shuffle=False)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4, shuffle=False)