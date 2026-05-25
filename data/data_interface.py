import lightning as L
import inspect
import importlib
import pickle as pkl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .dataset_neims import MSPDataset

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Lightning (works for both 'lightning' and 'pytorch_lightning' packages)
try:
    from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
except Exception:
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# matchms for MSP IO
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp
from matchms import Spectrum


class DInterface(L.LightningDataModule):
    def __init__(self, train_msp: Optional[str], val_msp: Optional[str], test_msp: Optional[str],
                 n_bits=4096, radius=2, max_mz=2000, batch_size=64, num_workers=0):
        super().__init__()
        self.train_msp = train_msp
        self.val_msp = val_msp
        self.test_msp = test_msp
        self.n_bits = n_bits
        self.radius = radius
        self.max_mz = max_mz
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.train_msp:
            self.train_ds = MSPDataset(self.train_msp,f"{self.train_msp}_cache.npz", self.n_bits, self.radius, self.max_mz)
        else:
            self.train_ds = None
        if self.val_msp:
            self.val_ds = MSPDataset(self.val_msp,f"{self.val_msp}_cache.npz", self.n_bits, self.radius, self.max_mz)
        else:
            self.val_ds = None
        if self.test_msp:
            self.test_ds = MSPDataset(self.test_msp,f"{self.test_msp}_cache.npz", self.n_bits, self.radius, self.max_mz)
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False)
