import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class MolecularDataset(Dataset):
    def __init__(self, filename, train=True, seed=777):
        self.smiles = pd.read_csv(filename, header=None).values[:, 0]
        
        np.random.seed(seed)
        
        train_test_perm = np.random.permutation(self.smiles.shape[0])
        
        if train:
            self.smiles = self.smiles[train_test_perm[:200000]]
        else:
            self.smiles = self.smiles[train_test_perm[200000:]]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], torch.ones(1)