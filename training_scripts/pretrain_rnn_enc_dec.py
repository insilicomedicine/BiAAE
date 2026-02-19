import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger

import sys
sys.path.append('..')

import argparse

from networks import RNNDecoder, RNNEncoder
from torch.utils.data import DataLoader
from dataloader import MolecularDataset

from rdkit import Chem
import numpy as np

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class RNN_VAE(pl.LightningModule):
    def __init__(self, train_data):
        super(RNN_VAE, self).__init__()
        self.z_dim = 44
        self.train_data = train_data

        self.enc = RNNEncoder(out_dim=2 * self.z_dim)
        self.dec = RNNDecoder(in_dim=self.z_dim)
        
        self.beta = 0.01
        
    @staticmethod
    def sample_repar_z(means, logvar):
        return means + torch.randn_like(means) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_div(means_q, logvar_q, means_p=None, logvar_p=None):
        if means_p is None:  # prior is N(0, I)
            return -0.5 * torch.mean(
                torch.sum(1 + logvar_q - means_q.pow(2) - logvar_q.exp(),
                          dim=-1))
        else:
            return -0.5 * torch.mean(
                torch.sum(1 - logvar_p + logvar_q -
                          (means_q.pow(2) + logvar_q.exp()) * (-logvar_p).exp(),
                          dim=-1))

    def training_step(self, batch, batch_nb):
        # pair of objects
        x, _ = batch

        # compute proposal distributions
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)

        # sample z
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)

        # compute kl divergence
        kl = self.kl_div(p_z_means, p_z_logvar)

        # compute reconstrunction loss
        x_by_z_logprob = self.dec.get_log_prob(x=x, z=z_sample).mean()

        loss = (-x_by_z_logprob + self.beta * kl)

        return {'loss': loss,
                'log': {
                    'x_by_z_logprob': x_by_z_logprob,
                    'kl': kl}
                }
    
    def validation_step(self, batch, batch_nb):
        x, _ = batch
        
        # compute proposal distributions
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)

        # sample z
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)
        
        sampled_sm = self.dec.sample(z_sample)
        
        valid_proc = len([s for s in sampled_sm if Chem.MolFromSmiles(s) is not None]) / len(sampled_sm)
        unique_proc = len(np.unique(sampled_sm)) / len(sampled_sm)
        eq_proc = len([s  for (x_ob, s) in zip(x, sampled_sm) if (Chem.MolFromSmiles(s) is not None) and (Chem.MolToSmiles(Chem.MolFromSmiles(s))) == x_ob]) / len(sampled_sm)
        
        return {'valid': valid_proc, 'unique': unique_proc, 'equal': eq_proc}
        
    def validation_end(self, outputs):
        val_stats = {}

        val_stats['val_valid'] = np.array([x['valid'] for x in outputs]).mean()
        val_stats['val_unique'] = np.array([x['unique'] for x in outputs]).mean()
        val_stats['val_equal'] = np.array([x['equal'] for x in outputs]).mean()

        return {'log': val_stats}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MolecularDataset(self.train_data, train=True),
                          batch_size=512, shuffle=True, num_workers=10)
   
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MolecularDataset(self.train_data, train=False),
                          batch_size=512, shuffle=True, num_workers=10)

if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description=
                                     'Script to pretrain RNN encoder and decoder')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--train_data', type=str, default='../data/train.csv')

    args = parser.parse_args()

    model = RNN_VAE(train_data=args.train_data)

    logger = TestTubeLogger(save_dir='../logs', name='rnn_vae')

    tr = pl.Trainer(gpus=([] if (args.gpu < 0) else [args.gpu]),
                    logger=logger,
                    early_stop_callback=False,
                    max_nb_epochs=200, show_progress_bar=False)
    tr.fit(model)
    torch.save(model.enc.state_dict(), '../saved_models/rnn_enc.ckpt')
    torch.save(model.dec.state_dict(), '../saved_models/rnn_dec.ckpt')
