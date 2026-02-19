import argparse
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DataLoader

sys.path.append('..')

from dataloader import MolecularDataset
from networks import RNNDecoder, RNNEncoder

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class RNN_VAE(pl.LightningModule):
    def __init__(self, train_data):
        super().__init__()
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
        if means_p is None:
            return -0.5 * torch.mean(torch.sum(1 + logvar_q - means_q.pow(2) - logvar_q.exp(), dim=-1))
        return -0.5 * torch.mean(
            torch.sum(1 - logvar_p + logvar_q - (means_q.pow(2) + logvar_q.exp()) * (-logvar_p).exp(), dim=-1)
        )

    def training_step(self, batch, batch_nb):
        x, _ = batch
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)
        kl = self.kl_div(p_z_means, p_z_logvar)
        x_by_z_logprob = self.dec.get_log_prob(x=x, z=z_sample).mean()
        loss = -x_by_z_logprob + self.beta * kl
        self.log('train_loss', loss, prog_bar=False)
        self.log('kl', kl, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_nb):
        x, _ = batch
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)
        sampled_sm = self.dec.sample(z_sample)

        valid_proc = len([s for s in sampled_sm if Chem.MolFromSmiles(s) is not None]) / len(sampled_sm)
        unique_proc = len(np.unique(sampled_sm)) / len(sampled_sm)
        eq_proc = len(
            [
                s
                for (x_ob, s) in zip(x, sampled_sm)
                if (Chem.MolFromSmiles(s) is not None) and (Chem.MolToSmiles(Chem.MolFromSmiles(s))) == x_ob
            ]
        ) / len(sampled_sm)
        return {'valid': valid_proc, 'unique': unique_proc, 'equal': eq_proc}

    def validation_epoch_end(self, outputs):
        self.log('val_valid', np.array([x['valid'] for x in outputs]).mean(), prog_bar=False)
        self.log('val_unique', np.array([x['unique'] for x in outputs]).mean(), prog_bar=False)
        self.log('val_equal', np.array([x['equal'] for x in outputs]).mean(), prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def train_dataloader(self):
        return DataLoader(MolecularDataset(self.train_data, train=True), batch_size=512, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(MolecularDataset(self.train_data, train=False), batch_size=512, shuffle=True, num_workers=10)


if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Script to pretrain RNN encoder and decoder')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--train_data', type=str, default='../data/train.csv')
    args = parser.parse_args()

    model = RNN_VAE(train_data=args.train_data)
    logger = TensorBoardLogger(save_dir='../logs', name='rnn_vae')

    if args.gpu < 0:
        tr = pl.Trainer(accelerator='cpu', devices=1, logger=logger, enable_progress_bar=False, max_epochs=200)
    else:
        tr = pl.Trainer(accelerator='gpu', devices=[args.gpu], logger=logger, enable_progress_bar=False, max_epochs=200)

    tr.fit(model)
    torch.save(model.enc.state_dict(), '../saved_models/rnn_enc.ckpt')
    torch.save(model.dec.state_dict(), '../saved_models/rnn_dec.ckpt')
