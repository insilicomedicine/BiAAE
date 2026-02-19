# CVAE model implementation
# https://papers.nips.cc/paper/5775-learning-structured-output-
# representation-using-deep-conditional-generative-models

import torch
import pytorch_lightning as pl

import sys
sys.path.append('..')

from networks import MnistCNNDecoder, ExprDiffDecoder, ConditionedDecoder, RNNEncoder, FinetunedEncoder
from networks import MnistCNNEncoder, ExprDiffEncoder, JointEncoder, RNNDecoder, FinetunedDecoder


class VIB(pl.LightningModule):
    def __init__(self, dataset='paired_mnist'):
        super(VIB, self).__init__()
        self.dataset = dataset

        if self.dataset == 'paired_mnist':
            self.z_dim = 16
            
            self.loss_rec_lambda_x = 10
            self.beta = 0.01

            self.enc_y = MnistCNNEncoder(out_dim=2 * self.z_dim)
            self.dec_x = MnistCNNDecoder(in_dim=self.z_dim)
        elif self.dataset == 'lincs_rnn':
            self.z_dim = 10

            self.loss_rec_lambda_x = 5
            self.beta = 1
            
            self.enc_y = ExprDiffEncoder(out_dim=2 * self.z_dim)

            rnn = RNNDecoder(in_dim=44)
            rnn.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cpu'))
            self.dec_x = FinetunedDecoder(rnn, in_dim=self.z_dim) 
        elif self.dataset == 'lincs_rnn_reverse':
            self.z_dim = 10

            self.loss_rec_lambda_x = 1
            self.beta = 1
            
            rnn = RNNEncoder(out_dim=88)
            rnn.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.enc_y = FinetunedEncoder(rnn, out_dim=2 * self.z_dim)

            self.dec_x = ExprDiffDecoder(in_dim=self.z_dim)
            

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

    def get_latents(self, batch):
        # pair of objects
        x, y = batch

        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        # sample z
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)

        return z_y_sample

    def get_log_p_x_by_y(self, batch):
        return self.dec_x.get_log_prob(batch[0], self.get_latents(batch))
    
    def restore(self, batch):
        # pair of objects
        x, y = batch

        # compute encoder outputs and split them into joint and exclusive parts
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        
        rest_x = self.dec_x.sample(z=z_y_sample)
    
        return (rest_x, y)
    
    def sample(self, y):
        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        # sample z
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)

        sampled_x = self.dec_x.sample(z=z_y_sample)
        return sampled_x

    
    def training_step(self, batch, batch_nb):
        # pair of objects
        x, y = batch

        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        # sample z
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)

        # compute kl divergence
        kl = self.kl_div(p_z_y_means, p_z_y_logvar)

        # compute reconstrunction loss
        x_by_z_logprob = self.dec_x.get_log_prob(x=x, z=z_y_sample)

        loss = (-self.loss_rec_lambda_x * x_by_z_logprob + self.beta * kl)

        return {'loss': loss,
                'log': {
                    'x_rec': -x_by_z_logprob,
                    'y_kl': kl}
                }

    def configure_optimizers(self):
        if self.dataset == 'paired_mnist':
            return torch.optim.Adam(self.parameters(), lr=1e-3)
        elif 'lincs' in self.dataset:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
