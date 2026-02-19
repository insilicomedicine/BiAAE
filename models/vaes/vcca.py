# JMVAE model implementation
# https://arxiv.org/abs/1611.01891

import torch
import pytorch_lightning as pl

import sys
sys.path.append('..')

from networks import MnistCNNDecoder, ExprDiffDecoder, ConditionedDecoder, RNNEncoder, FinetunedEncoder
from networks import MnistCNNEncoder, ExprDiffEncoder, JointEncoder, RNNDecoder, FinetunedDecoder


class VCCA(pl.LightningModule):
    def __init__(self, dataset='paired_mnist'):
        super(VCCA, self).__init__()
        self.dataset = dataset

        if self.dataset == 'paired_mnist':
            self.z_dim = 16
            self.joint_dim = 4

            self.enc_x = MnistCNNEncoder(out_dim=2 * (self.z_dim - self.joint_dim)) 
            self.enc_y = MnistCNNEncoder(out_dim=2 * self.z_dim)

            self.loss_rec_lambda_x = 10
            self.loss_rec_lambda_y = 10
            
            self.beta = 0.01
            
            self.dec_x = MnistCNNDecoder(in_dim=self.z_dim)
            self.dec_y = MnistCNNDecoder(in_dim=self.z_dim)
        elif self.dataset == 'lincs_rnn':
            self.z_dim = 20
            self.joint_dim = 10

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.enc_x = FinetunedEncoder(rnn_1, out_dim=2 * (self.z_dim - self.joint_dim))

            self.enc_y = ExprDiffEncoder(out_dim=2 * self.z_dim)

            self.loss_rec_lambda_x = 5
            self.loss_rec_lambda_y = 1
            
            self.beta = 1
            
            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cpu'))
            self.dec_x = FinetunedDecoder(rnn_2, in_dim=self.z_dim)
            self.dec_y = ExprDiffDecoder(in_dim=self.z_dim)
        elif self.dataset == 'lincs_rnn_reverse':
            self.z_dim = 20
            self.joint_dim = 10

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.enc_y = FinetunedEncoder(rnn_1, out_dim=2 * self.z_dim)

            self.enc_x = ExprDiffEncoder(out_dim=2 * (self.z_dim - self.joint_dim))

            self.loss_rec_lambda_x = 1
            self.loss_rec_lambda_y = 0.2
            
            self.beta = 1
            
            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cpu'))
            self.dec_y = FinetunedDecoder(rnn_2, in_dim=self.z_dim)
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
        p_z_x_means, p_z_x_logvar = torch.split(self.enc_x(x), self.z_dim - self.joint_dim, -1)
        p_zs_y_means, p_zs_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        p_z_y_means, p_s_y_means = torch.split(p_zs_y_means, self.z_dim - self.joint_dim, -1)
        p_z_y_logvar, p_s_y_logvar = torch.split(p_zs_y_logvar, self.z_dim - self.joint_dim, -1)
        
        z_x_sample = self.sample_repar_z(p_z_x_means, p_z_x_logvar)
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        s_y_sample = self.sample_repar_z(p_s_y_means, p_s_y_logvar)
        
        rest_x = self.dec_x.sample(torch.cat((z_x_sample, s_y_sample), -1))
        rest_y = self.dec_y.sample(torch.cat((z_y_sample, s_y_sample), -1))
    
        return (rest_x, rest_y)
    
    
    def sample(self, y):
        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        # sample z
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)

        sampled_x = self.dec_x.sample(z=z_y_sample)
        return sampled_x

    def sample_y(self, x):
        # compute proposal distributions
        p_z_x_means, p_z_x_logvar = torch.split(self.enc_x(x), self.z_dim, -1)

        # sample z
        z_x_sample = self.sample_repar_z(p_z_x_means, p_z_x_logvar)

        sampled_y = self.dec_x.sample(z=z_x_sample)
        return sampled_y


    
    def training_step(self, batch, batch_nb):
        # pair of objects
        x, y = batch

        # compute proposal distributions
        p_z_x_means, p_z_x_logvar = torch.split(self.enc_x(x), self.z_dim - self.joint_dim,
                                                -1)
        p_zs_y_means, p_zs_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        p_z_y_means, p_s_y_means = torch.split(p_zs_y_means,
                                               self.z_dim - self.joint_dim, -1)
        p_z_y_logvar, p_s_y_logvar = torch.split(p_zs_y_logvar,
                                               self.z_dim - self.joint_dim, -1)

        # sample z
        z_x_sample = self.sample_repar_z(p_z_x_means, p_z_x_logvar)
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        s_y_sample = self.sample_repar_z(p_s_y_means, p_s_y_logvar)

        # compute kl divergence
        z_x_kl = self.kl_div(p_z_x_means, p_z_x_logvar)
        z_y_kl = self.kl_div(p_z_y_means, p_z_y_logvar)
        s_y_kl = self.kl_div(p_s_y_means, p_s_y_logvar)

        # compute reconstrunction loss
        x_z_logprob = self.dec_x.get_log_prob(x,
                                              torch.cat((z_x_sample, s_y_sample), -1))
        y_z_logprob = self.dec_y.get_log_prob(y,
                                              torch.cat((z_y_sample, s_y_sample), -1))

        loss = -(self.loss_rec_lambda_x * x_z_logprob + self.loss_rec_lambda_y * y_z_logprob) + self.beta * (z_x_kl + z_y_kl + s_y_kl)

        return {'loss': loss,
                'log': {
                    'x_rec': -x_z_logprob, 'x_kl': z_x_kl,
                    'y_rec': -y_z_logprob, 'y_kl': z_y_kl,
                    'common_kl': s_y_kl}
                }

    def configure_optimizers(self):
        if self.dataset == 'paired_mnist':
            return torch.optim.Adam(self.parameters(), lr=3e-4)
        elif 'lincs' in self.dataset:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
