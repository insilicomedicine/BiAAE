# CVAE model implementation
# https://papers.nips.cc/paper/5775-learning-structured-output-
# representation-using-deep-conditional-generative-models

import torch
import pytorch_lightning as pl

import sys
sys.path.append('..')

from networks import MnistCNNDecoder, ExprDiffDecoder, ConditionedDecoder, RNNEncoder, FinetunedEncoder
from networks import MnistCNNEncoder, ExprDiffEncoder, JointEncoder, RNNDecoder, FinetunedDecoder


class CVAE(pl.LightningModule):
    def __init__(self, dataset='paired_mnist'):
        super(CVAE, self).__init__()
        self.dataset = dataset

        if self.dataset == 'paired_mnist':
            self.z_dim = 8
    
            self.loss_rec_lambda_x = 10
            self.loss_rec_lambda_y = 10
            
            self.beta = 0.01
            
            self.enc_y = MnistCNNEncoder(out_dim=2 * self.z_dim)

            self.enc_xy = JointEncoder(
                MnistCNNEncoder(out_dim=2 * self.z_dim, short_tail=True),
                MnistCNNEncoder(out_dim=2 * self.z_dim, short_tail=True),
                out_dim=2 * self.z_dim
            )

            self.dec_x_cond = ConditionedDecoder(
                dec=MnistCNNDecoder(in_dim=self.z_dim + self.z_dim),
                cond=MnistCNNEncoder(out_dim=self.z_dim, short_tail=True)
            )
        elif self.dataset == 'lincs_rnn':
            self.z_dim = 10

            self.loss_rec_lambda_x = 5
            self.loss_rec_lambda_y = 1
            
            self.beta = 1
            
            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cuda:1'))
            enc_x = FinetunedEncoder(rnn_1, out_dim=2 * self.z_dim)

            self.enc_xy = JointEncoder(
                enc_x,
                ExprDiffEncoder(out_dim=2 * self.z_dim),
                out_dim=2 * self.z_dim
            )

            self.enc_y = ExprDiffEncoder(out_dim=2 * self.z_dim)

            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cuda:1'))
            dec_x = FinetunedDecoder(rnn_2, in_dim=2 * self.z_dim)
            self.dec_x_cond = ConditionedDecoder(
                dec=dec_x,
                cond=ExprDiffEncoder(out_dim=self.z_dim)
            )
            
        elif self.dataset == 'lincs_rnn_reverse':
            self.z_dim = 10

            self.loss_rec_lambda_x = 1
            self.loss_rec_lambda_y = 0.2
            
            self.beta = 1
            
            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cuda:1'))
            self.enc_y = FinetunedEncoder(rnn_1, out_dim=2 * self.z_dim)
            
            self.enc_xy = JointEncoder(
                ExprDiffEncoder(out_dim=2 * self.z_dim),
                FinetunedEncoder(rnn_1, out_dim=2 * self.z_dim),
                out_dim=2 * self.z_dim
            )

            rnn_2 = RNNEncoder(out_dim=88)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cuda:1'))
            
            self.dec_x_cond = ConditionedDecoder(
                dec=ExprDiffDecoder(in_dim=2 * self.z_dim),
                cond=FinetunedEncoder(rnn_2, out_dim=self.z_dim)
            )

            
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
        return self.dec_x_cond.get_log_prob(batch[0], self.get_latents(batch), cond=batch[1])

    def restore(self, batch):
        # pair of objects
        x, y = batch

        # compute encoder outputs and split them into joint and exclusive parts
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        
        rest_x = self.dec_x_cond.sample(z=z_y_sample, cond=y)
    
        return (rest_x, y)
    
    def sample(self, y):
        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)

        # sample z
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)

        sampled_x = self.dec_x_cond.sample(z=z_y_sample, cond=y)
        return sampled_x

    def training_step(self, batch, batch_nb):
        # pair of objects
        x, y = batch

        # compute proposal distributions
        p_z_y_means, p_z_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        p_z_xy_means, p_z_xy_logvar = torch.split(self.enc_xy((x, y)),
                                                  self.z_dim, -1)

        # sample z
        z_xy_sample = self.sample_repar_z(p_z_xy_means, p_z_xy_logvar)

        # compute kl divergence
        xy_y_kl = self.kl_div(p_z_xy_means, p_z_xy_logvar,
                              p_z_y_means, p_z_y_logvar)

        # compute reconstrunction loss
        x_by_z_y_logprob = self.dec_x_cond.get_log_prob(x=x,
                                                        z=z_xy_sample,
                                                        cond=y)

        loss = (-self.loss_rec_lambda_x * x_by_z_y_logprob + self.beta * xy_y_kl)

        return {'loss': loss,
                'log': {
                    'x_rec': -x_by_z_y_logprob,
                    'y_kl': xy_y_kl}
                }

    def configure_optimizers(self):
        if self.dataset == 'paired_mnist':
            return torch.optim.Adam(self.parameters(), lr=3e-4)
        elif 'lincs' in self.dataset:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
