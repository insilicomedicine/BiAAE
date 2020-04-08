# proposed model implementation

import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl

import sys
sys.path.append('..')

from networks import MnistCNNDecoder, ExprDiffDecoder, ConditionedDecoder, RNNEncoder, FinetunedEncoder
from networks import MnistCNNEncoder, ExprDiffEncoder, JointEncoder, RNNDecoder, FinetunedDecoder
from networks import FCDiscriminator

from torch.optim.lr_scheduler import StepLR

class BiAAE(pl.LightningModule):
    def __init__(self, dataset='paired_mnist'):
        super(BiAAE, self).__init__()
        self.dataset = dataset

        if self.dataset == 'paired_mnist':
            self.z_dim = 16
            self.joint_dim = 4

            self.loss_rec_lambda_x = 10
            self.loss_rec_lambda_y = 10

            self.loss_normal_lambda = 0.3
            self.loss_indep_lambda_x = 1
            self.loss_indep_lambda_y = 1
            self.loss_mse_lambda = 0.1

            self.discr_steps = 1
            self.gen_steps = 1

            self.enc_x = MnistCNNEncoder(out_dim=self.z_dim)
            self.enc_y = MnistCNNEncoder(out_dim=self.z_dim)

            self.dec_x = MnistCNNDecoder(in_dim=self.z_dim)
            self.dec_y = MnistCNNDecoder(in_dim=self.z_dim)

            self.discr = FCDiscriminator(in_dim=2 * self.z_dim -
                                         self.joint_dim,
                                        use_sigmoid=False)

            self.discr_indep_x = FCDiscriminator(in_dim=2 * self.z_dim -
                                                 self.joint_dim,
                                                 use_sigmoid=False)

            self.discr_indep_y = FCDiscriminator(in_dim=2 * self.z_dim -
                                                 self.joint_dim,
                                                 use_sigmoid=False)
            
        elif self.dataset == 'lincs_rnn':
            self.z_dim = 20
            self.joint_dim = 10

            self.loss_rec_lambda_x = 5
            self.loss_rec_lambda_y = 1

            self.loss_normal_lambda = 0.5
            self.loss_indep_lambda_x = 0.5
            self.loss_indep_lambda_y = 0.5
            self.loss_mse_lambda = 0.5

            self.discr_steps = 3
            self.gen_steps = 1

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cuda:1'))
            self.enc_x = FinetunedEncoder(rnn_1, out_dim=self.z_dim)
            self.enc_y = ExprDiffEncoder(out_dim=self.z_dim)
            
            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cuda:1'))
            self.dec_x = FinetunedDecoder(rnn_2, in_dim=self.z_dim)

            self.dec_y = ExprDiffDecoder(in_dim=self.z_dim)
            
            self.discr = FCDiscriminator(in_dim=2 * self.z_dim - self.joint_dim,
                                        use_sigmoid=False)

            self.discr_indep_x = FCDiscriminator(in_dim=2 * self.z_dim - self.joint_dim,
                                                 use_sigmoid=False)

            self.discr_indep_y = FCDiscriminator(in_dim=2 * self.z_dim - self.joint_dim,
                                                 use_sigmoid=False)
        elif self.dataset == 'lincs_rnn_reverse':
            self.z_dim = 20
            self.joint_dim = 10

            self.loss_rec_lambda_x = 1
            self.loss_rec_lambda_y = 0.2
            
            self.loss_normal_lambda = 0.5
            self.loss_indep_lambda_x = 0.5
            self.loss_indep_lambda_y = 0.5
            self.loss_mse_lambda = 0.5
            

            self.discr_steps = 3
            self.gen_steps = 1

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cuda:1'))
            self.enc_y = FinetunedEncoder(rnn_1, out_dim=self.z_dim)
            self.enc_x = ExprDiffEncoder(out_dim=self.z_dim)
            
            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cuda:1'))
            self.dec_y = FinetunedDecoder(rnn_2, in_dim=self.z_dim)

            self.dec_x = ExprDiffDecoder(in_dim=self.z_dim)
            
            self.discr = FCDiscriminator(in_dim=2 * self.z_dim -
                                         self.joint_dim,
                                        use_sigmoid=False)

            self.discr_indep_x = FCDiscriminator(in_dim=2 * self.z_dim -
                                                 self.joint_dim,
                                                 use_sigmoid=False)

            self.discr_indep_y = FCDiscriminator(in_dim=2 * self.z_dim -
                                                 self.joint_dim,
                                                 use_sigmoid=False)
    # ------------------------------------------------------------------------
    #               TRAINING
    def get_latents(self, batch):
        # pair of objects
        x, y = batch
        
        z_y, s_y = torch.split(self.enc_y(y), self.z_dim - self.joint_dim, -1)
        z_x = torch.randn_like(z_y)

        return torch.cat((z_x, s_y), 1)

    def get_log_p_x_by_y(self, batch):
        return self.dec_x.get_log_prob(batch[0], self.get_latents(batch))
    
    def restore(self, batch):
        # pair of objects
        x, y = batch

        # compute encoder outputs and split them into joint and exclusive parts
        z_x, s_x = torch.split(self.enc_x(x), self.z_dim - self.joint_dim, -1)
        z_y, s_y = torch.split(self.enc_y(y), self.z_dim - self.joint_dim, -1)
        
        x_rest = self.dec_x.sample(torch.cat((z_x, s_x), 1))
        y_rest = self.dec_y.sample(torch.cat((z_y, s_y), 1))
    
        return (x_rest, y_rest)
    
    def sample(self, y):
        # sample z
        z_y, s_y = torch.split(self.enc_y(y), self.z_dim - self.joint_dim, -1)
        z_x = torch.randn_like(z_y)

        sampled_x = self.dec_x.sample(z=torch.cat((z_x, s_y), 1))
        return sampled_x
    
    def sample_y(self, x):
        # sample y
        z_x, s_x = torch.split(self.enc_x(x), self.z_dim - self.joint_dim, -1)
        z_y = torch.randn_like(z_x)

        sampled_x = self.dec_y.sample(z=torch.cat((z_y, s_x), 1))
        return sampled_x

    def training_step(self, batch, batch_nb, optimizer_i):
        # pair of objects
        x, y = batch

        # compute encoder outputs and split them into joint and exclusive parts
        z_x, s_x = torch.split(self.enc_x(x), self.z_dim - self.joint_dim, -1)
        z_y, s_y = torch.split(self.enc_y(y), self.z_dim - self.joint_dim, -1)
        
        if optimizer_i == 0:# GENERATOR LOSS
            # Reconstruction losses
            loss_x_rec = -(self.dec_x.get_log_prob(x, torch.cat((z_x, s_y), 1)).mean() +
                           self.dec_x.get_log_prob(x, torch.cat((z_x, s_x), 1)).mean()) / 2
            
            loss_y_rec = -(self.dec_y.get_log_prob(y, torch.cat((z_y, s_y), 1)).mean() + 
                           self.dec_y.get_log_prob(y, torch.cat((z_y, s_x), 1)).mean()) / 2

            # compute mse between common parts
            loss_mse =  torch.norm(s_x - s_y, p=2, dim=-1).mean()
            
            # run discriminators
            discr_outputs = self.discr(
                torch.cat((torch.cat((z_x, s_x, z_y), dim=-1),
                          torch.cat((z_x, s_y, z_y), dim=-1)), dim=0)
            )
            loss_norm = nn.BCEWithLogitsLoss()(discr_outputs, torch.ones_like(discr_outputs))
            
            discr_outputs_x = self.discr_indep_x(torch.cat((z_x, s_y.detach(), z_y.detach()), dim=-1))
            loss_indep_x = nn.BCEWithLogitsLoss()(discr_outputs_x, torch.ones_like(discr_outputs_x))
            
            discr_outputs_y = self.discr_indep_y(torch.cat((z_x.detach(), s_x.detach(), z_y), dim=-1))
            loss_indep_y = nn.BCEWithLogitsLoss()(discr_outputs_y, torch.ones_like(discr_outputs_y))

            g_loss = (loss_x_rec * self.loss_rec_lambda_x +
                      loss_y_rec * self.loss_rec_lambda_y +
                      loss_norm * self.loss_normal_lambda + 
                      loss_indep_x * self.loss_indep_lambda_x + 
                      loss_indep_y * self.loss_indep_lambda_y + 
                      loss_mse * self.loss_mse_lambda)

            return {'loss': g_loss,
                    'log': {
                        'loss_g': g_loss,
                        'x_rec': loss_x_rec,
                        'y_rec': loss_y_rec,
                        'loss_norm': loss_norm,
                        'loss_indep_x': loss_indep_x,
                        'loss_indep_y': loss_indep_y,
                        'loss_mse': loss_mse
                    }
                   }
        
        elif optimizer_i == 1:# DISCRIMINATOR LOSS
            z_x = z_x.detach()
            s_x = s_x.detach()
            s_y = s_y.detach()
            z_y = z_y.detach()

            # normal noise loss
            real_inputs_1 = torch.cat(
                (torch.cat((z_x, s_x, z_y), dim=-1),
                 torch.cat((z_x, s_y, z_y), dim=-1)), dim=0)
            real_dec_out_1 = self.discr(real_inputs_1)

            fake_inputs_1 = torch.randn_like(real_inputs_1)
            fake_dec_out_1 = self.discr(fake_inputs_1)

            probs_1 = torch.cat((real_dec_out_1, fake_dec_out_1), 0)
            targets_1 = torch.cat((torch.zeros_like(real_dec_out_1), 
                                 torch.ones_like(fake_dec_out_1)), 0)

            d_loss_normal = nn.BCEWithLogitsLoss()(probs_1, targets_1)
            
            # independece loss
            real_inputs_2 = torch.cat((z_x, s_y, z_y), dim=-1)
            real_dec_out_2 = self.discr_indep_x(real_inputs_2)
            
            real_input_shuffled_2 = torch.cat((z_x[np.random.permutation(z_x.shape[0])], s_y, z_y), dim=-1)
            fake_dec_out_2 = self.discr_indep_x(real_input_shuffled_2)
            
            probs_2 = torch.cat((real_dec_out_2, fake_dec_out_2), 0)
            targets_2 = torch.cat((torch.zeros_like(real_dec_out_2), 
                                 torch.ones_like(fake_dec_out_2)), 0)

            d_loss_indep_x = nn.BCEWithLogitsLoss()(probs_2, targets_2)
            
            #-----------------------
            
            real_inputs_3 = torch.cat((z_x, s_x, z_y), dim=-1)
            real_dec_out_3 = self.discr_indep_y(real_inputs_3)
            
            real_input_shuffled_3 = torch.cat((z_x, s_x, z_y[np.random.permutation(z_y.shape[0])]), dim=-1)
            fake_dec_out_3 = self.discr_indep_y(real_input_shuffled_3)
            
            probs_3 = torch.cat((real_dec_out_3, fake_dec_out_3), 0)
            targets_3 = torch.cat((torch.zeros_like(real_dec_out_3), 
                                 torch.ones_like(fake_dec_out_3)), 0)

            d_loss_indep_y = nn.BCEWithLogitsLoss()(probs_3, targets_3)
            
            return {
                'loss': d_loss_normal + d_loss_indep_x + d_loss_indep_y,
                'log': {'loss_d_normal': d_loss_normal,
                        'loss_d_indep_x': d_loss_indep_x,
                        'loss_d_indep_y': d_loss_indep_y
                       }
                    }


    def configure_optimizers(self):
        gen_params = torch.nn.ModuleList([self.enc_x, self.dec_x, self.enc_y, self.dec_y])
        discr_params = torch.nn.ModuleList([self.discr_indep_x, self.discr_indep_y, self.discr])

        gen_optim = torch.optim.Adam(gen_params.parameters(), lr=3e-4, betas=(0.5, 0.9))
        discr_optim = torch.optim.Adam(discr_params.parameters(), lr=3e-4, betas=(0.5, 0.9))

        discriminator_sched = StepLR(discr_optim, step_size=1000, gamma=0.99)

        return [gen_optim, discr_optim], [discriminator_sched]
    
    def zero_grad(self):
        self.enc_x.zero_grad()
        self.dec_x.zero_grad()
        self.enc_y.zero_grad()
        self.dec_y.zero_grad()

        self.discr.zero_grad()
        self.discr_indep_x.zero_grad()
        self.discr_indep_y.zero_grad()
        
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, optimizer_closure):
        discr_step = (batch_nb % (self.discr_steps + self.gen_steps)) < \
                     self.discr_steps

        gen_step = (not discr_step)

        if optimizer_i == 0:
            if gen_step:
                optimizer.step()
            optimizer.zero_grad()
            self.zero_grad()

        if optimizer_i == 1:
            if discr_step:
                optimizer.step()
            optimizer.zero_grad()
            self.zero_grad()

        if optimizer_i > 1:
            optimizer.step()
            optimizer.zero_grad()
            self.zero_grad()

