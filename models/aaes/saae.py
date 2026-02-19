# proposed model implementation

import torch
from torch import nn
import pytorch_lightning as pl

import sys
sys.path.append('..')

from networks import MnistCNNDecoder, ExprDiffDecoder, ConditionedDecoder, RNNEncoder, FinetunedEncoder
from networks import MnistCNNEncoder, ExprDiffEncoder, JointEncoder, RNNDecoder, FinetunedDecoder
from networks import FCDiscriminator

from torch.optim.lr_scheduler import StepLR

class SAAE(pl.LightningModule):
    def __init__(self, dataset='paired_mnist'):
        super(SAAE, self).__init__()
        self.dataset = dataset

        if self.dataset == 'paired_mnist':
            self.z_dim = 8

            self.loss_rec_lambda_x = 10

            self.loss_latent_lambda = 2

            self.discr_steps = 1
            self.gen_steps = 1

            self.enc_x = MnistCNNEncoder(out_dim=self.z_dim // 2)
            self.enc_y = MnistCNNEncoder(out_dim=self.z_dim // 2)

            self.dec_x = MnistCNNDecoder(in_dim=self.z_dim)

            self.discr = FCDiscriminator(in_dim=self.z_dim // 2, 
                                         use_sigmoid=False)
        elif self.dataset == 'lincs_rnn':
            self.z_dim = 20

            self.loss_rec_lambda_x = 5
            self.loss_rec_lambda_y = 1
            self.loss_latent_lambda = 1

            self.discr_steps = 1
            self.gen_steps = 3

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.enc_x = FinetunedEncoder(rnn_1, out_dim=self.z_dim // 2)
            self.enc_y = ExprDiffEncoder(out_dim=self.z_dim // 2)

            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cpu'))
            self.dec_x = FinetunedDecoder(rnn_2, in_dim=self.z_dim)

            self.dec_y = ExprDiffDecoder(in_dim=self.z_dim // 2)
            
            self.discr = FCDiscriminator(in_dim=self.z_dim // 2,
                                        use_sigmoid=False)
            
        elif self.dataset == 'lincs_rnn_reverse':
            self.z_dim = 20

            self.loss_rec_lambda_x = 1
            self.loss_rec_lambda_y = 0.2
            self.loss_latent_lambda = 1

            self.discr_steps = 1
            self.gen_steps = 3

            rnn_1 = RNNEncoder(out_dim=88)
            rnn_1.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.enc_y = FinetunedEncoder(rnn_1, out_dim=self.z_dim // 2)

            self.enc_x = ExprDiffEncoder(out_dim=self.z_dim // 2)

            rnn_2 = RNNDecoder(in_dim=44)
            rnn_2.load_state_dict(torch.load('../saved_models/rnn_dec.ckpt', map_location='cpu'))
            self.dec_y = FinetunedDecoder(rnn_2, in_dim=self.z_dim // 2)

            self.dec_x = ExprDiffDecoder(in_dim=self.z_dim)
            
            self.discr = FCDiscriminator(in_dim=self.z_dim // 2,
                                        use_sigmoid=False)
        

    # ------------------------------------------------------------------------
    #               TRAINING
    def get_latents(self, batch):
        # pair of objects
        x, y = batch

        z_y = self.enc_y(y)
        z_x = torch.randn_like(z_y)
        
        return torch.cat((z_x, z_y), 1)
    
    def get_log_p_x_by_y(self, batch):
        return self.dec_x.get_log_prob(batch[0], self.get_latents(batch))

    def restore(self, batch):
        # pair of objects
        x, y = batch

        # compute encoder outputs and split them into joint and exclusive parts
        z_x = self.enc_x(x)
        z_y = self.enc_y(y)
        
        x_rest = self.dec_x.sample(torch.cat((z_x, z_y), 1))
    
        return (x_rest, y)
    
    def sample(self, y):
        # sample z
        z_y = self.enc_y(y)
        z_x = torch.randn_like(z_y)

        sampled_x = self.dec_x.sample(z=torch.cat((z_x, z_y), 1))
        return sampled_x

    def training_step(self, batch, batch_nb, optimizer_i):
        # pair of objects
        x, y = batch

        z_x = self.enc_x(x)
        z_y = self.enc_y(y)

        if optimizer_i == 0: # GENERATOR LOSS
            rec_x = -self.dec_x.get_log_prob(x, torch.cat((z_x, z_y), 1)).mean()

            # run discriminators
            discr_outputs = self.discr(z_x)
            latent_loss = nn.BCEWithLogitsLoss()(discr_outputs,
                                       torch.ones_like(discr_outputs))

            g_loss = (rec_x * self.loss_rec_lambda_x +
                      latent_loss * self.loss_latent_lambda)

            return {'loss': g_loss,
                    'log': {'loss_g': g_loss,
                             'x_rec': rec_x,
                             'loss_norm': latent_loss
                             }
                    }
        elif optimizer_i == 1: # DISCRIMINATOR LOSS
            z_x = z_x.detach()

            # Compare <z_x, s_x, z_y> or <z_x, s_y, z_y> vs N(0, I)
            real_inputs = z_x
            real_dec_out = self.discr(real_inputs)
            
            fake_inputs = torch.randn_like(real_inputs)
            fake_dec_out = self.discr(fake_inputs)

            probs = torch.cat((real_dec_out, fake_dec_out), 0)
            targets = torch.cat((torch.zeros_like(real_dec_out), 
                                 torch.ones_like(fake_dec_out)), 0)
            
            d_loss = nn.BCEWithLogitsLoss()(probs, targets)

            return {'loss': d_loss,
                    'log': {
                        'loss_d_x': d_loss
                        }
                    }

    def configure_optimizers(self):
        gen_params = torch.nn.ModuleList([self.enc_x, self.dec_x, self.enc_y])
        discr_params = torch.nn.ModuleList([self.discr])

        gen_optim = torch.optim.Adam(gen_params.parameters(), lr=3e-4, betas=(0.5, 0.9))
        discr_optim = torch.optim.Adam(discr_params.parameters(), lr=3e-4, betas=(0.5, 0.9))

        discriminator_sched = StepLR(discr_optim, step_size=5000, gamma=0.5)

        return [gen_optim, discr_optim], [discriminator_sched]
    
    def zero_grad(self):
        self.enc_x.zero_grad()
        self.dec_x.zero_grad()
        self.enc_y.zero_grad()
        self.discr.zero_grad()
        
    def optimizer_step(self, *args, **kwargs):
        """Compatibility wrapper for old/new PyTorch Lightning optimizer_step signatures."""
        batch_nb = kwargs.get('batch_idx', kwargs.get('batch_nb'))
        optimizer = kwargs.get('optimizer')
        optimizer_i = kwargs.get('optimizer_idx', kwargs.get('optimizer_i'))
        optimizer_closure = kwargs.get('optimizer_closure')

        if batch_nb is None and len(args) > 1:
            batch_nb = args[1]
        if optimizer is None and len(args) > 2:
            optimizer = args[2]
        if optimizer_i is None and len(args) > 3:
            optimizer_i = args[3]
        if optimizer_closure is None and len(args) > 4:
            optimizer_closure = args[4]

        discr_step = (batch_nb % (self.discr_steps + self.gen_steps)) < self.discr_steps
        gen_step = not discr_step

        def _step(opt, closure):
            if callable(closure):
                opt.step(closure=closure)
            else:
                opt.step()

        if optimizer_i == 0:
            if gen_step:
                _step(optimizer, optimizer_closure)
            optimizer.zero_grad()
            self.zero_grad()

        if optimizer_i == 1:
            if discr_step:
                _step(optimizer, optimizer_closure)
            optimizer.zero_grad()
            self.zero_grad()

        if optimizer_i > 1:
            _step(optimizer, optimizer_closure)
            optimizer.zero_grad()
            self.zero_grad()


