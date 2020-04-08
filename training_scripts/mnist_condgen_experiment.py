import sys
sys.path.append('..')

import argparse

import torch

from networks import MnistCNNEncoder, JointEncoder
from networks import FCDiscriminator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

import numpy as np

from training_scripts.pretrain_mnist_clf import MNISTClassifier

from models import BiAAE, Lat_SAAE, CVAE, VCCA, JMVAE, VIB, UniAAE, SAAE



def make_conditional_experiment(model):
    class CondGenExperiment(model):
        def __init__(self):
            super().__init__(dataset='paired_mnist')

            self.mine_enc = MnistCNNEncoder(out_dim=self.z_dim)
            self.mine_fc = FCDiscriminator(in_dim=2 * self.z_dim)

            # classifier to measure accuracy of generation
            self.mnist_clf = MNISTClassifier()
            self.mnist_clf.load_state_dict(
                torch.load('../saved_models/mnist_clf.ckpt'))
            self.mnist_clf.eval()
            self.mnist_clf.freeze()

        def mi_computation(self, batch, z):
            x, y = batch
            x = x.detach()
            y = y.detach()
            z = z.detach()
            z_shuffled = z[np.random.permutation(z.shape[0])]

            x_lat = self.mine_enc(x)

            t = self.mine_fc(torch.cat((x_lat, z), dim=-1))
            et = torch.exp(self.mine_fc(torch.cat((x_lat, z_shuffled), dim=-1)))

            mi_lb = t.mean() - torch.log(et.mean())

            return mi_lb

        def training_step(self, batch, batch_nb, *args):
            stats = super().training_step(batch, batch_nb, *args)

            if len(args) == 0: # VAE-like approaches
                batch = (batch[0].detach(), batch[1].detach())

                z = self.get_latents(batch).detach()

                mi = self.mi_computation(batch, z)

                stats['loss'] += -mi
                stats['log']['mi[xz|y]'] = mi
            else: # AAE-like approaches
                if args[0] == 2: # optimizer number
                    batch = (batch[0].detach(), batch[1].detach())

                    z = self.get_latents(batch).detach()

                    mi = self.mi_computation(batch, z)

                    stats = {}
                    stats['loss'] = -mi
                    stats['log'] = {}
                    stats['log']['mi[xz|y]'] = mi

            return stats

        def configure_optimizers(self):
            optim = super().configure_optimizers()

            if isinstance(optim, tuple): # AAE like approaches
                mi_params = torch.nn.ModuleList([self.mine_fc, self.mine_enc]).parameters()

                optim[0].append(torch.optim.Adam(mi_params, lr=3e-4))

            return optim

        def validation_step(self, batch, batch_nb):
            # compute MINE on validation stage
            stats = super().validation_step()

            if stats is None:
                stats = {}

            z = self.get_latents(batch).detach()
            
            mi = self.mi_computation(batch, z)
            stats['mi[xz|y]'] = mi

            # perform sampling for visual expection of results and computing
            # accuracy metric

            x, y = batch
            sampled_x = self.sample(y)

            tgt = self.mnist_clf.predict(sampled_x)
            stats['acc'] = (self.mnist_clf.predict(x) == tgt).float().mean()

            stats['nll'] = -self.mnist_clf.get_logits(sampled_x)[torch.range(0, tgt.shape[0]-1).long(), tgt].mean()
            
            x_rest, y_rest = self.restore(batch)
            
            if batch_nb == 0:
                stats['sampled_pair'] = (sampled_x.cpu().numpy(), 
                                         x_rest.cpu().numpy(), x.cpu().numpy(), 
                                         y_rest.cpu().numpy(), y.cpu().numpy())

            return stats

        def validation_end(self, outputs):
            # plot samples to experiment
            if self.logger is not None:
                sampled = outputs[0]['sampled_pair']

                fig = plt.figure(num=0, figsize=(10, 10), dpi=300)
                for i in range(5):
                    for j in range(5):
                        plt.subplot(5, 5, i + 1 + 5 * j)
                        plt.axis('off')
                        plt.imshow(sampled[j][i][0].transpose(1, 0), cmap='gray')

                fig.canvas.draw()

                buf = fig.canvas.tostring_rgb()
                ncols, nrows = fig.canvas.get_width_height()
                fig_array = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3) / 255.
                fig_array = fig_array.transpose(2, 0, 1)

                self.logger.experiment.add_image('samples', fig_array)

            # compute mean values of validation statistics
            val_stats = {}
            
            for key in outputs[0].keys():
                if key != 'sampled_pair':
                    val_stats['val_' + key] = \
                        torch.stack([x[key] for x in outputs]).mean()

            return {'log': val_stats}

        @pl.data_loader
        def train_dataloader(self):
            data = np.load('../data/noisy_mnist.npz')
            dataset = TensorDataset(
                    *[torch.from_numpy(data['train'][i].astype('float32')).
                          reshape(-1, 1, 28, 28) for i in range(2)])

            dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                                        drop_last=False)

            return dataloader

        @pl.data_loader
        def val_dataloader(self):
            data = np.load('../data/noisy_mnist.npz')
            dataset = TensorDataset(
                    *[torch.from_numpy(data['valid'][i].astype('float32')).
                          reshape(-1, 1, 28, 28) for i in range(2)])

            dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                                    drop_last=False)

            return dataloader

    return CondGenExperiment


if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description='Script to perform conditional'
                                                 ' generation training_scripts')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default='biaae')

    args = parser.parse_args()

    models_dict = {'biaae': BiAAE, 'lat_saae': Lat_SAAE, 'uniaae': UniAAE, 
                   'cvae': CVAE, 'vcca': VCCA, 'jmvae': JMVAE, 'vib': VIB,
                   'saae': SAAE}

    model = make_conditional_experiment(models_dict[args.model])()

    model_checkpoint = ModelCheckpoint('../saved_models/pmnist_' + args.model,
                                       save_best_only=False, period=10)
    
    logger = TestTubeLogger(save_dir='../logs/', name='pmnist_' + args.model)
    tr = pl.Trainer(gpus=([] if (args.gpu < 0) else [args.gpu]),
                    logger=logger,
                    checkpoint_callback=model_checkpoint,
                    early_stop_callback=False,
                    max_nb_epochs=300, show_progress_bar=False,
                   default_save_path=None)
    tr.fit(model)
    torch.save(model.state_dict(), '../saved_models/pmnist_{}.ckpt'.format(args.model))
