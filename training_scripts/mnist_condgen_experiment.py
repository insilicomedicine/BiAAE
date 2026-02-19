import argparse
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('..')

from models import BiAAE, CVAE, JMVAE, Lat_SAAE, SAAE, UniAAE, VCCA, VIB
from networks import FCDiscriminator, MnistCNNEncoder
from training_scripts.pretrain_mnist_clf import MNISTClassifier


def make_conditional_experiment(model):
    class CondGenExperiment(model):
        def __init__(self):
            super().__init__(dataset='paired_mnist')
            self.mine_enc = MnistCNNEncoder(out_dim=self.z_dim)
            self.mine_fc = FCDiscriminator(in_dim=2 * self.z_dim)

            self.mnist_clf = MNISTClassifier()
            self.mnist_clf.load_state_dict(torch.load('../saved_models/mnist_clf.ckpt', map_location='cpu'))
            self.mnist_clf.eval()
            self.mnist_clf.freeze()

        def mi_computation(self, batch, z):
            x, y = batch
            z_shuffled = z[np.random.permutation(z.shape[0])]
            x_lat = self.mine_enc(x.detach())
            t = self.mine_fc(torch.cat((x_lat, z.detach()), dim=-1))
            et = torch.exp(self.mine_fc(torch.cat((x_lat, z_shuffled), dim=-1)))
            return t.mean() - torch.log(et.mean())

        def training_step(self, batch, batch_nb, *args):
            stats = super().training_step(batch, batch_nb, *args)
            if len(args) == 0:
                z = self.get_latents((batch[0].detach(), batch[1].detach())).detach()
                mi = self.mi_computation(batch, z)
                stats['loss'] += -mi
                stats['log']['mi[xz|y]'] = mi
            elif args[0] == 2:
                z = self.get_latents((batch[0].detach(), batch[1].detach())).detach()
                mi = self.mi_computation(batch, z)
                stats = {'loss': -mi, 'log': {'mi[xz|y]': mi}}
            return stats

        def configure_optimizers(self):
            optim = super().configure_optimizers()
            if isinstance(optim, (tuple, list)) and len(optim) > 0:
                mi_params = torch.nn.ModuleList([self.mine_fc, self.mine_enc]).parameters()
                optim[0].append(torch.optim.Adam(mi_params, lr=3e-4))
            return optim

        def validation_step(self, batch, batch_nb):
            base_validation_step = getattr(super(), 'validation_step', None)
            stats = base_validation_step(batch, batch_nb) if callable(base_validation_step) else {}
            z = self.get_latents(batch).detach()
            mi = self.mi_computation(batch, z)
            stats['mi[xz|y]'] = mi

            x, y = batch
            sampled_x = self.sample(y)
            tgt = self.mnist_clf.predict(sampled_x)
            stats['acc'] = (self.mnist_clf.predict(x) == tgt).float().mean()
            idx = torch.arange(tgt.shape[0], device=tgt.device).long()
            stats['nll'] = -self.mnist_clf.get_logits(sampled_x)[idx, tgt].mean()

            x_rest, y_rest = self.restore(batch)
            if batch_nb == 0:
                stats['sampled_pair'] = (
                    sampled_x.cpu().numpy(),
                    x_rest.cpu().numpy(),
                    x.cpu().numpy(),
                    y_rest.cpu().numpy(),
                    y.cpu().numpy(),
                )
            return stats

        def validation_epoch_end(self, outputs):
            if self.logger is not None and outputs:
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
                fig_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3) / 255.0
                fig_array = fig_array.transpose(2, 0, 1)
                self.logger.experiment.add_image('samples', fig_array, self.current_epoch)

            for key in outputs[0].keys():
                if key != 'sampled_pair':
                    self.log('val_' + key, torch.stack([x[key] for x in outputs]).mean(), prog_bar=False)

        def train_dataloader(self):
            data = np.load('../data/noisy_mnist.npz')
            dataset = TensorDataset(
                *[torch.from_numpy(data['train'][i].astype('float32')).reshape(-1, 1, 28, 28) for i in range(2)]
            )
            return DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)

        def val_dataloader(self):
            data = np.load('../data/noisy_mnist.npz')
            dataset = TensorDataset(
                *[torch.from_numpy(data['valid'][i].astype('float32')).reshape(-1, 1, 28, 28) for i in range(2)]
            )
            return DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)

    return CondGenExperiment


if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Script to perform conditional generation training')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default='biaae')
    args = parser.parse_args()

    models_dict = {
        'biaae': BiAAE,
        'lat_saae': Lat_SAAE,
        'uniaae': UniAAE,
        'cvae': CVAE,
        'vcca': VCCA,
        'jmvae': JMVAE,
        'vib': VIB,
        'saae': SAAE,
    }

    model = make_conditional_experiment(models_dict[args.model])()
    model_checkpoint = ModelCheckpoint(dirpath='../saved_models', filename=f'pmnist_{args.model}' + '-{epoch:03d}', every_n_epochs=10, save_top_k=-1)
    logger = TensorBoardLogger(save_dir='../logs/', name='pmnist_' + args.model)

    trainer_kwargs = dict(logger=logger, callbacks=[model_checkpoint], enable_progress_bar=False, max_epochs=300)
    if args.gpu < 0:
        tr = pl.Trainer(accelerator='cpu', devices=1, **trainer_kwargs)
    else:
        tr = pl.Trainer(accelerator='gpu', devices=[args.gpu], **trainer_kwargs)

    tr.fit(model)
    torch.save(model.state_dict(), f'../saved_models/pmnist_{args.model}.ckpt')
