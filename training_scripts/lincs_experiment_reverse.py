import argparse
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.append('..')

from dataloader import LincsDataSet, LincsSampler
from models import BiAAE, CVAE, JMVAE, Lat_SAAE, SAAE, UniAAE, VCCA, VIB
from networks import ExprDiffEncoder, FCDiscriminator


def make_conditional_experiment(model):
    class CondGenExperiment(model):
        def __init__(self):
            super().__init__(dataset='lincs_rnn_reverse')
            self.mine_enc = ExprDiffEncoder(out_dim=self.z_dim)
            self.mine_fc = FCDiscriminator(in_dim=2 * self.z_dim)

        def mi_computation(self, batch, z):
            x, y = batch
            z_shuffled = z[np.random.permutation(z.shape[0])]
            x_lat = self.mine_enc(x)
            t = self.mine_fc(torch.cat((x_lat, z), dim=-1))
            et = torch.exp(self.mine_fc(torch.cat((x_lat, z_shuffled), dim=-1)))
            return t.mean() - torch.log(et.mean())

        def training_step(self, batch, batch_nb, *args):
            batch = (batch[1], batch[0])
            stats = super().training_step(batch, batch_nb, *args)
            if len(args) == 0:
                z = self.get_latents((batch[0].detach(), batch[1])).detach()
                mi = self.mi_computation(batch, z)
                stats['loss'] += -mi
                stats['log']['mi[xz|y]'] = mi
            elif args[0] == 2:
                z = self.get_latents((batch[0].detach(), batch[1])).detach()
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
            batch = (batch[1], batch[0])
            base_validation_step = getattr(super(), 'validation_step', None)
            stats = base_validation_step(batch, batch_nb) if callable(base_validation_step) else {}
            z = self.get_latents(batch)
            stats['mi[xz|y]'] = self.mi_computation(batch, z)
            x, y = batch
            sampled_x = self.sample(y)
            diff1 = x[:, :978]
            diff2 = sampled_x[:, :978]
            stats['rmse'] = torch.norm(diff2 - diff1, dim=-1).mean()
            return stats

        def validation_epoch_end(self, outputs):
            self.log('val_mi[xz|y]', torch.stack([x['mi[xz|y]'] for x in outputs]).mean(), prog_bar=False)
            self.log('val_rmse', torch.stack([x['rmse'] for x in outputs]).mean(), prog_bar=False)

        def train_dataloader(self):
            dataset = LincsSampler(LincsDataSet('../data/lincs'), test_set=0, use_smiles=True)
            return DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

        def val_dataloader(self):
            dataset = LincsSampler(LincsDataSet('../data/lincs'), test_set=1, use_smiles=True)
            return DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    return CondGenExperiment


if __name__ == '__main__':
    torch.manual_seed(777)

    parser = argparse.ArgumentParser(description='Script to perform benchmark on LINCS datasets')
    parser.add_argument('--model', type=str, default='biaae')
    parser.add_argument('--gpu', type=int, default=-1)
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
    model_checkpoint = ModelCheckpoint(dirpath='../saved_models', filename=f'reverse_lincs_rnn_{args.model}' + '-{epoch:03d}', every_n_epochs=10, save_top_k=-1)
    logger = TensorBoardLogger(save_dir='../logs/', name='reverse_lincs_rnn_' + args.model)
    trainer_kwargs = dict(logger=logger, callbacks=[model_checkpoint], enable_progress_bar=False, max_epochs=400)

    if args.gpu < 0:
        tr = pl.Trainer(accelerator='cpu', devices=1, **trainer_kwargs)
    else:
        tr = pl.Trainer(accelerator='gpu', devices=[args.gpu], **trainer_kwargs)

    tr.fit(model)
    torch.save(model.state_dict(), f'../saved_models/reverse_lincs_{args.model}.ckpt')
