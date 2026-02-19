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
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from torch.utils.data import DataLoader

sys.path.append('..')

from dataloader import LincsDataSet, LincsSampler
from models import BiAAE, CVAE, JMVAE, Lat_SAAE, SAAE, UniAAE, VCCA, VIB
from networks import FCDiscriminator, FinetunedEncoder, RNNEncoder

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def make_conditional_experiment(model):
    class CondGenExperiment(model):
        def __init__(self):
            super().__init__(dataset='lincs_rnn')
            rnn = RNNEncoder(out_dim=88)
            rnn.load_state_dict(torch.load('../saved_models/rnn_enc.ckpt', map_location='cpu'))
            self.mine_enc = FinetunedEncoder(rnn, out_dim=self.z_dim)
            self.mine_fc = FCDiscriminator(in_dim=2 * self.z_dim)

        def mi_computation(self, batch, z):
            x, y = batch
            z_shuffled = z[np.random.permutation(z.shape[0])]
            x_lat = self.mine_enc(x)
            t = self.mine_fc(torch.cat((x_lat, z), dim=-1))
            et = torch.exp(self.mine_fc(torch.cat((x_lat, z_shuffled), dim=-1)))
            return t.mean() - torch.log(et.mean())

        def training_step(self, batch, batch_nb, *args):
            stats = super().training_step(batch, batch_nb, *args)
            if len(args) == 0:
                z = self.get_latents((batch[0], batch[1].detach())).detach()
                mi = self.mi_computation(batch, z)
                stats['loss'] += -mi
                stats['log']['mi[xz|y]'] = mi
            elif args[0] == 2:
                z = self.get_latents((batch[0], batch[1].detach())).detach()
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
            z = self.get_latents(batch)
            stats['mi[xz|y]'] = self.mi_computation(batch, z)
            x, y = batch
            stats['x_sam'] = self.sample(y)
            stats['x'] = x
            return stats

        def validation_epoch_end(self, outputs):
            self.log('val_mi[xz|y]', torch.stack([x['mi[xz|y]'] for x in outputs]).mean(), prog_bar=False)
            fig = plt.figure(num=0, figsize=(10, 4), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.imshow(
                Draw.MolsToGridImage(
                    [Chem.MolFromSmiles(s) for s in outputs[0]['x_sam'][:5]]
                    + [Chem.MolFromSmiles(s) for s in outputs[0]['x'][:5]],
                    molsPerRow=5,
                )
            )
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            fig_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3) / 255.0
            fig_array = fig_array.transpose(2, 0, 1)
            self.logger.experiment.add_image('samples', fig_array, self.current_epoch)

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
    model_checkpoint = ModelCheckpoint(dirpath='../saved_models', filename=f'lincs_rnn_{args.model}' + '-{epoch:03d}', every_n_epochs=10, save_top_k=-1)
    logger = TensorBoardLogger(save_dir='../logs/', name='lincs_rnn_' + args.model)
    trainer_kwargs = dict(logger=logger, callbacks=[model_checkpoint], enable_progress_bar=False, max_epochs=400)

    if args.gpu < 0:
        tr = pl.Trainer(accelerator='cpu', devices=1, **trainer_kwargs)
    else:
        tr = pl.Trainer(accelerator='gpu', devices=[args.gpu], **trainer_kwargs)

    tr.fit(model)
    torch.save(model.state_dict(), f'../saved_models/lincs_rnn_{args.model}.ckpt')
