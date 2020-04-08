import pytorch_lightning as pl

import torch
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms

from pytorch_lightning.logging import TestTubeLogger

import numpy as np
import argparse

class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        dropout2d = 0.2
        dropout1d = 0.2
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout2d(dropout2d),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self.tail = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        z = self.conv(x).reshape(x.shape[0], -1)
        return self.tail(z)

    def predict(self, x):
        return self.forward(x).max(dim=-1)[1]
    
    def get_logits(self, x):
        return torch.log_softmax(self.forward(x), dim=-1)

    def training_step(self, batch, batch_nb):
        x, y = batch

        logits = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)

        accuracy = (y == logits.max(dim=-1)[1]).float().mean()

        return {'loss': loss.view(1),
                'log':
                    {
                        'acc': accuracy.view(1)
                    }
                }

    def validation_step(self, batch, batch_nb):
        x, y = batch

        accuracy = (y == self.predict(x)).float().mean()

        return {'val_acc': accuracy}

    def validation_end(self, outputs):
        val_stats = {}

        val_stats['val_acc'] = torch.stack([x['val_acc'] for x in outputs]).mean()

        return {'val_loss': val_stats['val_acc'], 'log': val_stats}

    @pl.data_loader
    def train_dataloader(self):
        data = np.load('../data/noisy_mnist.npz')
        dataset = TensorDataset(
            torch.from_numpy(data['train'][0].astype('float32')).
                          reshape(-1, 1, 28, 28),
            torch.from_numpy(data['train_digit'].astype('int')).
                          reshape(-1))

        dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                                    drop_last=False)
        
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        data = np.load('../data/noisy_mnist.npz')
        dataset = TensorDataset(
            torch.from_numpy(data['valid'][0].astype('float32')).
                          reshape(-1, 1, 28, 28),
            torch.from_numpy(data['valid_digit'].astype('int')).
                          reshape(-1))

        dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                                    drop_last=False)
        
        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


if __name__ == '__main__':
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Script to pretrain '
                                                 'noisy mnist classifier')
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    model = MNISTClassifier()

    logger = TestTubeLogger(save_dir='../logs', name='mnist_clf')

    tr = pl.Trainer(gpus=([] if (args.gpu < 0) else [args.gpu]),
                    logger=logger,
                    early_stop_callback=False,
                    show_progress_bar=False,
                    max_nb_epochs=100)

    tr.fit(model)
    torch.save(model.state_dict(), '../saved_models/mnist_clf.ckpt')
