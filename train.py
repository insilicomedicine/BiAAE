import os
import sys
import logging

import torch
import torchvision

import shutil
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models import ConvolutionalBiAAE
from tensorboardX import SummaryWriter

import argparse
import json

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared_dim', type=int,
                        help='Shared latent dimension size', default=4)
    parser.add_argument('--exclusive_x_dim', type=int,
                        help='Latent dimensions for exclusive object part',
                        default=12)
    parser.add_argument('--exclusive_y_dim', type=int,
                        help='Latent dimensions for exclusive condition part',
                        default=12)
    parser.add_argument('--dropout1d', type=float, help='Dropout1D rate',
                        default=0.2)
    parser.add_argument('--dropout2d', type=float, help='Dropout2D rate',
                        default=0.2)
    parser.add_argument('--l2', type=float,
                        help='Weight decay on parameters',
                        default=0)

    parser.add_argument('--loss_rec_lambda_x', type=float,
                        help='Decoders loss coefficient', default=10)
    parser.add_argument('--loss_rec_lambda_y', type=float,
                        help='Decoders loss coefficient', default=10)
    parser.add_argument('--loss_shared_lambda', type=float,
                        help='Shared similarity loss coefficient', default=1)
    parser.add_argument('--loss_gen_lambda', type=float,
                        help='Generator (discriminator related) loss',
                        default=1)
    parser.add_argument('--discriminator_steps', type=int,
                        help='Number of discriminator steps', default=3)
    parser.add_argument('--generator_steps', type=int,
                        help='Number of generator steps', default=1)
    parser.add_argument('--learning_rate', type=float, help='Learning rate',
                        default=0.0005)
    parser.add_argument('--epochs', type=int, help='Number of epochs',
                        default=500)
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=64)

    parser.add_argument('--mode', type=str, help='BiAAE|UniAAE',
                        choices=['BiAAE', 'UniAAE'],
                        default='BiAAE')
    parser.add_argument('--data_path', type=str, help='Path to the dataset',
                        default='data/noisy_mnist.npz')
    parser.add_argument('--log_dir', type=str, help='Path to logs directory',
                        default='logs/')
    parser.add_argument('--model_dir', type=str,
                        help='Path to directory for saved models',
                        default='models/')
    parser.add_argument('--name', type=str,
                        help='Name of the experiment for logs and saved model',
                        default='BiAAE')

    parser.add_argument('--device', type=str,
                        help="Torch device, e.g. 'cuda:0'", default='cpu')
    parser.add_argument('--n_jobs', type=int,
                        help="Number of jobs for torch.set_num_threads",
                        default=1)
    parser.add_argument('--seed', type=int, help="Seed for pytorch and numpy",
                        default=0)

    args, unknown = parser.parse_known_args(argv[1:])
    if len(unknown) != 0:
        raise ValueError('Unknown argument {}\n'.format(unknown[0]))
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.model_dir = os.path.join(args.model_dir, args.name)
    for dir_name in [args.log_dir, args.model_dir]:
        if os.path.exists(dir_name):
            logger.warning(f'Deleting existing directory {dir_name}')
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    return args


if __name__ == "__main__":
    args = parse(sys.argv)

    logger.info('Experiment parameters:')
    logger.info(args)

    saved_args = {key: value for key, value in vars(args).items()
                  if key not in ['data_path', 'log_dir',
                                 'model_dir', 'device']}

    with open(os.path.join(args.model_dir, 'args.json'), 'w') as args_file:
        json.dump(saved_args, args_file, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.n_jobs)

    writer = {label: SummaryWriter(os.path.join(args.log_dir, label))
              for label in ['train', 'valid', 'test']}
    model = ConvolutionalBiAAE(args)
    for submodel in model.submodels.values():
        submodel.to(args.device)

    data = np.load(args.data_path)
    dataset = {key: TensorDataset(
        *[torch.from_numpy(data[key][i].astype('float32')).reshape(-1, 1, 28,
                                                                   28).to(
            args.device) for i in range(2)])
        for key in ['train', 'valid', 'test']}
    loader = {key: DataLoader(tensor_dataset, batch_size=args.batch_size,
                              shuffle=key == 'train',
                              drop_last=key == 'train')
              for key, tensor_dataset in dataset.items()}

    lr = args.learning_rate

    for epoch in range(args.epochs):
        logger.info(f'Epoch index {epoch}')

        logger.info('Training...')
        for submodel in model.submodels.values():
            submodel.train()
        with torch.enable_grad():
            for batch in tqdm(loader['train']):
                model.step(batch, backward=True, step=epoch,
                           writer=writer['train'])

        logger.info('Validating...')
        for submodel in model.submodels.values():
            submodel.eval()
        for label in ['valid', 'test']:
            with torch.no_grad():
                for batch in tqdm(loader[label]):
                    model.step(batch, backward=False, step=epoch,
                               writer=writer[label])

        logger.info('Generating a sample for the last test batch conditions')
        y = batch[1]
        x = model.generate(y)  # generate x from y
        writer['test'].add_images(
            'generated',
            torch.stack(
                [torchvision.utils.make_grid(torch.stack([y[i], x[i]]))
                 for i in range(x.shape[0])]).transpose(-1, -2),
            epoch)

        logger.info(f'Saving the model {args.name}')
        state_dict = {'model_arguments': saved_args}
        for name, submodel in model.submodels.items():
            state_dict[name] = submodel.to('cpu').state_dict()
            submodel.to(args.device)
        torch.save(state_dict, os.path.join(args.model_dir, 'model.tar'))
