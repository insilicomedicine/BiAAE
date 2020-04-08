import gzip
import tempfile
import numpy as np
import pickle
import argparse
import sys
import logging
import os
from torch.hub import _download_url_to_file


__VIEW1 = 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz'
__VIEW2 = 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz'
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def read_data(path):
    """
    Loads the data from the gzip pickled files and converts to numpy arrays
    """
    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    train_set_x, train_set_y = (np.asarray(arr) for arr in train_set)
    valid_set_x, valid_set_y = (np.asarray(arr) for arr in valid_set)
    test_set_x, test_set_y = (np.asarray(arr) for arr in test_set)

    return [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]


def download_data(output='data/noisy_mnist.npz'):
    """
    Downloads and extracts Noisy MNIST dataset (~286MB)
    The resulting .npz file contains train, valid, and test
        image pairs along with their labels in arrays train_digit,
        valid_digit, and test_difit
    Parameters:
        output: path to the output .npz file
    """
    with tempfile.NamedTemporaryFile() as view1, \
            tempfile.NamedTemporaryFile() as view2:
        logger.info("[Step 1/3] Downloading images x (rotated)")
        _download_url_to_file(__VIEW1, view1.name, None, True)
        logger.info("[Step 2/3] Downloading images y (rotated and noisy)")
        _download_url_to_file(__VIEW2, view2.name, None, True)
        logger.info("[Step 3/3] Preparing final file")
        img_x = read_data(view1.name)
        img_y = read_data(view2.name)
        train = np.asarray([img_x[0][0], img_y[0][0]])
        valid = np.asarray([img_x[1][0], img_y[1][0]])
        test = np.asarray([img_x[2][0], img_y[2][0]])
        train_digit = img_x[0][1]
        valid_digit = img_x[1][1]
        test_digit = img_x[2][1]
        assert np.all(img_y[0][1] == train_digit)
        assert np.all(img_y[1][1] == valid_digit)
        assert np.all(img_y[2][1] == test_digit)
        np.savez_compressed(output,
                            train=train,
                            valid=valid,
                            test=test,
                            train_digit=train_digit,
                            valid_digit=valid_digit,
                            test_digit=test_digit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/noisy_mnist.npz',
                        help='Path to the dataset (should end with .npz)')

    config, unknown = parser.parse_known_args()
    if any(unknown):
        raise ValueError(f"Unknown arguments {unknown}")
    if not config.output.endswith('.npz'):
        raise ValueError("Output filename should have an extension .npz")

    base_dir = os.path.dirname(config.output)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    download_data(output=config.output)
