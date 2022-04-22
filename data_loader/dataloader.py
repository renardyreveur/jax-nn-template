import gzip
import os
import pickle
from pathlib import Path

import numpy as np
import requests

from base import BaseDataLoader


# Sample Dataset class - MNIST
class Dataset:
    mnist_url = "http://yann.lecun.com/exdb/mnist/"
    mnist_files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    def __init__(self, root, test=False):
        # Create Raw data folder to download MNIST files
        if not Path(root, "raw").exists():
            os.mkdir(Path(root, "raw"))
            print(f"Raw data directory created at {Path(root, 'raw')}")

        # If the MNIST files haven't been downloaded, download them
        if not all([Path(root, "raw", x).exists() for x in self.mnist_files]):
            print("Downloading MNIST dataset ...")
            for mf in self.mnist_files:
                with open(Path(root, "raw", mf), 'wb') as f:
                    f.write(requests.get(self.mnist_url+mf).content)
            print("Done!\n")

        # Parse the bytes into a numpy array and save them for quicker access
        if not all(Path(root, x).exists() for x in ["training.npy", "testing.npy"]):
            print("Parsing MNIST dataset ...")
            training, testing = [], []
            for i, mf in enumerate(self.mnist_files):
                with gzip.open(Path(root, "raw",  mf), 'r') as f:
                    if int.from_bytes(f.read(4), byteorder='big') == 2051:
                        num_images = int.from_bytes(f.read(4), byteorder='big')
                        rows = int.from_bytes(f.read(4), byteorder='big')
                        cols = int.from_bytes(f.read(4), byteorder='big')
                        image_array = np.frombuffer(f.read(num_images*rows*cols), dtype=np.uint8)\
                            .reshape((num_images, rows, cols))
                        training.append(image_array) if i == 0 else testing.append(image_array)
                    else:
                        num_labels = int.from_bytes(f.read(4), byteorder='big')
                        label_array = np.frombuffer(f.read(num_labels), dtype=np.uint8).reshape((num_labels,))
                        training.append(label_array) if i == 1 else testing.append(label_array)

            with open(Path(root, "training.npy"), 'wb') as f:
                pickle.dump(training, f)
            with open(Path(root, "testing.npy"), 'wb') as f:
                pickle.dump(testing, f)
            print("Done!\n")

        # Read training/testing data
        with open(Path(root, "training.npy" if not test else "testing.npy"), 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        image, label = self.data[0][idx], self.data[1][idx]
        return image, label


class DataLoader(BaseDataLoader):
    def __init__(self, dataset_args, batch_size, shuffle=True, collate_fn=None):
        dataset = Dataset(**dataset_args)
        super().__init__(dataset, batch_size, shuffle, collate_fn)
