import torch

from utils.utils import accuracy
from modules.module import Module
from utils.dataset import DataLoaderTorch
batch = 3
if __name__ == '__main__':
    data_val = DataLoaderTorch(data_path='data/val.txt', batch_size=batch, num_workers=2).return_data_loader()

    for x,y in data_val:
        print(x.shape)