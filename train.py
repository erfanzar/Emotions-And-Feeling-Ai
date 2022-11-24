import math

import torch.nn

from modules.module import Module
from utils.dataset import DataLoaderTorch


def train(epochs: int = 300, batch: int = 64):
    data_train = DataLoaderTorch(data_path='data/train.txt', batch_size=batch, num_workers=2).return_data_loader()
    data_val = DataLoaderTorch(data_path='data/val.txt', batch_size=batch, num_workers=2).return_data_loader()
    data_test = DataLoaderTorch(data_path='data/test.txt', batch_size=batch, num_workers=2).return_data_loader()
    train_bt = math.ceil(data_train.__len__() / batch)
    eval_bt = math.ceil(data_val.__len__() / batch)
    test_bt = math.ceil(data_test.__len__() / batch)
    model = Module('configs/cfg.yaml')
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    for epoch in range(epochs):
        data = iter(data_train)
        for b in range(train_bt):
            x, y = data.__next__()
            x = model(x)
            print(x)


if __name__ == "__main__":
    train()
