import torch

from utils.utils import accuracy

if __name__ == '__main__':
    x = torch.zeros((64, 3))
    total, true = 0, 0
    y = torch.zeros((64, 3))
    y1 = torch.ones((64, 3))
    acc, total, true = accuracy(x, y, total=total, true=true)
    print(acc, total, true)
    acc, total, true = accuracy(x, y1, total=total, true=true)
    print(acc, total, true)
