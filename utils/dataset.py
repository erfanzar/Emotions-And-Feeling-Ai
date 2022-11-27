import os

import torch
# import numba as nb
from torch.utils.data import Dataset, DataLoader

from .utils import read_txt, Cp, str_to_list


class DatasetTorch(Dataset):
    def __init__(self, x: list, y: list, max_size: int, max: int):
        super(DatasetTorch, self).__init__()
        self.x, self.y, self.max_size, self.max = x, y, max_size, max
        print(f' self.x : {len(self.x)}')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        xs, ys = torch.zeros((1, self.max_size)), torch.zeros((1, self.max_size))
        y = torch.tensor(self.y[item] if len(torch.tensor(self.y[item]).shape) != 0 else [self.y[item]])
        x = torch.tensor(self.x[item])
        xs[0][0:x.shape[0]] = x
        ys[0][0:y.shape[0]] = y

        return [xs, ys]


class DataLoaderTorch:
    def __init__(self, data_path: [str, os.PathLike], batch_size: int = 6, num_workers: int = os.cpu_count() // 2,
                 max_size: int = 100,
                 sep: str = ';'):
        super(DataLoaderTorch, self).__init__()
        self.y = None
        self.x = None
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.sep = sep
        self.class_names = {}
        self.vocab_words = {}
        self.start()

    def start(self):
        data = read_txt(self.data_path)
        x, y = [], []
        x_txt = [v[:v.index(';')] for v in data]
        y_txt = [v[v.index(';') + 1:] for v in data]
        data = [d.replace(self.sep, '').lower() for d in data]
        va = 0

        for i, a in enumerate(data):
            va += 1
            if a not in self.vocab_words:
                aa = str_to_list(a)
                for v in aa:
                    self.vocab_words[v] = len(self.vocab_words)
            print(f'\r{Cp.CYAN}Creating Word Vocab {(i / len(data)) * 100:.1f}', end='')
        print(f'\n{Cp.GREEN}Total Words : {len(self.vocab_words)}')
        for i in y_txt:
            if i not in self.class_names:
                self.class_names[i] = len(self.class_names)

        print(f'{Cp.BLUE}Start Converting Data To Numerical Data {va} Texts are loaded')
        k = 0

        for d in x_txt:
            f = []
            aa = str_to_list(d)
            for a in aa:
                if a not in self.vocab_words:
                    self.vocab_words[a] = len(self.vocab_words)
                    k += 1
                    print(f'\r{Cp.RED}UnRead Words : {k}', end='')
                f.append(self.vocab_words[a])
            x.append(f)
        print()
        s = 0
        for a in y_txt:
            f = []
            if a not in self.class_names:
                self.class_names[a] = len(self.class_names)
                s += 1
                print(f'\r{Cp.RED}UnRead Classes : {s}', end='')
            y.append(self.class_names[a])
        print()

        self.x, self.y = x, y

    def return_data_loader(self):
        return DataLoader(DatasetTorch(self.x, self.y, max_size=self.max_size, max=len(self.vocab_words)),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          )
