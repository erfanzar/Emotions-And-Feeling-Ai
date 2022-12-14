import os

import torch
# import numba as nb
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .utils import read_txt, Cp, str_to_list
import time

class DatasetTorch(Dataset):
    def __init__(self, x: list, y: list, max_size: int, max: int):
        super(DatasetTorch, self).__init__()
        self.x, self.y, self.max_size, self.max = x, y, max_size, max
        print(f' self.x : {len(self.x)}')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        
        y = torch.tensor(self.y[item] if len(torch.tensor(self.y[item]).shape) != 0 else [self.y[item]])
        x = torch.from_numpy(np.array(self.x[item],dtype=np.float32))
        # print(x.dtype)

        return [x, y]

class DataLoaderTorch:
    def __init__(self, data_path: [str, os.PathLike], batch_size: int = 6, num_workers: int = os.cpu_count() // 2,
                 max_size: int = 5900,
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
        vs = 0
        for i, a in enumerate(data):
            va += 1
            aa = str_to_list(a)
            for v in aa:
                if v not in self.vocab_words:
                    self.vocab_words[f"{v}"] = len(self.vocab_words) 
            print(f'\r{Cp.CYAN}Creating Word Vocab {(i / len(data)) * 100:.1f}', end='')
        print(f'\n{Cp.GREEN}Total Words : {len(self.vocab_words)}')
        for i in y_txt:
            if i not in self.class_names:
                self.class_names[i] = len(self.class_names)
        # ask = [[i,v] for i,v in self.vocab_words.items()]

        # for i in range(400):
            # for ak in ask:
                # if ak[1] == i:
                    # print(ak)
        print(f'{Cp.BLUE}Start Converting Data To Numerical Data {va} Texts are loaded')
        k = 0
        # print(self.vocab_words)
        for d in x_txt:

            aa = str_to_list(d)
            npa = np.zeros((1,self.max_size))
            for a in aa:
                try:
                    npa[0,self.vocab_words[a]] = 1
                except:
                    pass

            x.append(npa)
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
