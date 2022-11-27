import os

import torch.cuda
from torch import nn as nn

from utils.utils import read_yaml, pars_model


class Module(nn.Module):
    def __init__(self, cfg_path: [str, os.PathLike], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(Module, self).__init__()
        cfg = read_yaml(cfg_path)
        self.model, self.save = pars_model(cfg['model'], device=device)

    def forward(self, x):
        route = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = route[m.f] if not isinstance(m.f, list) else [route[r] for r in m.f]
            x = m(x)
            route.append(x if i in self.save else None)
        return x
