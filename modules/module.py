import os

from torch import nn as nn


class Module(nn.Module):
    def __init__(self, cfg_path: [str, os.PathLike]):
        super(Module, self).__init__()
        cfg = read_yaml(cfg_path)
        self.model, self.save = pars_model(cfg)

    def forward(self, x):
        route = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = route[m.f] if not isinstance(m.f, list) else [route[r] for r in m.f]
            x = m(x)
            route.append(x if i in self.save else None)
        return x
