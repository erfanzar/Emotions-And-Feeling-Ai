import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, c1: int, c2: int, act: [str, torch.nn] = None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(c1, c2)

        self.act = eval(f"{act}" if f"{act}".startswith('nn') else f'nn.{act}') if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.linear(x))


class LazyLinear(nn.Module):
    def __init__(self, c1: int, act: [str, torch.nn] = None):
        super(LazyLinear, self).__init__()
        self.linear = nn.LazyLinear(out_features=c1)
        self.act = eval(act) if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.linear(x))


class BN(nn.Module):
    def __init__(self, nm: int, d: int = 1):
        super(BN, self).__init__()
        self.nn = eval(f"nn.BatchNorm{d}d({nm})")

    def forward(self, x):
        return self.nn(x)


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, auto_batch: bool = True,
                 last_peace: bool = True):
        super(LSTM, self).__init__()
        self.last_peace = last_peace
        self.hidden_size = hidden_size
        self.auto_batch = auto_batch
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size=self.hidden_size, input_size=self.input_size, num_layers=self.num_layers,
                            batch_first=False)

    def forward(self, x):
        if len(x.shape) == 2 and self.auto_batch:
            x = torch.unsqueeze(x, 0)
            x = x.permute(1, 0, 2)
        if len(x.shape) == 3 and self.auto_batch:
            x = x.permute(1, 0, 2)
        shape = (self.num_layers, x.shape[1], self.hidden_size)
        h0, c0 = torch.zeros(shape).to(x.device), torch.zeros(shape).to(x.device)
        out, (final_hidden, final_cell) = self.lstm(x, (h0, c0))

        return final_cell[-1] if self.last_peace else out


class LSTMCell(nn.Module):
    def __init__(self, in_dims: int, out_dim: int, activation: [str, nn] = 'nn.ReLU'):
        super(LSTMCell, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.cell = nn.LSTMCell(input_size=self.in_dims, hidden_size=self.out_dim)
        self.act = eval(f"{activation}()") if activation is not None else None

    def forward(self, x):
        shape: tuple = (x.shape[0], self.out_dim)
        h0, c0 = torch.zeros(shape), torch.zeros(shape)
        hidden, cell = self.cell(x, (h0, c0))
        return self.act(hidden) if self.act is not None else hidden
