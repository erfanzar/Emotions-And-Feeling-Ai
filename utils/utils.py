import os
import sys
import torch
import numba as nb
import numpy as np


class Cp:
    Type = 1
    BLACK = f'\033[{Type};30m'
    RED = f'\033[{Type};31m'
    GREEN = f'\033[{Type};32m'
    YELLOW = f'\033[{Type};33m'
    BLUE = f'\033[{Type};34m'
    MAGENTA = f'\033[{Type};35m'
    CYAN = f'\033[{Type};36m'
    WHITE = f'\033[{Type};1m'
    RESET = f"\033[{Type};39m"


def read_txt(path: [str, os.PathLike] = None):
    data = []
    with open(path, 'r') as r:
        data.append([v for v in r.readlines()])
    data = [d.replace('\n', '') for d in data[0]]
    return data


def str_to_list(string: str = ''):
    string += ' '
    cp = [i for i, v in enumerate(string) if v == ' ']
    word = [string[0:idx] if i == 0 else string[cp[i - 1]:idx] for i, idx in enumerate(cp)]
    word = [w.replace(' ', '') for w in word]
    return word
