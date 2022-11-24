import os

import yaml

from modules.commons import *


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


def fprint(*args, color: str = Cp.CYAN, **kwargs):
    print(*(f"{color}{arg}" for arg in args), **kwargs)


def attar_print(keys: [str, list[str]], values: [str, list[str]], color: str = Cp.CYAN, **kwargs):
    assert len(keys) == len(values), 'Keys And Vals Should Have same size'
    for i, (k, v) in enumerate(zip(keys, values)):
        fprint(f'{k} : {v}', color=color, **kwargs)


def read_yaml(path: [str, os.PathLike] = None):
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


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


def wrd_print(words: list = None, action: str = None):
    for idx, i in enumerate(words):
        name = f'{i=}'.split('=')[0]
        string = f"{name}" + ("." if action is not None else "") + (action if action is not None else "")
        print(
            f'\033[1;36m{idx} : {eval(string)}')


def pars_args(args):
    arg = '()' if len(args) == 0 else ''.join((f'{v},' if i != len(args) - 1 else f'{v}' for i, v in enumerate(args)))

    arg = f'({arg})' if len(args) != 0 else arg

    return arg


def pars_model(cfg: list):
    model = nn.ModuleList()
    index, save = 0, []
    for c in cfg:
        f, n, t, a = c
        args = pars_args(a)
        for i in range(n):
            string: str = f'{t}{args}'

            m = eval(string)
            setattr(m, 'f', f)
            model.append(m)
            if f != -1:
                save.append(index % f)
            index += 1

    return model, save
