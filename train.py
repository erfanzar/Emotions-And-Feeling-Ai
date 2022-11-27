import math

import torch.nn

from modules.module import Module
from utils.dataset import DataLoaderTorch
from utils.utils import print_train, max_args_to_max_non_tom, Cp, accuracy, fprint


def train(epochs: int = 300, batch: int = 64, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    data_train = DataLoaderTorch(data_path='data/train.txt', batch_size=batch, num_workers=2).return_data_loader()
    data_val = DataLoaderTorch(data_path='data/val.txt', batch_size=batch, num_workers=2).return_data_loader()
    data_test = DataLoaderTorch(data_path='data/test.txt', batch_size=batch, num_workers=2).return_data_loader()
    train_bt = math.ceil(data_train.__len__())
    eval_bt = math.ceil(data_val.__len__())
    test_bt = math.ceil(data_test.__len__())
    fprint(f'Length Train Data {data_train.__len__()}')
    fprint(f'Length Eval Data {data_val.__len__()}')
    fprint(f'Length Test Data {data_test.__len__()}')

    model = Module('configs/cfg.yaml', device=device)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)

    for epoch in range(epochs):
        total_train, trues_train = 0, 0
        total_eva, trues_eval = 0, 0

        model.train()
        # for b in range(train_bt):
        for b, (x, y) in enumerate(data_train):
            optimizer.zero_grad()
            # x, y = data_ta.__next__()
            x, y = x.to(device), y.to(device)
            x = model(x)
            x = x.reshape(x.shape[0], -1)
            y_s = x.shape[-1]
            torch_y = torch.zeros((x.shape[0], y_s))
            torch_y[:, 0:y_s] = y[:, 0, 0:y_s]
            y = max_args_to_max_non_tom(torch_y, device)
            # output = x / x.amax(dim=(0, 1), keepdim=True)
            # print(f'output : {output}')
            # x_f = x > 0.8
            # nx_f = x < 0.8
            # x[x_f] = 1
            # x[nx_f] = 0
            # print(x)
            x, y = x.float(), y.float()
            loss = loss_fn(input=x, target=y)
            # loss *= 0.05
            acc, total_train, trues_train = accuracy(x, y, total=total_train, true=trues_train)
            loss.backward()
            optimizer.step()
            print_train([b + 1, train_bt], [epoch + 1, epochs], loss, accuracy=acc, mode='[Train]')
        print('\n')
        model.eval()
        for b, (x, y) in enumerate(data_val):

            x, y = x.to(device), y.to(device)
            x = model(x)
            x = x.reshape(x.shape[0], -1)
            y_s = x.shape[-1]
            torch_y = torch.zeros((x.shape[0], y_s))
            torch_y[:, 0:y_s] = y[:, 0, 0:y_s]
            y = max_args_to_max_non_tom(torch_y, device)
            x, y = x.float(), y.float()
            loss = loss_fn(input=x, target=y)
            acc, total_eva, trues_eval = accuracy(x, y, total=total_eva, true=trues_eval)
            print_train([b + 1, eval_bt], [epoch + 1, epochs], loss, eval_accuracy=acc, mode='[Eval]', color=Cp.GREEN)
        print('\n')
        if epoch % 20 == 0:
            pack = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(pack, f'model-epoch{epoch}.pt')

if __name__ == "__main__":
    train()
