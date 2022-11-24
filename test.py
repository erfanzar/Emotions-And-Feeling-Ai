from utils.utils import read_yaml, pars_model

if __name__ == "__main__":
    cfg = read_yaml('configs/cfg.yaml')
    model_cfg = cfg['model']

    print(pars_model(model_cfg))
