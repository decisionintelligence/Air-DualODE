import sys
import yaml
import os

gpu_list = "0,1,2,3,4,5,6,7"  # GPU lst
device_map = {gpu: i for i, gpu in enumerate(gpu_list.split(','))}
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
sys.path.append('../../Air-DualODE')

import argparse
import torch
import random
from utils.utils import parsing_syntax, ConfigDict, load_config, update_config, fix_seed
from exp.exp_air import Exp_Air_Pollution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Air-DualODE')

    parser.add_argument('--config_filename', type=str, default='../Model_Config/basic_config.yaml', help='Configuration yaml file')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--des', type=str, help="description of experiment.")
    args, unknown = parser.parse_known_args()
    unknown = parsing_syntax(unknown)

    config = load_config(args.config_filename)
    config = ConfigDict(config)
    config = update_config(config, unknown)
    for attr, value in config.items():
        setattr(args, attr, value)

    # random seed
    fix_seed(args.random_seed)

    args.GPU.use_gpu = True if torch.cuda.is_available() and args.GPU.use_gpu else False

    if args.GPU.use_gpu and not args.GPU.use_multi_gpu:
        try:
            args.GPU.gpu = device_map[str(args.GPU.gpu)]
        except KeyError:
            raise KeyError("This GPU isn't available.")

    if args.GPU.use_gpu and args.GPU.use_multi_gpu:
        args.GPU.devices = args.GPU.devices.replace(' ', '')
        device_ids = args.GPU.devices.split(',')
        args.GPU.device_ids = [int(id_) for id_ in device_ids]
        args.GPU.gpu = args.GPU.device_ids[0]

    rmse_list, mae_list, mape_list = [], [], []
    for exp_idx in range(args.itr):
        args.exp_idx = exp_idx
        if args.to_stdout:
            print('\nNo%d experiment ~~~' % exp_idx)

        exp = Exp_Air_Pollution(args)
        exp.train()
        torch.cuda.empty_cache()
