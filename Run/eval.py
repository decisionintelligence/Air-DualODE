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
import numpy as np
from utils.utils import parsing_syntax, ConfigDict, load_config, update_config, fix_seed
from Evaluation.evaluation import Evaluation_Air_Pollution


def get_mean_std(data_list):
    return data_list.mean(), data_list.std()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Air-DualODE')

    parser.add_argument('--config_filename', type=str, default='../Model_Config/basic_config.yaml', help='Configuration yaml file')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--random_seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--des', type=str, help="description of experiment.")
    parser.add_argument("--report_filepath", type=str, default=None, help="evaluation report output")
    parser.add_argument("--save_results", type=bool, default=False, help="whether to save results")
    parser.add_argument("--save_plots", type=bool, default=False, help="whether to save plots")
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
        print('\nNo%d experiment ~~~' % exp_idx)

        exp = Evaluation_Air_Pollution(args)
        exp.vali()

        # 测试评估
        mae, mape, rmse, preds, truths = exp.test()
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    mae_list = np.array(mae_list)  # num_exp x num_seq
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    seq_len = [(0, 8), (8, 16), (16, 24)]  # seq_len * 3小时（3小时一个点）
    output_text = ''
    output_text += '--------- Air-DualODE Final Results ------------\n'
    for i, (start, end) in enumerate(seq_len):
        output_text += 'Evaluation seq {}h-{}h:\n'.format(start, end)
        output_text += 'MAE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mae_list[:, i])[0],
                                                                 get_mean_std(mae_list[:, i])[1])
        output_text += 'MAPE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mape_list[:, i])[0],
                                                                  get_mean_std(mape_list[:, i])[1])
        output_text += 'RMSE | mean: {:.4f} std: {:.4f}\n\n'.format(get_mean_std(rmse_list[:, i])[0],
                                                                    get_mean_std(rmse_list[:, i])[1])

    # Write the output text to a file
    with open('logs/air-dualode_results.txt', 'a') as file:
        file.write(output_text)