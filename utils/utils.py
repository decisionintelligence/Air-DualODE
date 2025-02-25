import yaml
import os
import logging
import sys
import torch
import numpy as np
import torch.nn as nn
import random


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parsing_syntax(unknown):
    unknown_dict = {}
    key = None
    for arg in unknown:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            unknown_dict[key] = None
        else:
            if key:
                unknown_dict[key] = arg
                key = None
    return unknown_dict


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            if key == 'data' and isinstance(value, str):
                dataset_config = load_config("../Model_Config/dataset_config/{}".format(value + ".yaml"))
                self[key]= ConfigDict(dataset_config)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


def update_config(config, unknown_args):
    for key, value in unknown_args.items():
        config_path = key.split('-')
        cur = config
        for node in config_path:
            assert node in cur.keys(), "path not exist"
            if isinstance(cur[node], ConfigDict):
                cur = cur[node]
            else:
                try:
                    cur[node] = eval(value)
                except NameError:
                    cur[node] = value
    return config


def load_graph_data(dataset_path):
    npz_path = os.path.join(dataset_path, 'graph_data.npz')
    data = np.load(npz_path)

    adj_mx = data['adj_mx']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']   # {diff_dist, dist_km, direction}
    node_attr = data['node_attr']

    return adj_mx, edge_index.T, edge_attr, node_attr


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO, to_stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add console handler.
    if to_stdout:
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    # Add file handler and stdout handler
    if log_dir:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Log directory: %s', log_dir)
    return logger


def init_network_weights(net, std=0.1):
    """
    Just for nn.Linear net.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    res = data[..., :last_dim], data[..., last_dim:]
    return res


def exchange_df_column(df, col1, col2):
    """
    exchange df column
    :return new_df
    """
    assert (col1 in df.columns) and (col2 in df.columns)
    df[col1], df[col2] = df[col2].copy(), df[col1].copy()
    df = df.rename(columns={col1: 'temp', col2: col1})
    df = df.rename(columns={'temp': col2})
    return df