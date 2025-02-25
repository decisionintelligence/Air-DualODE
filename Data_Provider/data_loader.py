import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.utils import exchange_df_column
import random
import chinese_calendar as calendar
from typing import Union, List
import metpy.calc as mpcalc
from metpy.units import units

class Dataset_Beijing1718(Dataset):
    def __init__(self, root_path, flag='train', seq_len=24, pred_len=24,
                 freq='1h', scale=True, embed=0,
                 normalized_col: Union[str, List[str]]='default'):
        self.columns = ["time", "PM2.5", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
        if normalized_col == 'default':
            self.normalized_col = ["PM2.5", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
        else:
            self.normalized_col = normalized_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len
        assert freq in ['1h', '3h']
        self.freq = freq
        self.scale = scale
        self.embed = embed
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.station_df_dict = {}
        self.station_info = pd.read_csv(os.path.join(self.root_path, "station.csv"))
        self.stations_dir = os.path.join(self.root_path, "stations")

        self.valid_indices, self.border = self._get_valid_indices()
        self.__read_data__()

    def _get_valid_indices(self):
        # find all valid index
        random_file = random.choice(os.listdir(self.stations_dir))
        station_data = pd.read_csv(os.path.join(self.stations_dir, random_file),
                                   usecols=self.columns)
        station_data['time'] = pd.to_datetime(station_data['time'])
        station_data.set_index('time', inplace=True)
        # sample frequence
        if self.freq == '3h':
            station_data = station_data[::3]
        # 7:1:2
        border1s = [0, int(len(station_data) * 0.7), int(len(station_data) * 0.8)]
        border2s = [int(len(station_data) * 0.7), int(len(station_data) * 0.8), len(station_data)]
        self.train_border = (border1s[0], border2s[0])
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # select dataset，find all valid index
        station_data = station_data.iloc[border1: border2]
        self.time_idx = station_data.index  # time
        self.time_info = self.cal_time_info(self.time_idx)
        valid_indices = []
        start = 0
        while start + self.window_size <= len(station_data):
            window = station_data.iloc[start: start + self.window_size]
            if not window.isna().any(axis=1).any():
                valid_indices.append(start)
            start += 1
        return valid_indices, (border1, border2)

    def cal_time_info(self, time_idx):
        def check_holiday(date):
            return 1 if calendar.is_holiday(date) or calendar.is_in_lieu(date) else 0

        time_info = pd.DataFrame({
            'time': time_idx,
            'hour_of_day': time_idx.hour,  # hour-day
            'day_of_week': time_idx.dayofweek,  # day-week
            'day_of_month': time_idx.day - 1,  # day-month
            'month_of_year': time_idx.month - 1,  # month-year
        })
        time_info['is_holiday'] = [check_holiday(d.date()) for d in time_idx]

        time_info.set_index('time', inplace=True)
        return time_info

    def __read_data__(self):
        train_set = []
        for csv in os.listdir(self.stations_dir):
            station_df = pd.read_csv(os.path.join(self.stations_dir, csv),
                                     usecols=self.columns)
            station_df = exchange_df_column(station_df, 'wind_direction', 'wind_speed')
            station_df['time'] = pd.to_datetime(station_df['time'])
            station_df = station_df.set_index('time')
            if self.freq == '3h':
                station_df = station_df.iloc[::3]
            train_set.append(station_df[self.normalized_col].iloc[self.train_border[0]: self.train_border[1]])
            station_df = station_df.iloc[self.border[0]: self.border[1]]
            station = csv.split(".")[0]
            self.station_df_dict[station] = station_df

        train_set = pd.concat(train_set, axis=0)
        if self.scale:
            self.scaler.fit(train_set)
            for station, df in self.station_df_dict.items():
                df[self.normalized_col] = self.scaler.transform(df[self.normalized_col])
                if self.embed:
                    self.station_df_dict[station] = pd.concat([df, self.time_info], axis=1)
                else:
                    self.station_df_dict[station] = pd.DataFrame(df, columns=self.columns[1:], index=self.time_idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_x = []
        seq_y = []
        x_start = self.valid_indices[idx]
        x_end = x_start + self.seq_len
        y_start = self.valid_indices[idx] + self.seq_len
        y_end = y_start + self.pred_len

        for station in self.station_info['station']:
            df = self.station_df_dict[station]
            seq_x.append(df.iloc[x_start: x_end].values)
            seq_y.append(df.iloc[y_start: y_end].values)
        seq_x = np.stack(seq_x).transpose(1, 0, 2)
        seq_y = np.stack(seq_y).transpose(1, 0, 2)
        return seq_x, seq_y

    def inverse_transform(self, data):
        assert self.scale is True
        pm25_mean = self.scaler.mean_[0]
        pm25_std = self.scaler.scale_[0]
        return (data * pm25_std) + pm25_mean


class Dataset_KnowAir(Dataset):
    def __init__(self, root_path, flag='train', seq_len=24, pred_len=24,
                 freq='3h', scale=True, embed=0,
                 normalized_col: Union[str, List[int]]='default'):
        if normalized_col == 'default':
            self.normalized_col = np.arange(0, 6)
        else:
            self.normalized_col = normalized_col

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len
        self.scale = scale
        self.embed = embed
        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.station_info = pd.read_csv(os.path.join(self.root_path, "station.csv"))
        self.stations_npy = os.path.join(self.root_path, "KnowAir.npy")
        metero_var = ['100m_u_component_of_wind', '100m_v_component_of_wind', '2m_dewpoint_temperature',
                       '2m_temperature', 'boundary_layer_height', 'k_index', 'relative_humidity+950',
                       'relative_humidity+975', 'specific_humidity+950', 'surface_pressure',
                       'temperature+925', 'temperature+950', 'total_precipitation', 'u_component_of_wind+950',
                       'v_component_of_wind+950', 'vertical_velocity+950', 'vorticity+950']
        metero_use = ['2m_temperature', 'surface_pressure', 'relative_humidity+950',
                      '100m_u_component_of_wind', '100m_v_component_of_wind']
        self.metero_idx = [metero_var.index(var) for var in metero_use]
        self.time_idx = pd.date_range(start='2015-01-01', end='2018-12-31 21:00', freq='3H')

        self.__process_raw_data__()
        self.__read_data__()

    def __process_raw_data__(self):
        raw_data = np.load(self.stations_npy)
        self.pm25 = raw_data[:, :, -1:]
        self.feature = raw_data[:, :, :-1]
        self.feature = self.feature[:, :, self.metero_idx]
        u = self.feature[:, :, -2] * units.meter / units.second   # m/s
        v = self.feature[:, :, -1] * units.meter / units.second   # m/s
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude    # km/h
        direc = mpcalc.wind_direction(u, v)._magnitude
        self.feature[:, :, -2] = speed
        self.feature[:, :, -1] = direc

        self.raw_data = np.concatenate([self.pm25, self.feature], axis=-1)  # T x N x D

    def __read_data__(self):
        # 2:1:1
        border1s = [0, int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75)]
        border2s = [int(len(self.raw_data) * 0.5), int(len(self.raw_data) * 0.75), len(self.raw_data)]
        self.train_border = (border1s[0], border2s[0])
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = self.raw_data[border1: border2]
        if self.embed:
            self.time_info = self.cal_time_info(self.time_idx[border1: border2]).values

        if self.scale:
            train_set = self.raw_data[self.train_border[0]: self.train_border[1], :, :]
            T, N, D = self.data.shape
            self.scaler.fit(train_set.reshape(-1, D)[:, self.normalized_col])
            self.data = self.data.reshape(-1, D)
            self.data[:, self.normalized_col] = self.scaler.transform(self.data[:, self.normalized_col])
            self.data = self.data.reshape(T, N, D)

    def cal_time_info(self, time_idx):
        def check_holiday(date):
            return 1 if calendar.is_holiday(date) or calendar.is_in_lieu(date) else 0

        time_info = pd.DataFrame({
            'time': time_idx,
            'hour_of_day': time_idx.hour,  # hour-day
            'day_of_week': time_idx.dayofweek,  # day-week
            'day_of_month': time_idx.day - 1,  # day-month
            'month_of_year': time_idx.month - 1,  # month-year
        })
        time_info['is_holiday'] = [check_holiday(d.date()) for d in time_idx]

        time_info.set_index('time', inplace=True)
        return time_info

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = x_start + self.seq_len
        y_start = idx + self.seq_len
        y_end = y_start + self.pred_len

        seq_x = self.data[x_start: x_end]
        seq_y = self.data[y_start: y_end]
        if self.embed:
            seq_x_time_info = self.time_info[x_start: x_end]
            seq_x_time_info = np.expand_dims(seq_x_time_info, axis=1).repeat(seq_x.shape[1],axis=1)
            seq_x = np.concatenate([seq_x, seq_x_time_info], axis=2)

            seq_y_time_info = self.time_info[y_start: y_end]
            seq_y_time_info = np.expand_dims(seq_y_time_info, axis=1).repeat(seq_x.shape[1],axis=1)
            seq_y = np.concatenate([seq_y, seq_y_time_info], axis=2)

        return seq_x, seq_y

    def inverse_transform(self, data):
        assert self.scale is True
        pm25_mean = self.scaler.mean_[0]
        pm25_std = self.scaler.scale_[0]
        return (data * pm25_std) + pm25_mean
