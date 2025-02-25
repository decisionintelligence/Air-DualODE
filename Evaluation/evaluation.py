from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.utils import get_logger, load_graph_data
from Data_Provider.data_factory import data_provider
from utils.metrics import *
from utils.tools import count_parameters
import csv

warnings.filterwarnings('ignore')


class Evaluation_Air_Pollution(Exp_Basic):
    def __init__(self, args):
        adj_mx, edge_index, edge_attr, node_attr = load_graph_data(args.data.root_path)
        args.adj_mx = adj_mx    # N x N
        args.edge_index = edge_index    # 2 x M
        args.edge_attr = edge_attr      # M x D
        args.node_attr = node_attr      # N x D

        self._logger = get_logger(None, args.model_name, 'info.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        if args.data.embed:
            args.model.input_dim = int(args.model.input_dim) + int(args.model.embed_dim)

        super(Evaluation_Air_Pollution, self).__init__(args)

        self.num_nodes = adj_mx.shape[0]
        self.input_var = int(self.args.model.input_dim)
        self.input_dim = int(self.args.model.X_dim)
        self.seq_len = int(self.args.model.seq_len)
        self.horizon = int(self.args.model.horizon)
        self.output_dim = int(self.args.model.X_dim)

        self.report_filepath = self.args.report_filepath
        self.result = []
        self.result.append([self.model.setting])
        self.result.append([self.model_parameters])

    def _build_model(self):
        dataset, _ = self._get_data('val')
        self.args.data.mean_ = dataset.scaler.mean_
        self.args.data.std_ = dataset.scaler.scale_
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        self.model_parameters = count_parameters(model)
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        self.inverse_transform = test_data.inverse_transform
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.model.setting, 'checkpoint.pth')))

        with torch.no_grad():
            self.model.eval()

            truths = []
            preds = []

            for _, (x, gt) in enumerate(test_loader):
                x, gt, y_embed = self._prepare_data(x, gt)
                pred, fe = self.model(x, y_embed)

                truths.append(gt.cpu().permute(1, 0, 2))   # B x T x N
                preds.append(pred.cpu().permute(1, 0, 2))

            truths = torch.cat(truths, dim=0)   # B x T x N
            preds = torch.cat(preds, dim=0)

            all_mae = []
            all_smape = []
            all_rmse = []

            assert self.horizon == 24
            for i in range(0, self.horizon, 8):
                pred = preds[:, i: i + 8]
                truth = truths[:, i: i + 8]
                mae, smape, rmse = self._compute_loss_eval(truth, pred)
                all_mae.append(mae)
                all_smape.append(smape)
                all_rmse.append(rmse)
                self._logger.info('Evaluation {}h-{}h: - mae - {:.4f} - rmse - {:.4f} - mape - {:.4f}'.format(
                    i*3, (i+8)*3, mae, rmse, smape))

            # three days
            mae, smape, rmse = self._compute_loss_eval(truths, preds)
            all_mae.append(mae)
            all_smape.append(smape)
            all_rmse.append(rmse)
            self._logger.info('Evaluation all: - mae - {:.4f} - rmse - {:.4f} - mape - {:.4f}'.format(
                mae, rmse, smape))

            all_metrics = {'mae': all_mae, 'rmse': all_rmse, 'smape': all_smape}

            test_res = list(np.array([v for k, v in all_metrics.items()]).T.flatten())
            self.result.append(list(map(lambda x: round(x, 4), test_res)))

            truths_scaled = self.inverse_transform(truths).numpy()
            preds_scaled = self.inverse_transform(preds).numpy()

            return all_mae, all_smape, all_rmse, preds_scaled, truths_scaled

    def vali(self):
        vali_data, vali_loader = self._get_data(flag='val')
        self.inverse_transform = vali_data.inverse_transform
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.model.setting, 'checkpoint.pth')))

        with torch.no_grad():
            self.model.eval()

            truths = []
            preds = []
            for i, (x, gt) in enumerate(vali_loader):
                x, gt, y_embed = self._prepare_data(x, gt)

                pred, fe = self.model(x, y_embed)
                truths.append(gt.cpu().permute(1, 0, 2))    # B x T x N
                preds.append(pred.cpu().permute(1, 0, 2))

            truths = torch.cat(truths, dim=0)
            preds = torch.cat(preds, dim=0)

            mae, smape, rmse = self._compute_loss_eval(truths, preds)

            self._logger.info('Evaluation: - mae - {:.4f} - smape - {:.4f} - rmse - {:.4f}'
                              .format(mae, smape, rmse))
            val_res = [mae, rmse, smape]
            self.result.append(list(map(lambda x: round(x, 4), val_res)))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)  # B x 24(72 hours) x N x D
        x, y, y_embed = self._get_x_y_in_correct_dims(x, y)  # 24 x B x N x D
        return x.to(self.device), y.to(self.device), y_embed  # 24 x B x 35 * 11

    def _get_x_y(self, x, y):
        x = x.float()
        y = y.float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        batch_size = x.size(1)
        if self.args.data.embed:
            station_x = torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(self.seq_len, batch_size, 1, 1)
            station_y = torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(self.horizon, batch_size, 1, 1)
            x = torch.cat([x, station_x], dim=-1)
            y = torch.cat([y, station_y], dim=-1)
            x = x.reshape(self.seq_len, batch_size, self.num_nodes * self.input_var)
            embed = [6, 7, 8, 9, 10, 11]
            y_embed = y[..., embed].reshape(self.horizon, batch_size, self.num_nodes*len(embed))
            y = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                              self.num_nodes*self.output_dim)
        else:
            x = x[..., :self.input_var].reshape(self.seq_len, batch_size, self.num_nodes * self.input_var)
            y = y[..., :self.output_dim].reshape(self.horizon, batch_size,
                                                 self.num_nodes * self.output_dim)
            y_embed = None
        return x, y, y_embed

    def _compute_loss_eval(self, y_true, y_predicted):
        y_true = self.inverse_transform(y_true)
        y_predicted = self.inverse_transform(y_predicted)
        return compute_all_metrics(y_predicted, y_true)