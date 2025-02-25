from exp.exp_basic import Exp_Basic
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.utils import get_logger, load_graph_data
from Data_Provider.data_factory import data_provider
from utils.metrics import *
from utils.tools import EarlyStopping, count_parameters

warnings.filterwarnings('ignore')

class Exp_Air_Pollution(Exp_Basic):
    def __init__(self, args):
        adj_mx, edge_index, edge_attr, node_attr = load_graph_data(args.data.root_path)
        args.adj_mx = adj_mx    # N x N
        args.edge_index = edge_index    # adjacent list: 2 x M
        args.edge_attr = edge_attr      # M x D
        args.node_attr = node_attr      # N x D

        if args.to_log_file:
            self._log_dir = self._get_log_dir(args)
        else:
            self._log_dir = None
        self._logger = get_logger(self._log_dir, args.model_name, 'info.log',
                                  level=args.log_level, to_stdout=args.to_stdout)
        args.logger = self._logger

        if args.data.embed:
            args.model.input_dim = int(args.model.input_dim) + int(args.model.embed_dim)

        super(Exp_Air_Pollution, self).__init__(args)

        self.num_nodes = adj_mx.shape[0]
        self.input_var = int(self.args.model.input_dim)
        self.input_dim = int(self.args.model.X_dim)
        self.seq_len = int(self.args.model.seq_len)
        self.horizon = int(self.args.model.horizon)
        self.output_dim = int(self.args.model.X_dim)

    def _build_model(self):
        dataset, _ = self._get_data('val')
        self.args.data.mean_ = dataset.scaler.mean_
        self.args.data.std_ = dataset.scaler.scale_
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        self._logger.info("Model created")
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(model))
        )
        if self.args.GPU.use_multi_gpu and self.args.GPU.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.GPU.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.train.lr, eps=1e-8)
        return model_optim

    def _select_criterion(self):
        if self.args.model.loss.criterion == "mse":
            criterion = nn.MSELoss()
        elif self.args.model.loss.criterion == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def _select_lr_scheduler(self, optimizer, train_loader):
        if self.args.train.lradj == 'MultiStep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.train.steps,
                                                                gamma=self.args.train.lr_decay_ratio)
        elif self.args.train.lradj == 'TST':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                               steps_per_epoch=len(train_loader),
                                                               pct_start=self.args.train.pct_start,
                                                               epochs=self.args.train.epochs,
                                                               max_lr=self.args.train.lr)
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return lr_scheduler

    def vali(self, vali_data, vali_loader, epoch_num, save=False):
        with torch.no_grad():
            self.model.eval()

            preds = []
            truths = []
            for i, (x, gt) in enumerate(vali_loader):
                x, gt, y_embed = self._prepare_data(x, gt)
                pred, fe = self.model(x, y_embed)

                truths.append(gt.cpu())    # T x B x N
                preds.append(pred.cpu())

                if self.args.model_name in ["AirPhyNet", "Origin_AirPhyNet"]:
                    loss = self.pred_loss(gt, pred)
                else:
                    loss = self.get_loss(x, gt)

                if self.TB_logger:
                    self.TB_logger.add_scalar("val/loss", loss, epoch_num * len(vali_loader) + i)

            truths = torch.cat(truths, dim=1)
            preds = torch.cat(preds, dim=1)   # T x B x N
            val_loss = self.criterion(truths, preds)

            truths = truths.permute(1, 0, 2)
            preds = preds.permute(1, 0, 2)   # B x T x N
            mae, smape, rmse = self._compute_loss_eval(truths, preds)

            self._logger.info('Evaluation: - mae - {:.4f} - smape - {:.4f} - rmse - {:.4f}'
                              .format(mae, smape, rmse))

            return val_loss

    def train(self):
        if self.args.TB_dir:
            self.TB_logger = self._build_TB_logger(self.model.setting)
        else:
            self.TB_logger = None
        self._logger.info('Model mode: train')
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        self.inverse_transform = train_data.inverse_transform
        self.criterion = self._select_criterion()

        model_save_path = os.path.join(self.args.checkpoints, self.model.setting)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        optimizer = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.train.patience, verbose=True, logger=self._logger)
        lr_scheduler = self._select_lr_scheduler(optimizer, train_loader)

        time_now = time.time()
        train_steps = len(train_loader)

        self._logger.info('Start training ...')
        num_batches = self.args.data.batch_size
        self._logger.info("num_batches: {}".format(num_batches))

        for epoch_num in range(1, self.args.train.epochs + 1):
            if self.args.to_stdout:
                print('\nTrain epoch %s:' % (epoch_num))
            self.model.train()

            losses = []
            iter_count = 0
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x, batch_y, y_embed = self._prepare_data(batch_x, batch_y)
                output, fe = self.model(batch_x, y_embed)

                if self.args.model_name in ["AirPhyNet", "Origin_AirPhyNet"]:
                    loss = self.pred_loss(batch_y, output)
                else:
                    loss = self.get_loss(batch_x, batch_y)
                self._logger.debug("FE: number - {}, time - {:.3f} s, err - {:.3f}".format(*fe, loss.item()))

                self._logger.debug(loss.item())
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if self.TB_logger:
                    self.TB_logger.add_scalar("train/loss", loss.item(), epoch_num * train_steps + i)

                if self.args.train.lradj == 'TST':
                    lr_scheduler.step()

                del output, loss, batch_x, batch_y
                torch.cuda.empty_cache()

            val_loss = self.vali(vali_data, vali_loader, epoch_num)

            if (epoch_num % self.args.train.log_every) == self.args.train.log_every - 1:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train.epochs - epoch_num) * train_steps - i)
                message = ('Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}'
                           .format(epoch_num, self.args.train.epochs,
                                   np.mean(losses), val_loss, optimizer.param_groups[0]['lr']))
                self._logger.info(message)
                self._logger.info('speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # 学习率动态调整
            if self.args.train.lradj == 'MultiStep':
                lr_scheduler.step()
            elif self.args.train.lradj == 'TST':
                pass
            else:
                lr_scheduler.step(val_loss)

            early_stopping(val_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self._logger.info("---" * 30)

    @staticmethod
    def _get_log_dir(args):
        log_dir = args.train.get('log_dir')
        if log_dir is None:
            run_id = '%s_%s/' % (
                args.model_name, time.strftime('%m-%d-%H-%M-%S'))
            base_dir = args.log_base_dir
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

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

    def pred_loss(self, y_true, y_predicted):
        y_true = self.inverse_transform(y_true)
        y_predicted = self.inverse_transform(y_predicted)
        return masked_loss(y_predicted, y_true, MAE)

    def _compute_loss_eval(self, y_true, y_predicted):
        y_true = self.inverse_transform(y_true)
        y_predicted = self.inverse_transform(y_predicted)
        return compute_all_metrics(y_predicted, y_true)

    def kl_loss(self, mu, logvar):
        # n_traj x B x N x Latent_dim
        var = torch.exp(logvar)
        loss = 1/2 * (var + mu**2 - logvar - 1)
        return torch.mean(loss.sum(dim=(-1, -2)))

    def get_loss(self, x_true, y_true):
        loss = torch.zeros(1).to(self.device)
        x_true = torch.reshape(x_true, (self.seq_len, -1, self.num_nodes, self.input_dim))
        x_true = x_true[:, :, :, 0]
        if self.args.model.loss.kl_loss:
            kl_loss = self.kl_loss(self.model.means_z0, self.model.logvar_z0)
            loss += kl_loss
        if self.args.model.loss.recon_loss:
            recon_loss = self.criterion(x_true, self.model.recon_x)
            loss += self.args.model.loss.recon_coeff * recon_loss
        if self.args.model.loss.pred_loss:
            pred_loss = self.criterion(y_true, self.model.pred_y)
            loss += pred_loss
        if self.args.model.loss.cl_loss:
            loss += self.args.model.loss.cl_coeff * self.model.loss_CL
        return loss