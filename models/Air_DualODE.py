from torch import nn
import torch
from models.layers.Explicit_odefunc import ODEFunc
from models.layers.Diffeq_solver import DiffeqSolver
from typing import Optional
from models.layers.tools import Air_Attrs
from models.layers.Encoder import Encoder_unk_z, Coeff_Estimator_new as Coeff_Estimator, Encoder_phy_z
from models.layers.Decoder import Conv1d_Decoder as Conv_Decoder
from models.layers.Dynamics_Funsion import GNN_Knowledge_Fusion
from models.layers.Unk_Dynamics import Unk_odefunc_ATT as Unk_odefunc
from models.layers.Embed import AirEmbedding
from models.layers.soft_losses import *

from utils.utils import ConfigDict, load_config

class Model(nn.Module, Air_Attrs):
    def __init__(self, args: Optional[ConfigDict] = None):
        nn.Module.__init__(self)
        Air_Attrs.__init__(self, args)
        self._logger = args.logger
        self.setting = self.get_setting(args)

        # gpu device
        self.device = torch.device("cuda:{}".format(args.GPU.gpu))

        # embedding
        self.embed_h2d = self.args.model.embedding.hour2day
        self.embed_d2w = self.args.model.embedding.day2week
        self.embed_d2m = self.args.model.embedding.day2month
        self.embed_m2y = self.args.model.embedding.month2year
        self.embed_station = self.args.model.embedding.station
        self.embedding_dim = 0

        if args.data.embed:
            self.embedding_dim = self.embed_h2d + self.embed_d2w + self.embed_d2m + \
                                 self.embed_m2y + self.embed_station + 1   # holiday
            self.embedding_air = AirEmbedding(self.embed_h2d, self.embed_d2w, self.embed_d2m,
                                              self.embed_m2y, self.embed_station, self.num_nodes)

        # phy_func
        self.is_phy = self.args.model.phy_func.enable
        self.knowledge = self.args.model.phy_func.knowledge
        self.phy_gnn_layers = self.args.model.phy_func.gnn_layers
        self.phy_gnn_hid_dim = self.args.model.phy_func.gnn_hid_dim  # GCN: N x D --> N x D'
        self.cheb_k = self.args.model.phy_func.cheb_k
        self.is_estimator = self.args.model.phy_func.coeff_estimator
        self.phy_rnn_layers = self.args.model.phy_func.rnn_layers
        self.phy_rnn_dim =self.args.model.phy_func.rnn_dim
        self.phy_input_dim = self.args.model.phy_func.input_dim + self.embedding_dim
        self.phy_latent_dim = self.args.model.phy_func.latent_dim
        self.phy_atol = float(self.args.model.phy_func.odeint_atol)
        self.phy_rtol = float(self.args.model.phy_func.odeint_rtol)
        self.phy_solver = DiffeqSolver(
            method=self.args.model.phy_func.ode_method,
            odeint_atol=self.phy_atol,
            odeint_rtol=self.phy_rtol,
            adjoint=self.args.model.phy_func.adjoint
        )

        # unk_func
        self.is_unk = self.args.model.unk_func.enable
        self.unk_rnn_layers = self.args.model.unk_func.rnn_layers
        self.unk_rnn_dim =self.args.model.unk_func.rnn_dim
        self.unk_input_dim = self.args.model.unk_func.input_dim + self.embedding_dim
        self.unk_latent_dim = self.args.model.unk_func.latent_dim  # Z^D_t: N x unk_latent_dim
        self.unk_n_heads = self.args.model.unk_func.n_heads        # in NODE ATT
        self.unk_d_f = self.args.model.unk_func.d_f
        self.unk_atol = float(self.args.model.unk_func.odeint_atol)
        self.unk_rtol = float(self.args.model.unk_func.odeint_rtol)
        self.unk_solver = DiffeqSolver(
            method=self.args.model.unk_func.ode_method,
            odeint_atol=self.unk_atol,
            odeint_rtol=self.unk_rtol,
            adjoint=self.args.model.unk_func.adjoint
        )

        # fusion
        self.fusion_latent_dim = self.args.model.fusion.latent_dim
        self.fusion_output_dim = self.args.model.fusion.output_dim
        self.fusion_gnn_layers = self.args.model.fusion.num_layers
        self.fusion_gnn_type = self.args.model.fusion.gnn_type

        # decoder
        self.is_decoder = self.args.model.decoder.enable

        # Adjacent Matrix
        self.adj_mx = args.adj_mx
        self.edge_index = torch.tensor(args.edge_index, dtype=torch.int32).to(self.device)
        self.edge_attr = torch.from_numpy(args.edge_attr).float().to(self.device)

        # wind mean and std
        self.wind_mean = args.data.mean_[-2:]
        self.wind_std = args.data.std_[-2:]

        if self.is_phy:
            self.phy_odefunc = ODEFunc(self.phy_gnn_hid_dim, self.X_dim, self.adj_mx, self.edge_index, self.edge_attr,
                                       self.cheb_k, self.num_nodes, self.device, num_layers=self.phy_gnn_layers,
                                       filter_type=self.knowledge, estimate=self.is_estimator)
            self.coeff_estimator = None
            if self.is_estimator:
                self.coeff_estimator = Coeff_Estimator(input_dim=self.input_dim,
                                                       coeff_dim=self.X_dim, num_nodes=self.num_nodes,
                                                       rnn_dim=self.phy_rnn_dim, n_layers=self.phy_rnn_layers)

            self.RNN_encoder_pred = Encoder_phy_z(self.phy_input_dim, self.phy_latent_dim,
                                                  self.phy_rnn_layers, self.num_nodes)

        if self.is_unk:
            self.encoder = Encoder_unk_z(self.unk_input_dim, self.unk_latent_dim, self.num_nodes,
                                         self.unk_rnn_dim, self.unk_rnn_layers)
            self.unk_odefunc = Unk_odefunc(self.unk_latent_dim, self.num_nodes, self.unk_n_heads,
                                           self.device, self.adj_mx, self.unk_d_f)

        # Knowledge_Fusion
        if self.is_phy and self.is_unk:
            self.gatef_fusion = GNN_Knowledge_Fusion(self.num_nodes, self.phy_latent_dim, self.unk_latent_dim,
                                                     self.fusion_output_dim, self.edge_index, self.edge_attr[:, :1],
                                                     hid_dim=self.fusion_latent_dim, gnn_type=self.fusion_gnn_type,
                                                     num_layers=self.fusion_gnn_layers)   # T x B x N*output_dim

        if self.is_decoder:
            if self.is_phy and self.is_unk:
                ld = self.fusion_output_dim
            elif self.is_phy and not self.is_unk:
                ld = self.unk_latent_dim
            elif not self.is_phy and self.is_unk:
                ld = self.phy_latent_dim
            else:
                ld = None
                assert NotImplementedError

            self.decoder = Conv_Decoder(latent_dim=ld,
                                        output_dim=self.X_dim,
                                        num_nodes=self.num_nodes)
        else:
            assert self.fusion_output_dim == self.X_dim

    def forward(self, inputs, y_embed=None):
        # (X, A)
        seq_len, batch_size = inputs.size(0), inputs.size(1)
        inputs = inputs.reshape(seq_len, batch_size, self.num_nodes, self.input_dim)  # T x B x N x D
        X = inputs[:, :, :, :self.X_dim].reshape((seq_len, batch_size, self.num_nodes * self.X_dim))  # T x B x N*X_dim
        last_X = X[-1]  # B x N*X_dim
        wind_vars = inputs[:, :, :, 4: 6]  # T x B x N x 2   wind speed and wind direction
        last_wind_vars = wind_vars[-1]  # B x N x 2

        if self.embedding_dim:
            x_embed = self.embedding_air(inputs[..., 6:].long())
            inputs = torch.cat((inputs[..., :6], x_embed), -1)    # after embedding

        # MOL on PDEs and solve
        if self.is_phy:
            alpha, beta = None, None
            if self.is_estimator:   # estimate alpha and beta
                alpha, beta = self.coeff_estimator(inputs)
            phy_y, phy_fe = self.phy_part(last_X, last_wind_vars, alpha, beta)  # T x B x N*X_dim
            if self.embedding_dim:
                y_embed = y_embed.reshape(seq_len, batch_size, self.num_nodes, -1)
                y_embed = self.embedding_air(y_embed.long().to(self.device))
                phy_y = torch.cat([phy_y.unsqueeze(-1), y_embed], -1)
            phy_z = self.RNN_encoder_pred(phy_y)   # T x B x N*phy_latent_dim
        else:
            phy_z, phy_fe = None, (0, 0)

        # Data-Driven Dynamics
        if self.is_unk:
            Z0 = self.encoder(inputs)   # B x N*latent_dim
            unk_z, unk_fe = self.unk_part(Z0)   # T x B x N*unk_latent_dim
        else:
            unk_z, unk_fe = None, (0, 0)

        # Dynamics Fusion
        self.loss_CL = None
        if self.is_phy and self.is_unk:
            assert self.unk_latent_dim == self.phy_latent_dim
            self.loss_CL = temporal_alignment(phy_z, unk_z, self.num_nodes, self.unk_latent_dim)
            Z = self.gatef_fusion(phy_z, unk_z)   # T x B x N x latent_dim
            if self.is_decoder:
                self.pred_y = self.decoder(Z)
            else:
                self.pred_y = Z
        elif self.is_unk and not self.is_phy:
            self.pred_y = self.decoder(unk_z)
        elif self.is_phy and not self.is_unk:
            self.pred_y = self.decoder(phy_z)

        return self.pred_y, phy_fe + unk_fe

    def phy_part(self, last_X, last_wind_vars, alpha=None, beta=None):
        self.phy_odefunc.create_equation(last_wind_vars, self.wind_mean, self.wind_std, alpha, beta)
        time_steps_to_predict = torch.arange(start=0, end=self.horizon + 1, step=1).float()  # horizon 1 + 24
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        pred_y, fe = self.phy_solver.solve(self.phy_odefunc, last_X, time_steps_to_predict)  # T x B x N*D
        pred_y = pred_y[1:]

        return pred_y, fe

    def unk_part(self, Z0):
        time_steps_to_predict = torch.arange(start=0, end=self.horizon + 1, step=1).float()  # horizon 1 + 24
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
        pred_z, fe = self.unk_solver.solve(self.unk_odefunc, Z0, time_steps_to_predict)  # T x B x N*D
        pred_z = pred_z[1:]
        return pred_z, fe

    def get_setting(self, args):
        setting = 'Air-DualODE_{}--{}_{}_lr{}_loss{}-{}_cl-coeff_{}_bs{}_ft{}_sl{}_pl{}_Phy{}_Unk{}_Fusion{}_des_{}-{}'.format(
            args.model.phy_func.knowledge,
            f"{int(args.model.phy_func.enable)}{int(args.model.unk_func.enable)}",
            args.data.data_name + "_" + args.data.interval,
            args.train.lr,
            f"{int(args.model.loss.cl_loss)}{int(args.model.loss.pred_loss)}",
            args.model.loss.criterion,
            args.model.loss.cl_coeff,
            args.data.batch_size,
            args.model.input_dim,
            args.model.seq_len,
            args.model.horizon,

            f"{args.model.phy_func.rnn_layers}-" \
            f"{args.model.phy_func.rnn_dim}-"\
            f"{args.model.phy_func.latent_dim}-"\
            f"{args.model.phy_func.gnn_hid_dim}-"\
            f"{args.model.phy_func.gnn_layers}",

            f"{args.model.unk_func.rnn_layers}-"\
            f"{args.model.unk_func.rnn_dim}-"\
            f"{args.model.unk_func.latent_dim}-"\
            f"{args.model.unk_func.n_heads}",

            f"{args.model.fusion.latent_dim}-"\
            f"{args.model.fusion.output_dim}-"\
            f"{args.model.fusion.num_layers}-",

            args.des,
            args.exp_idx
        )
        if len(self._logger.handlers) == 0:
            print('Setting: ', setting)
        else:
            self._logger.info(setting)
        return setting
