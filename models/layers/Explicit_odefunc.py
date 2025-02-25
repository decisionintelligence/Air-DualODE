import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse
from models.layers.DiffAdv_Fusion import Simple_Gated_Fusion as Gated_Fusion


class ODEFunc(nn.Module):
    def __init__(self, gcn_hidden_dim, input_dim, adj_mx, edge_index, edge_attr,
                 K_neighbour, num_nodes, device, num_layers=2,
                 activation='tanh', filter_type="diff_adv", estimate=False):
        """
        ODEs on explicit space
        :param gcn_hidden_dim: dimensionality of the hidden layers
        :param input_dim: dimensionality used for ODE (input and output).
        :param adj_mx: for diff_adj_mx and adv_adj_mx
        :param edge_index: M x 2
        :param edge_attr: M x D   {diff_dist, dist_km, direction}
        :param K_neighbour:
        :param num_nodes:
        :param num_layers: hidden layers in each ode func.
        :param activation:
        :param filter_type:
        """
        super(ODEFunc, self).__init__()
        self.device = device

        self._activation = torch.tanh if activation == 'tanh' else torch.relu

        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nfe = 0  # Number of function integration

        self._filter_type = filter_type

        self.adj_mx = adj_mx
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64).to(self.device)
        self.edge_attr = edge_attr
        self.K_neighbour = K_neighbour

        self.diff_edge_attr = self.edge_attr[:, 0]
        self.adv_edge_attr = None

        self.estimate = estimate
        if not estimate:
            self.diff_coeff = 0.1
            self.beta = nn.Parameter(torch.zeros(self.num_nodes * self.input_dim))
        else:
            self.diff_coeff, self.beta = None, None

        self.residual = nn.Identity()

        if self._filter_type == "diff":
            self.diff_cheb_conv = self.laplacian_operator()

        elif self._filter_type == "adv":
            self.adv_cheb_conv = self.laplacian_operator()

        elif self._filter_type == "diff_adv":
            self.gated_fusion = Gated_Fusion(self.num_nodes, self.input_dim)

            self.diff_cheb_conv = self.laplacian_operator()
            self.adv_cheb_conv = self.laplacian_operator()
        else:
            raise "Knowledge not registered"

    def create_adv_matrix(self, last_wind_vars, wind_mean, wind_std):
        """

        :param last_wind_vars: last_wind_vars: B x N x 2  (wind_speed[Norm], wind_direction)
        :return: adv_edge_attr  B x M x 1  based on adj_mx
        """
        batch_size = last_wind_vars.shape[0]
        edge_src, edge_target = self.edge_index
        node_src = last_wind_vars[:, edge_src, :]
        node_target = last_wind_vars[:, edge_target, :]

        src_wind_speed = node_src[:, :, 0] * wind_std[0] + wind_mean[0]    # km/h
        src_wind_dir = node_src[:, :, 1] * wind_std[1] + wind_mean[1]
        dist = self.edge_attr[:, 1].unsqueeze(dim=0).repeat(batch_size, 1)
        dist_dir = self.edge_attr[:, 2].unsqueeze(dim=0).repeat(batch_size, 1)

        src_wind_dir = (src_wind_dir + 180) % 360
        theta = torch.abs(dist_dir - src_wind_dir)
        adv_edge_attr = F.relu(3 * src_wind_speed * torch.cos(theta) / dist)  # B x M

        return adv_edge_attr

    def create_equation(self, last_wind_vars, wind_mean, wind_std, diff_coeff, beta):
        if self.estimate:
            self.diff_coeff = 0.1
            self.beta = beta
        if self._filter_type == "diff":
            pass
        elif self._filter_type == "adv":
            self.adv_edge_attr = self.create_adv_matrix(last_wind_vars, wind_mean, wind_std)
        elif self._filter_type == "diff_adv":
            self.adv_edge_attr = self.create_adv_matrix(last_wind_vars, wind_mean, wind_std)
        else:
            print("Invalid Filter Type")

    def forward(self, t_local, Xt):
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, Xt)
        return grad

    def get_ode_gradient_nn(self, t, Xt):
        if (self._filter_type == "diff"):
            grad = - self.diff_coeff * self.ode_func_net_diff(Xt, self.diff_edge_attr)
        elif (self._filter_type == "adv"):
            grad = - self.ode_func_net_adv(Xt, self.adv_edge_attr)
        elif (self._filter_type == "diff_adv"):
            grad_diff = - self.diff_coeff * self.ode_func_net_diff(Xt, self.diff_edge_attr)
            grad_adv = - self.ode_func_net_adv(Xt, self.adv_edge_attr)
            grad = self.gated_fusion(grad_diff, grad_adv)
        else:
            raise "Invalid Filter Type"

        grad = grad + self.beta * Xt

        return grad

    def ode_func_net_diff(self, x, edge_attr):
        # x: B x N*var_dim
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.num_nodes, self.input_dim))

        x = self.diff_cheb_conv[0](x, self.edge_index, edge_attr, lambda_max=2)
        x = self._activation(x)

        for op in self.diff_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, self.edge_index, edge_attr, lambda_max=2)
            x = self._activation(x) + residual

        x = self.diff_cheb_conv[-1](x, self.edge_index, edge_attr, lambda_max=2)

        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    def ode_func_net_adv(self, x, edge_attr):
        batch_size = x.shape[0]
        batch = torch.arange(0, batch_size)
        batch = torch.repeat_interleave(batch, self.num_nodes).to(self.device)
        x = x.reshape(batch_size * self.num_nodes, -1)  # B*N x input_dim
        edge_indices = []
        for i in range(batch_size):
            edge_indices.append(self.edge_index + i * self.num_nodes)
        edge_index = torch.cat(edge_indices, dim=1)  # 2 x B*M
        edge_attr = edge_attr.flatten()  # B*M

        x = self.adv_cheb_conv[0](x, edge_index, edge_attr, batch=batch, lambda_max=2)
        x = self._activation(x)

        for op in self.adv_cheb_conv[1:-1]:
            residual = self.residual(x)
            x = op(x, edge_index, edge_attr, batch=batch, lambda_max=2)
            x = self._activation(x) + residual

        x = self.adv_cheb_conv[-1](x, edge_index, edge_attr, batch=batch, lambda_max=2)

        x = x.reshape(batch_size, self.num_nodes, self.input_dim)
        return x.reshape((batch_size, self.num_nodes * self.input_dim))

    @staticmethod
    def dense_to_sparse(adj: torch.Tensor):
        batch_size, num_nodes, _ = adj.size()
        edge_indices = []
        edge_attrs = []

        for i in range(batch_size):
            edge_index, edge_attr = dense_to_sparse(adj[i])
            edge_indices.append(edge_index + i * num_nodes)
            edge_attrs.append(edge_attr)

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)

        return edge_index, edge_attr

    def laplacian_operator(self):
        # approximate Laplacian
        operator = nn.ModuleList()
        operator.append(
            ChebConv(in_channels=self.input_dim, out_channels=self.gcn_hidden_dim,
                     K=self.K_neighbour, normalization='sym',
                     bias=True)
        )

        for _ in range(self.num_layers - 2):
            operator.append(
                ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.gcn_hidden_dim,
                         K=self.K_neighbour, normalization='sym',
                         bias=True)
            )

        operator.append(
            ChebConv(in_channels=self.gcn_hidden_dim, out_channels=self.input_dim,
                     K=self.K_neighbour, normalization='sym',
                     bias=True)
        )

        return operator