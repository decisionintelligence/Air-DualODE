from utils.utils import init_network_weights, ConfigDict, split_last_dim
from torch import nn
import torch
from typing import Optional
from torch.nn.modules.rnn import GRU


class Coeff_Estimator_new(nn.Module):
    # diffusion coefficient: BxNx1
    # boundary condition: BxNx1
    def __init__(self, input_dim, coeff_dim, num_nodes, rnn_dim, n_layers):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.coeff_dim = coeff_dim
        self.num_nodes = num_nodes
        self.n_layers = n_layers
        self.rnn_dim = rnn_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.coeff_dim*2)
        )

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: T x B x N*D
        :return: alpha, beta: B x N*coeff_dim
        """
        # shape of outputs: (seq_len, batch, num_senor * rnn_dim)
        seq_len, batch_size = inputs.size(0), inputs.size(1)
        inputs = inputs.reshape(seq_len, batch_size, self.num_nodes, self.input_dim)
        last_inputs = inputs[-1, ...]

        coeff = self.net(last_inputs)  # B x N x 2
        coeff = torch.reshape(coeff, (batch_size, self.num_nodes*self.coeff_dim, 2))

        alpha = coeff[..., 0]
        beta = coeff[..., 1]

        return alpha, beta


class Encoder_phy_z(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, num_nodes):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.phy_latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.gru_rnn = GRU(input_dim, latent_dim, num_layers=self.num_layers)

    def forward(self, X_p):
        """
        :param X_p: T x B x N x D(1 + station_embed_dim + time_embed_dim)
        :return: Z: T x B x N*phy_latent_dim
        """
        T, B = X_p.size(0), X_p.size(1)
        X_p = X_p.reshape(T, B * self.num_nodes, self.input_dim)  # T x B*N x input_dim

        Z_p, _ = self.gru_rnn(X_p)  # T x B*N x phy_latent_dim

        Z_p = Z_p.reshape(T, B, self.num_nodes*self.phy_latent_dim)

        return Z_p


class Encoder_unk_z(nn.Module):
    def __init__(self, input_dim, latent_dim, num_nodes, rnn_dim, n_layers):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.n_layers = n_layers
        self.rnn_dim = rnn_dim
        self.gru_rnn = GRU(self.input_dim, rnn_dim, num_layers=n_layers)

        # hidden to z0 settings
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_dim, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim))

        init_network_weights(self.hiddens_to_z0)

    def forward(self, X):
        """
        encoder forward pass on t time steps
        :param X: shape (seq_len, batch_size, num_nodes, D)
        :return: Z0: shape (batch_size, num_nodes * latent_dim)
        """
        seq_len, batch_size = X.size(0), X.size(1)
        X = X.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)  # (24, 32 * 35 = 1120, input_dim)

        outputs, _ = self.gru_rnn(X)  # 24 x 35*32 x 64(rnn_dim)

        last_output = outputs[-1]
        # (batch_size, num_nodes, rnn_dim)
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, self.rnn_dim))
        Z0 = self.hiddens_to_z0(last_output)
        Z0 = torch.reshape(Z0, (batch_size, self.num_nodes*self.latent_dim))

        return Z0