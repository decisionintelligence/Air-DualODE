import torch
from torch import nn

class Linear_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes):
        super(Linear_Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim*num_nodes, output_dim*num_nodes)

    def forward(self, z):
        # T x N*latent_dim -> T x N*output_dim
        return self.linear(z)

class Conv1d_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, k=1):
        super(Conv1d_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        padding = (k - 1) // 2
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim,
                      out_channels=output_dim,
                      kernel_size=k,
                      padding=padding,
                      bias=True)
        )

    def forward(self, z):
        # T x B x N*latent_dim -> T x B x N*Dout
        T, B, _ = z.shape
        z = z.reshape(T, B*self.num_nodes, self.latent_dim)
        z = z.permute(1, 2, 0)
        z = self.decoder(z)      # B*N x Dout x T
        z = z.permute(2, 0, 1)   # T x B*N x Dout
        z = z.reshape(T, B, self.num_nodes * self.output_dim)
        return z


class Conv2d_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, d_f=64):
        super(Conv2d_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,
                      out_channels=d_f,
                      kernel_size=(1, 1),
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=d_f,
                      out_channels=output_dim,
                      kernel_size=(1, 1),
                      bias=True)
        )

    def forward(self, z):
        # T x B x N*latent_dim -> T x B x N*output_dim
        T, B, _ = z.shape
        z = z.reshape(T, B, self.num_nodes, self.latent_dim)
        z = z.permute(1, 3, 2, 0)   # B x latent_dim x N x T
        z = self.decoder(z)  # B x output_dim x N x T
        z = z.permute(3, 0, 2, 1)
        z = z.reshape(T, B, self.num_nodes*self.output_dim)
        return z


class Conv_seq_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, seq_len=24):
        super(Conv_seq_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.seq_len = 24
        self.spatial_conv = nn.Sequential(
                nn.Conv1d(in_channels=seq_len*latent_dim,
                          out_channels=seq_len*output_dim,
                          kernel_size=1,
                          bias=True)
        )

    def forward(self, z):
        # T x B x N*latent_dim -> T x B x N*Dout
        # spatial_conv's input: B x T*latent_dim x N
        T, B, _ = z.shape
        assert T == self.seq_len
        z = z.reshape(T, B, self.num_nodes, self.latent_dim)
        z = z.permute(1, 0, 3, 2)   # B x T x latent_dim x N
        z = z.reshape(B, T*self.latent_dim, self.num_nodes)
        z = self.spatial_conv(z)    # B x T*Dout x N
        z = z.reshape(B, T, self.output_dim, self.num_nodes)
        z = z.permute(1, 0, 3, 2)
        z = z.reshape(T, B, self.num_nodes*self.output_dim)
        return z