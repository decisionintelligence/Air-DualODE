import torch
from torch import nn
from torch.nn import functional as F

class Unk_odefunc_ATT(nn.Module):
    def __init__(self, latent_dim, num_nodes, n_heads, device, adj_mask=None, d_f=32):
        super(Unk_odefunc_ATT, self).__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        if adj_mask is None:
            self.adj_mask = None
        else:
            self.adj_mask = torch.tensor(adj_mask, dtype=torch.int8,
                            device=device) + torch.eye(self.num_nodes, device=device)

        self.fc = nn.Linear(latent_dim, latent_dim)
        self.spatial_att = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(latent_dim)
        self.layer_norm_2 = nn.LayerNorm(latent_dim)
        # residual
        self.residual_1 = nn.Identity()
        self.residual_2 = nn.Identity()

    def forward(self, t, z):
        """
        F^D with attention block
        :param t:
        :param z: B x N*latent_dim
        :return:
        """
        self.nfe += 1
        B = z.shape[0]
        z = z.reshape(B, self.num_nodes, self.latent_dim)  # B x N x latent_dim
        # att-add&norm
        # Masked self-attention
        z = self.residual_1(z) + self.spatial_att(z, z, z, attn_mask=self.adj_mask)[0]
        z = self.residual_1(z) + self.spatial_att(z, z, z)[0]
        z = self.layer_norm_1(z)
        # ffd-add&norm
        z = self.residual_2(z) + F.relu(self.fc(z))
        z = self.layer_norm_2(z)

        return z.reshape(B, self.num_nodes * self.latent_dim)
