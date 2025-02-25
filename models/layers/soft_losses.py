import torch
from torch.nn import functional as F
from models.layers.timelags import *
from torch import nn

def temp_CL_soft(z1, z2, timelag_L, timelag_R, num_nodes):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    z = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = torch.sum(logits[:,t]*timelag_L)
    loss += torch.sum(logits[:,T + t]*timelag_R)
    loss /= (2*B*T*num_nodes)
    return loss

def temporal_alignment(Z_P, Z_D, num_nodes, latent_dim):
    T, B, _ = Z_P.shape
    Z_P = Z_P.reshape(T, B, num_nodes, latent_dim)
    Z_D = Z_D.reshape(T, B, num_nodes, latent_dim)
    Z_P = Z_P.permute(1, 2, 0, 3)    # B x N x T x D
    Z_D = Z_D.permute(1, 2, 0, 3)
    Z_P = Z_P.reshape(B*num_nodes, T, latent_dim)
    Z_D = Z_D.reshape(B*num_nodes, T, latent_dim)

    lag = torch.tensor(timelag_sigmoid(T), device=Z_P.device).float()
    timelag_L, timelag_R = dup_matrix(lag)
    loss = temp_CL_soft(Z_P, Z_D, timelag_L, timelag_R, num_nodes)

    return loss

