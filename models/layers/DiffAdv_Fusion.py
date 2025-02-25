import torch
import torch.nn as nn
from torch.nn import functional as F

class Simple_Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim):
        super(Simple_Gated_Fusion, self).__init__()

        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.gated_fc = nn.Linear(2, 1)

    def forward(self, grad_diff, grad_adv):
        """

        :param grad_diff: B x NDout
        :param grad_adv: B x NDout
        :return: B x NDout
        """
        B = grad_diff.shape[0]
        grad_diff = grad_diff.reshape(B, self.num_nodes, self.var_dim)
        grad_adv = grad_adv.reshape(B, self.num_nodes, self.var_dim)
        concat = torch.cat((grad_diff, grad_adv), dim=-1)  # B x N x 2
        g = torch.sigmoid(self.gated_fc(concat))

        grad_diff_adv = g * grad_diff + (1 - g) * grad_adv  # B x N x 1

        return grad_diff_adv.reshape(B, self.num_nodes*self.var_dim)