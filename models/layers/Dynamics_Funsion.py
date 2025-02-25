import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv
from utils.utils import load_graph_data
from torch.nn import functional as F

class GNN_Knowledge_Fusion(nn.Module):
    def __init__(self, num_nodes, phy_dim, unk_dim, output_dim, edge_index, edge_attr,
                 hid_dim=64, gnn_type='GCN', num_layers=3):
        super(GNN_Knowledge_Fusion, self).__init__()
        self.num_nodes = num_nodes
        self.phy_dim = phy_dim
        self.unk_dim = unk_dim
        self.concat_dim = phy_dim + unk_dim
        self.output_dim = output_dim
        self.edge_index = edge_index   # 2 x M
        self.edge_attr = edge_attr
        self.gnn_type = gnn_type
        if edge_attr.shape[1] > 1:
            assert gnn_type == 'GAT'
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.residual = nn.Identity()

        self.fusion = self.get_fusion()

    def forward(self, phy_hidden, unk_hidden):
        T, B, _ = phy_hidden.shape
        phy_hidden = phy_hidden.reshape(T*B, self.num_nodes, self.phy_dim)
        unk_hidden = unk_hidden.reshape(T*B, self.num_nodes, self.unk_dim)

        concat_hidden = torch.cat((phy_hidden, unk_hidden), dim=2)
        out = self.fusion[0](concat_hidden, self.edge_index, self.edge_attr)
        out = F.relu(out)

        for gnn in self.fusion[1:-1]:
            residual = self.residual(out)
            out = gnn(out, self.edge_index, self.edge_attr)
            out = F.relu(out) + residual

        x = self.fusion[-1](out, self.edge_index, self.edge_attr)

        return x.reshape(T, B, self.num_nodes*self.output_dim)

    def get_fusion(self):
        fusion = nn.ModuleList()
        if self.gnn_type == "GCN":
            fusion.append(
                GCNConv(in_channels=self.concat_dim, out_channels=self.hid_dim,
                        cached=True)
            )

            for _ in range(self.num_layers - 2):
                fusion.append(
                    GCNConv(in_channels=self.hid_dim, out_channels=self.hid_dim,
                            cached=True)
                )

            fusion.append(
                GCNConv(in_channels=self.hid_dim, out_channels=self.output_dim,
                        cached=True)
            )

        elif self.gnn_type == "Cheb":
            fusion.append(
                ChebConv(in_channels=self.concat_dim, out_channels=self.hid_dim,
                         K=3)
            )

            for _ in range(self.num_layers - 2):
                fusion.append(
                    ChebConv(in_channels=self.hid_dim, out_channels=self.hid_dim,
                             K=3)
                )

            fusion.append(
                ChebConv(in_channels=self.hid_dim, out_channels=self.output_dim,
                         K=3)
            )

        elif self.gnn_type == "GAT":
            fusion.append(
                GATConv(in_channels=self.concat_dim, out_channels=self.hid_dim,
                        heads=4, concat=False, edge_dim=self.edge_attr.shape[1])
            )

            for _ in range(self.num_layers - 2):
                fusion.append(
                    GATConv(in_channels=self.hid_dim, out_channels=self.hid_dim,
                            heads=4, concat=False, edge_dim=self.edge_attr.shape[1])
                )

            fusion.append(
                GATConv(in_channels=self.hid_dim, out_channels=self.output_dim,
                        heads=4, concat=False, edge_dim=self.edge_attr.shape[1])
            )

        else:
            raise NotImplementedError

        return fusion

