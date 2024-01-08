import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi as PI
from torch_geometric.nn import MessagePassing

__all__ = [
    'SchNetEncoder',
]

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, mlp, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.mlp = mlp
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.mlp(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        return self.lin2(x)

    def message(self, x_j, W):
        return x_j * W

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        return self.lin(self.act(self.conv(x, edge_index, edge_length, edge_attr)))
    
class SchNetEncoder(nn.Module):
    def __init__(self, addt, hidden_channels=128, num_filters=128,
                num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.addt = addt

        self.embedding = nn.Embedding(100, hidden_channels, max_norm=10.0)
        # self.embedding_t = nn.Embedding(5000, hidden_channels, max_norm=10.0)
        if addt:
            self.reduc_linear = nn.Sequential(nn.Linear(hidden_channels +1, hidden_channels),
                                            nn.ReLU(),
                                            nn.Linear(hidden_channels, hidden_channels))
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)

    def forward(self, z, edge_index, edge_length, edge_attr, t, embed_node=True):
        if embed_node:
            h = self.embedding(z)
            # t = self.embedding_t(t)
            if self.addt:
                h = torch.cat([h, t.unsqueeze(-1)], dim=-1)
                h = self.reduc_linear(h)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        return h