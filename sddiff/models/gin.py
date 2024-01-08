import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Union
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

from torch_geometric.nn import JumpingKnowledge, global_mean_pool
import torch.nn.functional as F
__all__ = [
    'GINEncoder',
]

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super(MultiLayerPerceptron, self).__init__()

        self.layers = [nn.Linear(input_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class GINEConv(MessagePassing):
    def __init__(self, mlp: Callable, eps: float = 0., train_eps: bool = False, **kwargs):
        super(GINEConv, self).__init__(aggr='add', **kwargs)
        self.mlp = mlp
        self.initial_eps = eps

        self.activation = nn.ReLU()     

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.activation(x_j + edge_attr)

    def __repr__(self):
        return '{}(mlp={})'.format(self.__class__.__name__, self.mlp)        

class GINEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, addt, num_convs=3, short_cut=True, concat_hidden=False, use_jump=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(100, hidden_dim)
        self.addt = addt
        # self.emb_t = nn.Embedding(5000, hidden_dim)
        if addt:
            self.reduc_linear = nn.Sequential(nn.Linear(hidden_dim +1, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim))

        self.activation = nn.ReLU()
        
        self.convs = nn.ModuleList()
        for _ in range(self.num_convs):
            self.convs.append(GINEConv(MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim])))

        self.use_jump = use_jump        
        if use_jump:
            self.jump = JumpingKnowledge('cat')
            self.lin_jump = nn.Linear(num_convs * hidden_dim, hidden_dim)
        
    def forward(self, z, edge_index, edge_attr, t):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

        node_attr = self.node_emb(z)    # (num_node, hidden)
        if self.addt:
            node_attr = torch.cat([node_attr, t.unsqueeze(-1)], dim=-1)
            node_attr = self.reduc_linear(node_attr)
 
        hiddens = []
        conv_input = node_attr # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1:
                hidden = self.activation(hidden)
               
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden += conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if self.use_jump:
            x = self.jump(hiddens)
            # x = global_mean_pool(x, batch)
            x = F.relu(self.lin_jump(x))
            # node_feature = F.dropout(x, p=0.5, training=self.training)

        return node_feature