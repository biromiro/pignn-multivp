import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()

        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = torch.nn.Sequential(
                torch.nn.Linear(edge_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))

        self.out_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.out_layer(x)

        return x
