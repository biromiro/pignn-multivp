import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr) 
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Final output layer
        out = self.fc_out(x)
        return out
