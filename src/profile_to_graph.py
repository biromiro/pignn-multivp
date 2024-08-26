import torch
from torch_geometric.data import Data, Dataset
import matplotlib.pyplot as plt

class ProfilesToGraphDataset(Dataset):
    def __init__(self, X, y):
        super(ProfilesToGraphDataset, self).__init__()
        self.X = X
        self.y = y        

    def len(self):
        return self.X.shape[0]

    def get(self, idx):
        # Select the profile
        x_profile = self.X[idx]
        y_profile = self.y[idx]

        # Number of nodes (540 in this case)
        num_nodes = x_profile.shape[0]

        # Edge index (adjacency list)
        edge_index = torch.tensor(
            [[i, i+1] for i in range(num_nodes-1)] + [[i+1, i]
                                                      for i in range(num_nodes-1)],
            dtype=torch.long
        ).t().contiguous()

        # Edge features (R, L differences)
        R_diff = x_profile[1:, 0] - x_profile[:-1, 0]  # R differences
        L_diff = x_profile[1:, 1] - x_profile[:-1, 1]  # L differences
        edge_attr = torch.stack([R_diff, L_diff], dim=1)
        # Duplicate for both directions
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Create the graph object
        data = Data(x=x_profile[:, [2, 3, 4]], edge_index=edge_index, edge_attr=edge_attr, y=y_profile)

        return data
