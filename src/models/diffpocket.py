import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DiffPocket(nn.Module):
    def __init__(self, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128):
        super().__init__()
        self.prot_proj = nn.Linear(esm_dim, h)
        self.gat1 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.gat2 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.delta_proj = nn.Linear(h * 3, h)
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, h),
            nn.ReLU(),
            nn.Linear(h, h)
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode_pocket(self, x, edge_index, edge_attr):
        h = F.relu(self.prot_proj(x))
        h = F.relu(self.gat1(h, edge_index, edge_attr))
        return F.relu(self.gat2(h, edge_index, edge_attr))

    def forward(self, x_mut, x_wt, ei_m, ea_m, ei_w, ea_w, drug_feat):
        h_mut = self.encode_pocket(x_mut, ei_m, ea_m)
        h_wt = self.encode_pocket(x_wt, ei_w, ea_w)

        # Differential token: [h_mut ; h_wt ; (h_mut - h_wt)]
        delta = F.relu(self.delta_proj(torch.cat([h_mut, h_wt, h_mut - h_wt], dim=-1)))
        z_bar = delta.mean(dim=0, keepdim=True)  # Mean-pooling

        h_drug = self.drug_proj(drug_feat)
        joint = torch.cat([z_bar.expand(h_drug.size(0), -1), h_drug], dim=-1)
        return self.head(joint).squeeze(-1)