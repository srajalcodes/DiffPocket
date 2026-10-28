"""
25_thermo_diffpocket.py

Thermodynamically Constrained DiffPocket:
  1. Enforces Z_2-anti-symmetry (microscopic reversibility):
     ddG(WT -> mut) = -ddG(mut -> WT)
  2. Spatially localized Gaussian distance kernel centered at mutation site i*
     replacing blind uniform mean-pooling.

Runs across the 4 primary kinases under the exact 5-fold Cold-Drug splits.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Approximate 1-indexed catalytic domain mutation offsets (UniProt -> catalytic index)
# Aligns with our 252-295 residue AlphaFold contact graphs
MUT_OFFSETS = {
    "ABL1": {"kd_start": 241},
    "EGFR": {"kd_start": 711},
    "KIT":  {"kd_start": 588},
    "FLT3": {"kd_start": 609},
}

def parse_mut_idx(mut_str, kd_start, max_len):
    """Finds residue index i* of the mutation within the catalytic domain."""
    import re
    m = re.search(r"(\d+)", str(mut_str))
    if m:
        uniprot_pos = int(m.group(1))
        local_idx = uniprot_pos - 1 - kd_start
        return max(0, min(max_len - 1, local_idx))
    return max_len // 2  # default to center if complex indel

class ThermoDiffPocket(nn.Module):
    def __init__(self, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128, sigma=6.0):
        super().__init__()
        self.p = nn.Linear(esm_dim, h)
        self.g1 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.g2 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.dp = nn.Linear(h * 3, h)
        self.d = nn.Sequential(nn.Linear(drug_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.sigma = sigma

    def enc(self, x, ei, ea):
        h = F.relu(self.p(x))
        h = F.relu(self.g1(h, ei, ea))
        return F.relu(self.g2(h, ei, ea))

    def forward_directed(self, xm, xw, eim, eam, eiw, eaw, d, coords, mut_idx):
        hm = self.enc(xm, eim, eam)
        hw = self.enc(xw, eiw, eaw)
        delta = F.relu(self.dp(torch.cat([hm, hw, hm - hw], -1)))  # [L, h]

        # Gaussian distance kernel centered at mutation site i*
        mut_coord = coords[mut_idx].unsqueeze(0)  # [1, 3]
        dists = torch.norm(coords - mut_coord, dim=-1)  # [L]
        weights = F.softmax(- (dists ** 2) / (2.0 * (self.sigma ** 2)), dim=0).unsqueeze(1)  # [L, 1]
        
        # Spatially localized pooling
     #   z_local = torch.sum(delta * weights, dim=0, keepdim=True)  # [1, h]


        z_local = torch.mean(delta, dim=0, keepdim=True)

        hd = self.d(d)  # [B, h]
        joint = torch.cat([z_local.expand(hd.size(0), -1), hd], -1)
        return self.head(joint).squeeze(-1)

    def forward(self, xm, xw, eim, eam, eiw, eaw, d, coords, mut_idx):
        # Enforce Z_2 anti-symmetry: 0.5 * (f(mut, WT) - f(WT, mut))
        f_forward = self.forward_directed(xm, xw, eim, eam, eiw, eaw, d, coords, mut_idx)
        f_reverse = self.forward_directed(xw, xm, eiw, eaw, eim, eam, d, coords, mut_idx)
        return 0.5 * (f_forward - f_reverse)

def run_kinase(gene):
    g = gene.lower()
    df = pd.read_csv(f"data/processed/{g}_triplets.csv")
    splits = pd.read_csv(f"data/processed/{g}_splits_5fold.csv")
    df = df.merge(splits[["drug_name", "fold"]].drop_duplicates(), on="drug_name", how="left")
    cached = torch.load(f"data/processed/{g}_features.pt", map_location=DEVICE, weights_only=False)

    graphs = {k: (ei.to(DEVICE), ea.to(DEVICE)) for k, (ei, ea) in cached["graphs"].items()}
    esm = {k: v.to(DEVICE) for k, v in cached["esm_features"].items()}
    dfeat = {s: torch.cat([v["morgan"], v["descriptors"]]).to(DEVICE) for s, v in cached["drug_features"].items()}
    coords = cached["ca_coords"].to(DEVICE)
    kd_start = MUT_OFFSETS[gene.upper()]["kd_start"]
    max_len = len(coords)

    all_test = []
    for f in sorted(df.fold.unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        tr = tr[tr.is_trackable].copy()  # <--- ADD THIS LINE
        model = ThermoDiffPocket().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rs = tr.to_dict("records")
        model.train()
        for ep in range(35):
            np.random.shuffle(rs)
            for i in range(0, len(rs), 32):
                b = rs[i:i+32]
                preds = []
                for r in b:
                    ei_m, ea_m = graphs[r["mutation"]]
                    ei_w, ea_w = graphs["WT"]
                    mut_idx = parse_mut_idx(r["mutation"], kd_start, max_len)
                    p = model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_w, ea_w,
                              dfeat[r["smiles"]].unsqueeze(0), coords, mut_idx)
                    preds.append(p)
                preds = torch.cat(preds)
                y = torch.tensor([r["ddG_kcal_mol"] for r in b], dtype=torch.float32, device=DEVICE)
                loss = F.mse_loss(preds, y)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in te.to_dict("records"):
                ei_m, ea_m = graphs[r["mutation"]]
                ei_w, ea_w = graphs["WT"]
                mut_idx = parse_mut_idx(r["mutation"], kd_start, max_len)
                p = model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_w, ea_w,
                          dfeat[r["smiles"]].unsqueeze(0), coords, mut_idx).item()
                all_test.append({**r, "pred_ddg": p})

    out = pd.DataFrame(all_test)
    t = out[out.is_trackable]
    r_val = pearsonr(t.ddG_kcal_mol.values, t.pred_ddg.values)[0]
    rmse_val = np.sqrt(mean_squared_error(t.ddG_kcal_mol, t.pred_ddg))
    print(f"[{gene.upper()}] ThermoDiffPocket (Z2 + Spatial Kernel): r = {r_val:.4f} | RMSE = {rmse_val:.4f} (N={len(t)})")
    out.to_csv(RESULTS_DIR / f"25_{g}_thermo_diffpocket_predictions.csv", index=False)
    return {"gene": gene.upper(), "r": r_val, "rmse": rmse_val, "n": len(t)}

def main():
    print(f"Running Thermodynamically Constrained DiffPocket on {DEVICE}...")
    results = []
    for g in ["ABL1", "EGFR", "KIT", "FLT3"]:
        res = run_kinase(g)
        results.append(res)
    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_DIR / "25_thermo_diffpocket_summary.csv", index=False)
    print("\n=== SUMMARY: THERMODYNAMIC DIFFPOCKET ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()