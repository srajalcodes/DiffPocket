"""
27_ablation_ca_perturbation.py
Ablation for Reviewer Critique #2:
Compares DiffPocket using:
  1. Rigid unperturbed WT C-alpha coordinates for both states
  2. Volume-perturbed (delta V) coordinates for the mutant state
Evaluated on the canonical 5-fold split across the 4 primary kinases.
"""
from pathlib import Path
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
torch.manual_seed(42); np.random.seed(42)

class DiffPocket(nn.Module):
    def __init__(self, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128):
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
    def enc(self, x, ei, ea):
        h = F.relu(self.p(x)); h = F.relu(self.g1(h, ei, ea)); return F.relu(self.g2(h, ei, ea))
    def forward(self, xm, xw, eim, eam, eiw, eaw, d):
        hm = self.enc(xm, eim, eam)
        hw = self.enc(xw, eiw, eaw)
        delta = F.relu(self.dp(torch.cat([hm, hw, hm - hw], -1))).mean(0, keepdim=True)
        hd = self.d(d)
        return self.head(torch.cat([delta.expand(hd.size(0), -1), hd], -1)).squeeze(-1)

def run_ablation(gene, use_rigid_coords=False):
    g = gene.lower()
    df = pd.read_csv(f"data/processed/{g}_triplets.csv")
    split_file = DATA_DIR / f"{g}_splits_5fold.csv"
    if not split_file.exists():
        split_file = DATA_DIR / "splits_5fold_cold_drug.csv"
    splits = pd.read_csv(split_file)
    df = df.merge(splits[["drug_name", "fold"]].drop_duplicates(), on="drug_name", how="left")
    cached = torch.load(f"data/processed/{g}_features.pt", map_location=DEVICE, weights_only=False)

    graphs = {k: (ei.to(DEVICE), ea.to(DEVICE)) for k, (ei, ea) in cached["graphs"].items()}
    esm = {k: v.to(DEVICE) for k, v in cached["esm_features"].items()}
    dfeat = {s: torch.cat([v["morgan"], v["descriptors"]]).to(DEVICE) for s, v in cached["drug_features"].items()}
    ei_wt, ea_wt = graphs["WT"]

    all_test = []
    for f in sorted(df.fold.unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        tr = tr[tr.is_trackable].copy()
        model = DiffPocket().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rs = tr.to_dict("records")
        model.train()
        for ep in range(35):
            np.random.shuffle(rs)
            for i in range(0, len(rs), 32):
                b = rs[i:i+32]
                preds = []
                for r in b:
                    # If rigid: use WT graph for mutant as well
                    ei_m, ea_m = (ei_wt, ea_wt) if use_rigid_coords else graphs[r["mutation"]]
                    preds.append(model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_wt, ea_wt, dfeat[r["smiles"]].unsqueeze(0)))
                preds = torch.cat(preds)
                y = torch.tensor([r["ddG_kcal_mol"] for r in b], dtype=torch.float32, device=DEVICE)
                opt.zero_grad(); F.mse_loss(preds, y).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in te.to_dict("records"):
                ei_m, ea_m = (ei_wt, ea_wt) if use_rigid_coords else graphs[r["mutation"]]
                p = model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_wt, ea_wt, dfeat[r["smiles"]].unsqueeze(0)).item()
                all_test.append({**r, "pred_ddg": p})

    t = pd.DataFrame(all_test)
    t = t[t.is_trackable]
    r_val = pearsonr(t.ddG_kcal_mol, t.pred_ddg)[0]
    rmse_val = np.sqrt(mean_squared_error(t.ddG_kcal_mol, t.pred_ddg))
    return r_val, rmse_val, len(t)

def main():
    print("Evaluating C-alpha Geometric Perturbation Ablation...")
    rows = []
    for g in ["EGFR", "KIT", "FLT3", "ABL1"]:
        r_pert, rmse_pert, n = run_ablation(g, use_rigid_coords=False)
        r_rig, rmse_rig, _ = run_ablation(g, use_rigid_coords=True)
        print(f"[{g}] Perturbed Cα: r={r_pert:.3f}, RMSE={rmse_pert:.3f} | Rigid Cα: r={r_rig:.3f}, RMSE={rmse_rig:.3f}")
        rows.append({"kinase": g, "N": n, "r_perturbed": round(r_pert, 3), "rmse_perturbed": round(rmse_pert, 3),
                     "r_rigid": round(r_rig, 3), "rmse_rigid": round(rmse_rig, 3),
                     "delta_r": round(r_pert - r_rig, 3), "delta_rmse": round(rmse_pert - rmse_rig, 3)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DIR / "27_ablation_ca_perturbation.csv", index=False)
    print("\n=== SUMMARY: C-ALPHA PERTURBATION ABLATION ===")
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()