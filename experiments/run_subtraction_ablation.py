"""
29_ablation_joint_no_subtraction.py
Resolves Reviewer Critique #5:
Tests an architecture receiving [h_mut ; h_WT] WITHOUT the subtraction term (h_mut - h_WT).
Proves that explicit differential subtraction is the key inductive bias.
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

class JointConcatNoDelta(nn.Module):
    """Takes [h_mut ; h_WT] but NO (h_mut - h_WT) difference."""
    def __init__(self, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128):
        super().__init__()
        self.p = nn.Linear(esm_dim, h)
        self.g1 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.g2 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        # Takes 2*h instead of 3*h (NO delta term)
        self.dp = nn.Linear(h * 2, h)
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
        # CONCATENATION ONLY: [hm ; hw], NO (hm - hw)
        combined = F.relu(self.dp(torch.cat([hm, hw], -1))).mean(0, keepdim=True)
        hd = self.d(d)
        return self.head(torch.cat([combined.expand(hd.size(0), -1), hd], -1)).squeeze(-1)

def run_kinase(gene):
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
        model = JointConcatNoDelta().to(DEVICE)
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
                    preds.append(model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_wt, ea_wt, dfeat[r["smiles"]].unsqueeze(0)))
                preds = torch.cat(preds)
                y = torch.tensor([r["ddG_kcal_mol"] for r in b], dtype=torch.float32, device=DEVICE)
                opt.zero_grad(); F.mse_loss(preds, y).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in te.to_dict("records"):
                ei_m, ea_m = graphs[r["mutation"]]
                p = model(esm[r["mutation"]], esm["WT"], ei_m, ea_m, ei_wt, ea_wt, dfeat[r["smiles"]].unsqueeze(0)).item()
                all_test.append({**r, "pred_ddg": p})

    t = pd.DataFrame(all_test)
    t = t[t.is_trackable]
    r_val = pearsonr(t.ddG_kcal_mol, t.pred_ddg)[0]
    rmse_val = np.sqrt(mean_squared_error(t.ddG_kcal_mol, t.pred_ddg))
    print(f"[{gene}] Joint-without-Delta (Concat only): r = {r_val:.3f} | RMSE = {rmse_val:.3f} (N={len(t)})")
    return {"kinase": gene, "r_no_delta": round(r_val, 3), "rmse_no_delta": round(rmse_val, 3)}

def main():
    print("Evaluating Joint-without-Delta Ablation (Concatenation only)...")
    res = []
    for g in ["EGFR", "KIT", "FLT3", "ABL1"]:
        res.append(run_kinase(g))
    summary = pd.DataFrame(res)
    summary.to_csv(RESULTS_DIR / "29_ablation_joint_no_delta_summary.csv", index=False)
    print("\n=== SUMMARY: JOINT-WITHOUT-DELTA ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()