"""
26_siamese_all_kinases.py
Runs Siamese-loss Independent-GAT across EGFR, KIT, FLT3, and ABL1.
Isolates loss objective (direct ddG MSE) from architecture (independent GAT).
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
R_KCAL_T_LN10 = 1.987204e-3 * 298.15 * math.log(10.0)

class IndependentGAT(nn.Module):
    def __init__(self, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128):
        super().__init__()
        self.p = nn.Linear(esm_dim, h)
        self.g1 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.g2 = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
        self.d = nn.Sequential(nn.Linear(drug_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def enc(self, x, ei, ea, b=None):
        h = F.relu(self.p(x)); h = F.relu(self.g1(h, ei, ea)); h = F.relu(self.g2(h, ei, ea))
        if b is None: b = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return global_mean_pool(h, b)
    def forward(self, x, ei, ea, d, b=None):
        return self.head(torch.cat([self.enc(x, ei, ea, b), self.d(d)], -1)).squeeze(-1)

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
    eiw, eaw = graphs["WT"]

    all_test = []
    for f in sorted(df.fold.unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        tr = tr[tr.is_trackable].copy()
        model = IndependentGAT().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rs = tr.to_dict("records")
        model.train()
        for ep in range(35):
            np.random.shuffle(rs)
            for i in range(0, len(rs), 32):
                b = rs[i:i+32]
                pwt_list, pmut_list = [], []
                for r in b:
                    eim, eam = graphs[r["mutation"]]
                    d = dfeat[r["smiles"]].unsqueeze(0)
                    pwt_list.append(model(esm["WT"], eiw, eaw, d))
                    pmut_list.append(model(esm[r["mutation"]], eim, eam, d))
                p_wt = torch.cat(pwt_list)
                p_mut = torch.cat(pmut_list)
                pred_ddg = R_KCAL_T_LN10 * (p_wt - p_mut)
                y = torch.tensor([r["ddG_kcal_mol"] for r in b], dtype=torch.float32, device=DEVICE)
                loss = F.mse_loss(pred_ddg, y)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in te.to_dict("records"):
                eim, eam = graphs[r["mutation"]]
                d = dfeat[r["smiles"]].unsqueeze(0)
                pwt = model(esm["WT"], eiw, eaw, d).item()
                pmut = model(esm[r["mutation"]], eim, eam, d).item()
                pred_ddg = R_KCAL_T_LN10 * (pwt - pmut)
                all_test.append({**r, "pred_ddg": pred_ddg})

    out = pd.DataFrame(all_test)
    t = out[out.is_trackable]
    r_val = pearsonr(t.ddG_kcal_mol.values, t.pred_ddg.values)[0]
    rmse_val = np.sqrt(mean_squared_error(t.ddG_kcal_mol, t.pred_ddg))
    print(f"[{gene.upper()}] Siamese Independent-GAT: r = {r_val:.4f} | RMSE = {rmse_val:.4f} (N={len(t)})")
    out.to_csv(RESULTS_DIR / f"26_{g}_siamese_predictions.csv", index=False)
    return {"gene": gene.upper(), "r": r_val, "rmse": rmse_val, "n": len(t)}

def main():
    print(f"Running Siamese Independent-GAT across 4 kinases on {DEVICE}...")
    results = []
    for g in ["ABL1", "EGFR", "KIT", "FLT3"]:
        res = run_kinase(g)
        results.append(res)
    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_DIR / "26_siamese_all_kinases_summary.csv", index=False)
    print("\n=== SUMMARY: SIAMESE INDEPENDENT-GAT (LOSS CONTROL) ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()