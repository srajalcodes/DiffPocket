"""
21_run_neurips_benchmark.py
Automated 4-way baseline ladder on the 3 NeurIPS 2025 split regimes:
  - Setting 1: New-Drug (Tanimoto <= 0.50)
  - Setting 2: New-Protein / LOKO (Leave-One-Kinase-Out)
  - Setting 3: Both-New (Unseen Drug + Unseen Kinase)
Runs all models sequentially with tqdm and saves checkpoints and metrics.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score
from tqdm import tqdm

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
R_KCAL_T_LN10 = 1.987204e-3 * 298.15 * math.log(10.0)

AA = "ACDEFGHIKLMNPQRSTVWYX"
AA2IDX = {a: i + 1 for i, a in enumerate(AA)}

def encode_seq(seq, max_len=300):
    idx = [AA2IDX.get(c, 0) for c in seq[:max_len]]
    idx += [0] * (max_len - len(idx))
    return torch.tensor(idx, dtype=torch.long)

# ----------------- Models ----------------- #
class DeepDTAStyle(nn.Module):
    def __init__(self, vocab=22, emb_dim=64, drug_dim=1030, h=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 64, 8), nn.ReLU(),
            nn.Conv1d(64, 96, 8), nn.ReLU(),
            nn.Conv1d(96, 128, 8), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.head = nn.Sequential(
            nn.Linear(128 + h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, seq_idx, d):
        h = self.conv(self.embed(seq_idx).transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([h, self.drug_proj(d)], -1)).squeeze(-1)

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
    def forward(self, x, ei, ea, d, b=None):
        h = F.relu(self.p(x)); h = F.relu(self.g1(h, ei, ea)); h = F.relu(self.g2(h, ei, ea))
        if b is None: b = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return self.head(torch.cat([global_mean_pool(h, b), self.d(d)], -1)).squeeze(-1)

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

# ----------------- Metrics ----------------- #
def safe_corr(y, p, fn):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0: return 0.0
    v, _ = fn(y, p); return float(v) if not np.isnan(v) else 0.0

def eval_metrics(t):
    if len(t) == 0: return dict(n=0, r=0.0, rho=0.0, rmse=0.0, mae=0.0, auroc=0.0, aupr=0.0, rp_same_lig=0.0)
    y, p = t["ddG_kcal_mol"].values, t["pred_ddg"].values
    rp = safe_corr(y, p, pearsonr)
    rs = safe_corr(y, p, spearmanr)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    yres = t["is_resistant"].astype(int).values
    auroc = roc_auc_score(yres, p) if len(np.unique(yres)) > 1 else np.nan
    aupr = average_precision_score(yres, p) if len(np.unique(yres)) > 1 else np.nan

    # Same-ligand, different-modifications Rp (Eq 7 from NeurIPS)
    per_lig_r = []
    for (g, smi), grp in t.groupby(["gene", "smiles"]):
        if len(grp) >= 3:
            r_val = safe_corr(grp["ddG_kcal_mol"].values, grp["pred_ddg"].values, pearsonr)
            per_lig_r.append(r_val)
    rp_same_lig = float(np.mean(per_lig_r)) if per_lig_r else np.nan

    return dict(n=len(t), r=rp, rho=rs, rmse=rmse, mae=mae, auroc=auroc, aupr=aupr, rp_same_lig=rp_same_lig)

def is_test(series):
    return series.astype(str).str.lower().isin(["test", "true", "1"])

def is_train(series):
    return series.astype(str).str.lower().isin(["train", "false", "0"])
# ----------------- Main Execution ----------------- #
def main():
    print(f"Device: {DEVICE}")
    master = pd.read_csv(DATA_DIR / "pan_kinase_triplets.csv")
    print(f"Loaded master: {len(master)} rows, {master['is_trackable'].sum()} trackable.")

    # Load all cached kinase graphs
    print("Loading pre-cached kinase graphs...")
    cached_all = {}
    for g in master["gene"].unique():
        f = DATA_DIR / f"{g.lower()}_features.pt"
        if f.exists():
            c = torch.load(f, map_location=DEVICE, weights_only=False)
            graphs = {k: (ei.to(DEVICE), ea.to(DEVICE)) for k, (ei, ea) in c["graphs"].items()}
            esm = {k: v.to(DEVICE) for k, v in c["esm_features"].items()}
            dfeat = {s: torch.cat([v["morgan"], v["descriptors"]]).to(DEVICE) for s, v in c["drug_features"].items()}
            cached_all[g] = {"graphs": graphs, "esm": esm, "dfeat": dfeat}

    seq_cache = {}
    for g, grp in master.groupby("gene"):
        seq_cache[f"{g}_WT"] = encode_seq(grp["wt_sequence"].iloc[0]).to(DEVICE)
        for mut, mgrp in grp.groupby("mutation"):
            seq_cache[f"{g}_{mut}"] = encode_seq(mgrp["mut_sequence"].iloc[0]).to(DEVICE)

    # Define the 3 Settings to run
    experiments = []

    # 1. Setting 1: New-Drug
# 1. Setting 1: New-Drug
    tr_nd = master[is_train(master["new_drug_split"])].copy()
    te_nd = master[is_test(master["new_drug_split"])].copy()
    experiments.append(("Setting1_NewDrug", "all", tr_nd, te_nd))

    # 2. Setting 2: LOKO (Leave-One-Kinase-Out)
    for g in ["EGFR", "KIT", "FLT3", "ABL1", "RET", "MET"]:
        col_np = f"new_protein_holdout_{g}"
        if col_np in master.columns:
            tr_loko = master[is_train(master[col_np])].copy()
            te_loko = master[is_test(master[col_np])].copy()
        else:
            tr_loko = master[master["gene"] != g].copy()
            te_loko = master[master["gene"] == g].copy()
        experiments.append((f"Setting2_LOKO_{g}", g, tr_loko, te_loko))

    # 3. Setting 3: Both-New
    for g in ["EGFR", "KIT", "FLT3", "ABL1"]:
        col_bn = f"both_new_holdout_{g}"
        if col_bn in master.columns:
            tr_bn = master[is_train(master[col_bn])].copy()
            te_bn = master[is_test(master[col_bn])].copy()
        else:
            tr_bn = master[(master["gene"] != g) & is_train(master["new_drug_split"])].copy()
            te_bn = master[(master["gene"] == g) & is_test(master["new_drug_split"])].copy()
        experiments.append((f"Setting3_BothNew_{g}", g, tr_bn, te_bn))

    all_summary = []
    pbar = tqdm(experiments, desc="Running Benchmark Suites")

    for exp_name, target_g, tr, te in pbar:
        te_track = te[te.is_trackable].copy()
        pbar.set_postfix_str(f"{exp_name} (train={len(tr)}, test_track={len(te_track)})")

        # ---- Model 1: Ligand-Blind ----
        mut_map = {m: i for i, m in enumerate(tr["mutation"].unique())}
        class Blind(nn.Module):
            def __init__(self, k): super().__init__(); self.emb = nn.Embedding(k+1, 16); self.h = nn.Linear(16, 1)
            def forward(self, idx): return self.h(self.emb(idx)).squeeze(-1)
        m_blind = Blind(len(mut_map)).to(DEVICE)
        opt = torch.optim.Adam(m_blind.parameters(), lr=1e-2)
        idx_tr = torch.tensor([mut_map.get(m, len(mut_map)) for m in tr.mutation], device=DEVICE)
        y_tr = torch.tensor(tr.ddG_kcal_mol.values, dtype=torch.float32, device=DEVICE)
        m_blind.train()
        for _ in range(60):
            opt.zero_grad(); F.mse_loss(m_blind(idx_tr), y_tr).backward(); opt.step()
        m_blind.eval()
        with torch.no_grad():
            idx_te = torch.tensor([mut_map.get(m, len(mut_map)) for m in te_track.mutation], device=DEVICE)
            preds_blind = m_blind(idx_te).cpu().numpy()
        te_track["pred_ddg"] = preds_blind
        m1 = eval_metrics(te_track); m1.update(dict(experiment=exp_name, model="LigandBlind"))
        all_summary.append(m1)

        # ---- Model 2: DeepDTA-Style ----
        m_dta = DeepDTAStyle().to(DEVICE)
        opt = torch.optim.AdamW(m_dta.parameters(), lr=1e-3, weight_decay=1e-4)
        samp_dta = []
        for _, r in tr.iterrows():
            g_i = r["gene"]
            if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                df_vec = cached_all[g_i]["dfeat"][r["smiles"]]
                samp_dta.append((seq_cache[f"{g_i}_WT"], df_vec, r["pKd_wt"]))
                samp_dta.append((seq_cache[f"{g_i}_{r['mutation']}"], df_vec, r["pKd_mut"]))
        m_dta.train()
        for ep in range(25):
            np.random.shuffle(samp_dta)
            for i in range(0, len(samp_dta), 64):
                b = samp_dta[i:i+64]
                sq = torch.stack([x[0] for x in b])
                df_b = torch.stack([x[1] for x in b])
                yb = torch.tensor([x[2] for x in b], dtype=torch.float32, device=DEVICE)
                opt.zero_grad(); F.mse_loss(m_dta(sq, df_b), yb).backward(); opt.step()
        m_dta.eval()
        preds_dta = []
        with torch.no_grad():
            for _, r in te_track.iterrows():
                g_i = r["gene"]
                if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                    d_v = cached_all[g_i]["dfeat"][r["smiles"]].unsqueeze(0)
                    pwt = m_dta(seq_cache[f"{g_i}_WT"].unsqueeze(0), d_v).item()
                    pmt = m_dta(seq_cache[f"{g_i}_{r['mutation']}"].unsqueeze(0), d_v).item()
                    preds_dta.append(R_KCAL_T_LN10 * (pwt - pmt))
                else: preds_dta.append(0.0)
        te_track["pred_ddg"] = preds_dta
        m2 = eval_metrics(te_track); m2.update(dict(experiment=exp_name, model="DeepDTA_Style"))
        all_summary.append(m2)

        # ---- Model 3: Independent-GAT ----
        m_gat = IndependentGAT().to(DEVICE)
        opt = torch.optim.AdamW(m_gat.parameters(), lr=1e-3, weight_decay=1e-4)
        samp_gat = []
        for _, r in tr.iterrows():
            g_i = r["gene"]
            if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                cg = cached_all[g_i]
                df_vec = cg["dfeat"][r["smiles"]]
                ei_w, ea_w = cg["graphs"]["WT"]
                ei_m, ea_m = cg["graphs"][r["mutation"]]
                samp_gat.append((cg["esm"]["WT"], ei_w, ea_w, df_vec, r["pKd_wt"]))
                samp_gat.append((cg["esm"][r["mutation"]], ei_m, ea_m, df_vec, r["pKd_mut"]))
        m_gat.train()
        for ep in range(25):
            np.random.shuffle(samp_gat)
            for i in range(0, len(samp_gat), 64):
                b = samp_gat[i:i+64]
                df_b = torch.stack([x[3] for x in b])
                yb = torch.tensor([x[4] for x in b], dtype=torch.float32, device=DEVICE)
                dl = [Data(x=x[0], edge_index=x[1], edge_attr=x[2]) for x in b]
                pb = Batch.from_data_list(dl).to(DEVICE)
                opt.zero_grad(); F.mse_loss(m_gat(pb.x, pb.edge_index, pb.edge_attr, df_b, pb.batch), yb).backward(); opt.step()
        m_gat.eval()
        preds_gat = []
        with torch.no_grad():
            for _, r in te_track.iterrows():
                g_i = r["gene"]
                if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                    cg = cached_all[g_i]
                    d_v = cg["dfeat"][r["smiles"]].unsqueeze(0)
                    ei_w, ea_w = cg["graphs"]["WT"]
                    ei_m, ea_m = cg["graphs"][r["mutation"]]
                    pwt = m_gat(cg["esm"]["WT"], ei_w, ea_w, d_v).item()
                    pmt = m_gat(cg["esm"][r["mutation"]], ei_m, ea_m, d_v).item()
                    preds_gat.append(R_KCAL_T_LN10 * (pwt - pmt))
                else: preds_gat.append(0.0)
        te_track["pred_ddg"] = preds_gat
        m3 = eval_metrics(te_track); m3.update(dict(experiment=exp_name, model="IndependentGAT"))
        all_summary.append(m3)

        # ---- Model 4: DiffPocket (Mean-Pool Delta) ----
        m_diff = DiffPocket().to(DEVICE)
        opt = torch.optim.AdamW(m_diff.parameters(), lr=1e-3, weight_decay=1e-4)
        samp_diff = []
        for _, r in tr[tr.is_trackable].iterrows():
            g_i = r["gene"]
            if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                samp_diff.append((g_i, r["mutation"], r["smiles"], float(r["ddG_kcal_mol"])))
        m_diff.train()
        for ep in range(30):
            np.random.shuffle(samp_diff)
            for i in range(0, len(samp_diff), 32):
                b = samp_diff[i:i+32]
                preds = []
                for g_i, mut, smi, _ in b:
                    cg = cached_all[g_i]
                    d_v = cg["dfeat"][smi].unsqueeze(0)
                    ei_m, ea_m = cg["graphs"][mut]
                    ei_w, ea_w = cg["graphs"]["WT"]
                    preds.append(m_diff(cg["esm"][mut], cg["esm"]["WT"], ei_m, ea_m, ei_w, ea_w, d_v))
                preds = torch.cat(preds)
                yb = torch.tensor([x[3] for x in b], dtype=torch.float32, device=DEVICE)
                opt.zero_grad(); F.mse_loss(preds, yb).backward(); opt.step()
        m_diff.eval()
        preds_diff = []
        with torch.no_grad():
            for _, r in te_track.iterrows():
                g_i = r["gene"]
                if g_i in cached_all and r["smiles"] in cached_all[g_i]["dfeat"]:
                    cg = cached_all[g_i]
                    d_v = cg["dfeat"][r["smiles"]].unsqueeze(0)
                    ei_m, ea_m = cg["graphs"][r["mutation"]]
                    ei_w, ea_w = cg["graphs"]["WT"]
                    preds_diff.append(m_diff(cg["esm"][r["mutation"]], cg["esm"]["WT"], ei_m, ea_m, ei_w, ea_w, d_v).item())
                else: preds_diff.append(0.0)
        te_track["pred_ddg"] = preds_diff
        m4 = eval_metrics(te_track); m4.update(dict(experiment=exp_name, model="DiffPocket"))
        all_summary.append(m4)

        # Save Checkpoint
        torch.save(m_diff.state_dict(), CKPT_DIR / f"{exp_name}_diffpocket.pt")

    # Final Output Summary Table
    res_df = pd.DataFrame(all_summary)
    res_df.to_csv(RESULTS_DIR / "21_neurips_benchmark_results.csv", index=False)
    print("\n================ BENCHMARK COMPLETED ================")
    print(res_df[["experiment", "model", "n", "r", "rmse", "rp_same_lig"]].to_string(index=False))
    print(f"\nSaved full results -> results/21_neurips_benchmark_results.csv")

if __name__ == "__main__":
    main()