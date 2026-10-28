
"""
23_multiseed_variance.py
Runs Baseline (Independent-GAT) and DiffPocket (mean-pool) across 5
pre-registered seeds {3, 21, 42, 111, 300} for every available gene.
Seeds are FIXED and reported in full -- no cherry-picking.
Logs to results/23_run_log.txt, saves per-seed CSVs + an Excel workbook
with one sheet per seed plus a Summary sheet (mean +/- std).
"""
import glob, logging, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

SEEDS = [3, 21, 42, 111, 300]
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R_KCAL_T_LN10 = 1.987204e-3 * 298.15 * math.log(10.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(RESULTS / "23_run_log.txt"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger()

def set_seed(seed):
    safe_seed = seed % (2**31 - 1) if seed < 0 else seed  # avoid torch negative-seed overflow
    np.random.seed(safe_seed)
    try:
        torch.manual_seed(seed)
    except RuntimeError:
        log.warning(f"torch.manual_seed({seed}) failed, using {safe_seed} instead")
        torch.manual_seed(safe_seed)
    return safe_seed  # use for sklearn's random_state (must be non-negative)

class Baseline(nn.Module):
    def __init__(s, esm_dim=1280, drug_dim=1030, edge_dim=4, h=128):
        super().__init__()
        s.p=nn.Linear(esm_dim,h); s.g1=GATConv(h,h//4,heads=4,edge_dim=edge_dim); s.g2=GATConv(h,h//4,heads=4,edge_dim=edge_dim)
        s.d=nn.Sequential(nn.Linear(drug_dim,h),nn.ReLU(),nn.Linear(h,h))
        s.head=nn.Sequential(nn.Linear(h*2,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,64),nn.ReLU(),nn.Linear(64,1))
    def enc(s,x,ei,ea,b):
        h=F.relu(s.p(x)); h=F.relu(s.g1(h,ei,ea)); h=F.relu(s.g2(h,ei,ea))
        if b is None: b=torch.zeros(h.size(0),dtype=torch.long,device=h.device)
        return global_mean_pool(h,b)
    def forward(s,x,ei,ea,d,b=None):
        return s.head(torch.cat([s.enc(x,ei,ea,b), s.d(d)],-1)).squeeze(-1)

class Enc(nn.Module):
    def __init__(s, esm_dim=1280, edge_dim=4, h=128):
        super().__init__(); s.p=nn.Linear(esm_dim,h); s.g1=GATConv(h,h//4,heads=4,edge_dim=edge_dim); s.g2=GATConv(h,h//4,heads=4,edge_dim=edge_dim)
    def forward(s,x,ei,ea):
        h=F.relu(s.p(x)); h=F.relu(s.g1(h,ei,ea)); return F.relu(s.g2(h,ei,ea))

class DiffPocket(nn.Module):
    def __init__(s, drug_dim=1030, h=128):
        super().__init__(); s.enc=Enc(h=h); s.dp=nn.Linear(h*3,h)
        s.d=nn.Sequential(nn.Linear(drug_dim,h),nn.ReLU(),nn.Linear(h,h))
        s.head=nn.Sequential(nn.Linear(h*2,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,xm,xw,ei,ea,d):
        hm,hw=s.enc(xm,ei,ea),s.enc(xw,ei,ea)
        delta=F.relu(s.dp(torch.cat([hm,hw,hm-hw],-1))).mean(0,keepdim=True)
        hd=s.d(d)
        return s.head(torch.cat([delta.expand(hd.size(0),-1),hd],-1)).squeeze(-1)

def get_graph(cached, mut):
    g = cached.get("graphs", None)
    if g is not None: return g[mut][0].to(DEVICE), g[mut][1].to(DEVICE)
    return cached["edge_index"].to(DEVICE), cached["edge_attr"].to(DEVICE)

def run_baseline(tr, te, cached):
    esm={k:v.to(DEVICE) for k,v in cached["esm_features"].items()}
    dfeat={s:torch.cat([v["morgan"],v["descriptors"]]).to(DEVICE) for s,v in cached["drug_features"].items()}
    m=Baseline().to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    samp={}
    for _,r in tr.iterrows():
        eiw,eaw=get_graph(cached,"WT"); eim,eam=get_graph(cached,r.mutation)
        samp[(r.smiles,"WT")]=(esm["WT"],eiw,eaw,dfeat[r.smiles],r.pKd_wt)
        samp[(r.smiles,r.mutation)]=(esm[r.mutation],eim,eam,dfeat[r.smiles],r.pKd_mut)
    samp=list(samp.values()); m.train()
    for ep in range(35):
        np.random.shuffle(samp)
        for i in range(0,len(samp),64):
            b=samp[i:i+64]
            dl=[Data(x=x[0],edge_index=x[1],edge_attr=x[2]) for x in b]
            pb=Batch.from_data_list(dl).to(DEVICE)
            d=torch.stack([x[3] for x in b]); y=torch.tensor([x[4] for x in b],dtype=torch.float32,device=DEVICE)
            p=m(pb.x,pb.edge_index,pb.edge_attr,d,pb.batch)
            opt.zero_grad(); F.mse_loss(p,y).backward(); opt.step()
    m.eval(); rows=[]
    with torch.no_grad():
        for _,r in te.iterrows():
            eiw,eaw=get_graph(cached,"WT"); eim,eam=get_graph(cached,r.mutation)
            d=dfeat[r.smiles].unsqueeze(0)
            pwt=m(esm["WT"],eiw,eaw,d).item(); pmut=m(esm[r.mutation],eim,eam,d).item()
            rows.append({**r,"pred_ddg":R_KCAL_T_LN10*(pwt-pmut)})
    return pd.DataFrame(rows)

def run_diffpocket(tr, te, cached):
    esm={k:v.to(DEVICE) for k,v in cached["esm_features"].items()}
    dfeat={s:torch.cat([v["morgan"],v["descriptors"]]).to(DEVICE) for s,v in cached["drug_features"].items()}
    m=DiffPocket().to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    rows_tr=tr.to_dict("records"); m.train()
    for ep in range(35):
        np.random.shuffle(rows_tr)
        for i in range(0,len(rows_tr),32):
            b=rows_tr[i:i+32]; preds=[]
            for r in b:
                eiw,eaw=get_graph(cached,"WT"); eim,eam=get_graph(cached,r["mutation"])
                preds.append(m(esm[r["mutation"]],esm["WT"],eim,eam,dfeat[r["smiles"]].unsqueeze(0)))
            p=torch.cat(preds); y=torch.tensor([r["ddG_kcal_mol"] for r in b],dtype=torch.float32,device=DEVICE)
            opt.zero_grad(); F.mse_loss(p,y).backward(); opt.step()
    m.eval(); rows=[]
    with torch.no_grad():
        for r in te.to_dict("records"):
            eiw,eaw=get_graph(cached,"WT"); eim,eam=get_graph(cached,r["mutation"])
            p=m(esm[r["mutation"]],esm["WT"],eim,eam,dfeat[r["smiles"]].unsqueeze(0)).item()
            rows.append({**r,"pred_ddg":p})
    return pd.DataFrame(rows)

def metrics(df, true_col="ddG_kcal_mol"):
    t=df[df.is_trackable]
    if len(t)<2 or t[true_col].std()==0 or t.pred_ddg.std()==0:
        return {"n":len(t),"pearson_r":0.0,"rmse":np.nan,"mae":np.nan}
    r=pearsonr(t[true_col],t.pred_ddg)[0]
    return {"n":len(t),"pearson_r":float(r),
            "rmse":float(np.sqrt(mean_squared_error(t[true_col],t.pred_ddg))),
            "mae":float(mean_absolute_error(t[true_col],t.pred_ddg))}

def get_genes():
    files = glob.glob("data/processed/*_triplets.csv")
    genes = []
    for f in files:
        g = Path(f).stem.replace("_triplets","")
        if Path(f"data/processed/{g}_features.pt").exists():
            genes.append(g)
    return sorted(set(genes))

def main():
    genes = get_genes()
    log.info(f"Genes found with cache ready: {genes}")
    log.info(f"Seeds (fixed, pre-registered, ALL reported): {SEEDS}")

    all_rows = []
    t_start = time.time()
    seed_bar = tqdm(SEEDS, desc="Seeds", position=0)
    for seed in seed_bar:
        seed_bar.set_postfix_str(f"seed={seed}")
        safe_seed = set_seed(seed)
        log.info(f"=== Starting seed {seed} (sklearn safe_seed={safe_seed}) ===")

        gene_bar = tqdm(genes, desc=f"  genes (seed {seed})", position=1, leave=False)
        for gene in gene_bar:
            gene_bar.set_postfix_str(gene)
            try:
                df = pd.read_csv(f"data/processed/{gene}_triplets.csv")
                cached = torch.load(f"data/processed/{gene}_features.pt", map_location=DEVICE, weights_only=False)
                drugs = df["drug_name"].unique()
                kf = KFold(n_splits=5, shuffle=True, random_state=safe_seed)
                d2f = {}
                for fold,(_,te_idx) in enumerate(kf.split(drugs)):
                    for d in drugs[te_idx]: d2f[d]=fold
                df["fold"] = df["drug_name"].map(d2f)

                for model_name, run_fn in [("baseline", run_baseline), ("diffpocket", run_diffpocket)]:
                    t0 = time.time()
                    fold_preds = []
                    for f in range(5):
                        tr, te = df[df.fold!=f], df[df.fold==f]
                        out = run_fn(tr, te, cached)
                        fold_preds.append(out)
                    pooled = pd.concat(fold_preds, ignore_index=True)
                    m = metrics(pooled)
                    m.update({"seed": seed, "gene": gene, "model": model_name,
                              "time_sec": round(time.time()-t0, 1)})
                    all_rows.append(m)
                    log.info(f"  seed={seed} gene={gene} model={model_name} "
                             f"r={m['pearson_r']:.4f} rmse={m.get('rmse',float('nan')):.4f} "
                             f"n={m['n']} ({m['time_sec']}s)")
            except Exception as e:
                log.error(f"  FAILED seed={seed} gene={gene}: {e}")
                continue

        # checkpoint after every seed (so an overnight crash doesn't lose prior seeds)
        pd.DataFrame(all_rows).to_csv(RESULTS / "23_multiseed_raw.csv", index=False)
        log.info(f"=== Finished seed {seed}, checkpoint saved, "
                 f"elapsed {(time.time()-t_start)/60:.1f} min total ===")

    # Final save: Excel with per-seed sheets + summary
    raw = pd.DataFrame(all_rows)
    with pd.ExcelWriter(RESULTS / "23_multiseed_results.xlsx") as xl:
        for seed in SEEDS:
            raw[raw.seed==seed].to_excel(xl, sheet_name=f"seed_{seed}", index=False)
        summary = raw.groupby(["gene","model"]).agg(
            pearson_r_mean=("pearson_r","mean"), pearson_r_std=("pearson_r","std"),
            rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
            n_seeds=("seed","count")).reset_index()
        summary.to_excel(xl, sheet_name="Summary_mean_std", index=False)

    log.info(f"\nDONE. Total time: {(time.time()-t_start)/60:.1f} min")
    log.info(f"Saved -> results/23_multiseed_results.xlsx, results/23_multiseed_raw.csv, "
             f"results/23_run_log.txt")
    print("\n=== SUMMARY (mean +/- std across all 5 seeds) ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
