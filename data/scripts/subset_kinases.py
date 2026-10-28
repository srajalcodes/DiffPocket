"""
10_subset_davis_generalized.py
Same logic as 01_subset_davis_complete.py but parameterized by gene name,
so you can re-run for a second protein (check DAVIS-complete's unique
protein list first -- EGFR/KRAS/BRAF may or may not have mutant entries
in this dataset; ABL1 does, confirmed. Run --list-genes first to check
what's actually available before assuming EGFR is there.)
"""
import argparse, math, re, sys
from pathlib import Path
import pandas as pd

R_KCAL = 1.987204e-3; T_KELVIN = 298.15; PKD_FLOOR = 5.00
PROTEIN_COL, SMILES_COL, PKD_COL = "protein", "compound_iso_smiles", "y"


def list_genes(path):
    df = pd.read_csv(path, usecols=[PROTEIN_COL])
    genes = sorted(set(re.sub(r"_(p|[a-z0-9]+)$", "", p, count=0) if False else
                        p.split("_")[0] for p in df[PROTEIN_COL].dropna().unique()))
    print(f"{len(genes)} unique base gene prefixes found, sample:")
    for g in genes[:50]:
        print(f"  - {g}")
    print("\nCheck if your target gene (lowercase, e.g. 'egfr') appears above "
          "AND has mutant suffixes -- not all kinases in Davis have mutant pairs.")


def parse_protein(name, gene_prefix):
    if not isinstance(name, str) or not name.lower().startswith(gene_prefix.lower()):
        return False, None, None
    s = name.strip().lower()
    is_phospho = s.split("_")[-1] == "p"
    core = s.split("_")[:-1] if is_phospho else s.split("_")
    mut = "WT" if len(core) == 1 else core[1].upper()
    return True, mut, is_phospho


def build(input_path, output_path, gene_prefix, phospho_mode):
    df = pd.read_csv(input_path, low_memory=False)
    parsed = df[PROTEIN_COL].apply(lambda n: parse_protein(n, gene_prefix))
    df["is_target"] = parsed.apply(lambda t: t[0])
    df["mutation"] = parsed.apply(lambda t: t[1])
    df["is_phospho"] = parsed.apply(lambda t: t[2])
    df["is_phospho"] = df["is_phospho"].astype("boolean")

    sub = df[df["is_target"]].copy()
    print(f"{gene_prefix.upper()} rows: {len(sub)}, mutations: {sorted(sub['mutation'].dropna().unique())}")
    if len(sub) == 0:
        print("ZERO rows -- this gene has no entries under this prefix in DAVIS-complete. "
              "Run --list-genes to find the correct prefix, or this protein isn't covered.", file=sys.stderr)
        sys.exit(1)

    if phospho_mode == "nonphospho_only":
        sub = sub[sub["is_phospho"] == False]
    elif phospho_mode == "phospho_only":
        sub = sub[sub["is_phospho"] == True]

    wt = sub[sub["mutation"] == "WT"].set_index([SMILES_COL, "is_phospho"])
    mut = sub[sub["mutation"] != "WT"]
    rows = []
    for _, r in mut.iterrows():
        key = (r[SMILES_COL], r["is_phospho"])
        if key not in wt.index: continue
        wt_row = wt.loc[key]
        if isinstance(wt_row, pd.DataFrame): wt_row = wt_row.iloc[0]
        pkd_wt, pkd_mut = float(wt_row[PKD_COL]), float(r[PKD_COL])
        ddg = R_KCAL * T_KELVIN * math.log(10 ** (-pkd_mut) / 10 ** (-pkd_wt))
        trackable = pkd_wt > PKD_FLOOR + 0.01 and pkd_mut > PKD_FLOOR + 0.01
        rows.append({"drug_name": r.get("drug_name", ""), "smiles": r[SMILES_COL],
                     "mutation": r["mutation"], "pKd_wt": pkd_wt, "pKd_mut": pkd_mut,
                     "ddG_kcal_mol": round(ddg, 4), "is_trackable": trackable,
                     "is_mutant_selective": ddg <= -1.36, "is_resistant": ddg >= 1.36,
                     "wt_sequence": wt_row.get("target_sequence", ""),
                     "mut_sequence": r.get("target_sequence", "")})
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} triplets -> {output_path}. Trackable: {out['is_trackable'].sum() if len(out) else 0}")
    if len(out):
        print(out["mutation"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/davis_complete.csv")
    ap.add_argument("--gene", default="egfr")
    ap.add_argument("--output", default=None)
    ap.add_argument("--phospho-mode", default="nonphospho_only")
    ap.add_argument("--list-genes", action="store_true")
    args = ap.parse_args()
    if args.list_genes:
        list_genes(args.input)
    else:
        out = args.output or f"data/processed/{args.gene}_triplets.csv"
        build(args.input, out, args.gene, args.phospho_mode)
