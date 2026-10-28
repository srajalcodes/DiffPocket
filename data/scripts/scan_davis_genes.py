"""
11_scan_all_genes.py
Scans DAVIS-complete for ALL genes with WT + >=1 mutant, reports trackable
pair counts. Run this FIRST before deciding which proteins to pursue.
"""
import math
from pathlib import Path
import pandas as pd

INPUT = Path("data/raw/davis_complete.csv")
PROTEIN_COL, SMILES_COL, PKD_COL = "protein", "compound_iso_smiles", "y"
PKD_FLOOR = 5.00

df = pd.read_csv(INPUT, low_memory=False)
df["gene"] = df[PROTEIN_COL].str.split("_").str[0].str.upper()
df["is_phospho"] = df[PROTEIN_COL].str.split("_").str[-1].str.lower() == "p"
df["is_wt"] = df[PROTEIN_COL].str.lower().apply(lambda s: len(s.split("_")) == 1 or
                                                  (len(s.split("_")) == 2 and s.split("_")[-1] == "p"))

rows = []
for gene, g in df.groupby("gene"):
    wt = g[g["is_wt"] & ~g["is_phospho"]]
    mut = g[~g["is_wt"] & ~g["is_phospho"]]
    if len(wt) == 0 or len(mut) == 0:
        continue
    wt_idx = wt.set_index(SMILES_COL)[PKD_COL]
    trackable = 0
    for _, r in mut.iterrows():
        if r[SMILES_COL] in wt_idx.index:
            pkd_wt = wt_idx.loc[r[SMILES_COL]]
            pkd_wt = pkd_wt.iloc[0] if hasattr(pkd_wt, "iloc") else pkd_wt
            if pkd_wt > PKD_FLOOR + 0.01 and r[PKD_COL] > PKD_FLOOR + 0.01:
                trackable += 1
    n_muts = mut[PROTEIN_COL].nunique()
    if trackable > 0:
        rows.append({"gene": gene, "n_mutant_variants": n_muts,
                     "n_mutant_rows": len(mut), "trackable_pairs": trackable})

summary = pd.DataFrame(rows).sort_values("trackable_pairs", ascending=False)
summary.to_csv("data/processed/all_genes_scan.csv", index=False)
print(summary.to_string(index=False))
print(f"\n{len(summary)} genes with any trackable pairs. "
      f"Suggested threshold: >=40 trackable pairs (same floor used for ABL1) "
      f"before committing to full training on a protein.")
print("\nNext: for each gene you choose, get its UniProt ID yourself "
      "(quick search), add to PROTEIN_CONFIGS in 13_run_pipeline_all_proteins.py")