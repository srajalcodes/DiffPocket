"""
13_run_all_proteins.py
Orchestrates data-subset + precache for every protein over the trackable
threshold. Does NOT retrain models itself (03/04/08 are hardcoded to
abl1_* filenames) -- after this runs, for each gene either:
  (a) temporarily edit DATA paths in 03/04/08 to f"{gene}_triplets.csv"
      etc. and re-run them per gene, OR
  (b) ask me to generalize 03/04/08 the same way 02->12 was done
      (straightforward, same pattern, just not done yet to save tokens
      until you confirm you want all 4 proteins trained).
"""
import subprocess, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

PROTEIN_CONFIGS = {
    # gene: uniprot -- VERIFY these yourself before trusting the download
    "abl1": "P00519",   # already done, skip
    "egfr": "P00533",
    "kit": "P10721",
    "flt3": "P36888",
}
THRESHOLD = 40

def make_cold_drug_split(gene):
    df = pd.read_csv(f"data/processed/{gene}_triplets.csv")
    drugs = df["drug_name"].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    d2f = {}
    for fold, (_, test_idx) in enumerate(kf.split(drugs)):
        for d in drugs[test_idx]: d2f[d] = fold
    df["fold"] = df["drug_name"].map(d2f)
    df.to_csv(f"data/processed/{gene}_splits_5fold.csv", index=False)
    print(f"  split saved: data/processed/{gene}_splits_5fold.csv")

def main():
    scan = pd.read_csv("data/processed/all_genes_scan.csv")
    for gene, uniprot in PROTEIN_CONFIGS.items():
        if gene == "abl1":
            print(f"{gene.upper()}: already done, skipping."); continue
        row = scan[scan.gene == gene.upper()]
        n_track = int(row["trackable_pairs"].iloc[0]) if len(row) else 0
        if n_track < THRESHOLD:
            print(f"{gene.upper()}: {n_track} trackable pairs < {THRESHOLD}, skipping."); continue

        print(f"\n=== {gene.upper()} ({n_track} trackable pairs) ===")
        subprocess.run([sys.executable, "10_subset_davis_generalized.py",
                         "--gene", gene, "--phospho-mode", "nonphospho_only"], check=True)
        subprocess.run([sys.executable, "12_precache_generalized.py",
                         "--gene", gene, "--uniprot", uniprot], check=True)
        make_cold_drug_split(gene)

    print("\nData + precache + splits done for all qualifying proteins.")
    print("Next: generalize 03/04/08 to accept --gene (same pattern as 01->10, "
          "02->12) so training loops the same way -- say the word and I'll write it.")

if __name__ == "__main__":
    main()
