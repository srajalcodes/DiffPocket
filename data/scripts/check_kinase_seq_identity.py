"""
20b_check_kinase_seq_identity.py
Checks pairwise sequence identity across the 10 kinases to verify
the NeurIPS protein_seqid <= 0.50 threshold for LOKO / New-Protein splits.
"""
from pathlib import Path
import pandas as pd
from Bio import pairwise2

DATA_DIR = Path("data/processed")
df = pd.read_csv(DATA_DIR / "pan_kinase_triplets.csv")
wt_seqs = df.drop_duplicates("gene").set_index("gene")["wt_sequence"].to_dict()

genes = sorted(wt_seqs.keys())
n = len(genes)
matrix = pd.DataFrame(index=genes, columns=genes, dtype=float)

print("Computing pairwise catalytic domain sequence identity (%):")
for i in range(n):
    g1 = genes[i]
    s1 = wt_seqs[g1]
    matrix.loc[g1, g1] = 100.0
    for j in range(i + 1, n):
        g2 = genes[j]
        s2 = wt_seqs[g2]
        align = pairwise2.align.globalxx(s1, s2, one_alignment_only=True)[0]
        matches = sum(res1 == res2 for res1, res2 in zip(align.seqA, align.seqB))
        ident = (matches / min(len(s1), len(s2))) * 100.0
        matrix.loc[g1, g2] = round(ident, 1)
        matrix.loc[g2, g1] = round(ident, 1)

print(matrix.to_string())
matrix.to_csv("results/20b_kinase_sequence_identity_matrix.csv")

# Report pairs > 50% identity
high_pairs = []
for i in range(n):
    for j in range(i + 1, n):
        val = matrix.loc[genes[i], genes[j]]
        if val > 50.0:
            high_pairs.append((genes[i], genes[j], val))

print("\n--- Summary ---")
if high_pairs:
    print("Pairs with >50% sequence identity:")
    for g1, g2, v in high_pairs:
        print(f"  - {g1} vs {g2}: {v}% (closely related family homologs)")
else:
    print("All kinase pairs are <= 50% sequence identity.")
print("Saved matrix -> results/20b_kinase_sequence_identity_matrix.csv")