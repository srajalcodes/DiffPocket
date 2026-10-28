"""
20_build_neurips_splits.py (CORRECTED)

Pools all kinase triplets and builds:
  1. New-Drug split: Tanimoto<=0.5 guaranteed via connected components of
     the >0.5-similarity graph (bulletproof -- no incremental-invalidation
     bug; any two drugs in different components are PROVABLY <=0.5 similar,
     since an edge exists iff sim>0.5, so cross-component pairs never have one).
  2. New-Protein: Leave-One-Kinase-Out, one column per held-out gene.
  3. Both-New: LOKO AND New-Drug-test, combined, one column per held-out gene.
  4. Legacy 5-fold cold-drug split (kept for continuity with earlier results).
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold

DATA_DIR = Path("data/processed")
OUT_MASTER = DATA_DIR / "pan_kinase_triplets.csv"
TANIMOTO_THRESH = 0.50
TARGET_TEST_FRAC = 0.20

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def pool_all_triplets():
    files = sorted(glob.glob(str(DATA_DIR / "*_triplets.csv")))
    dfs = []
    for f in files:
        gene = Path(f).stem.replace("_triplets", "").upper()
        tdf = pd.read_csv(f)
        tdf["gene"] = gene
        dfs.append(tdf)
    master = pd.concat(dfs, ignore_index=True)
    master = master.drop_duplicates(subset=["gene", "mutation", "smiles"])
    print(f"Pooled: {len(master)} rows, {master['gene'].nunique()} kinases: "
          f"{sorted(master['gene'].unique())}")
    print(f"Trackable: {master['is_trackable'].sum()} / {len(master)}")
    return master


def build_tanimoto_components(master):
    unique_drugs = sorted(master["drug_name"].unique())
    smiles_map = master.drop_duplicates("drug_name").set_index("drug_name")["smiles"].to_dict()
    fps = {d: gen.GetFingerprint(Chem.MolFromSmiles(smiles_map[d])) for d in unique_drugs}

    G = nx.Graph()
    G.add_nodes_from(unique_drugs)
    for i, d1 in enumerate(unique_drugs):
        for d2 in unique_drugs[i+1:]:
            if DataStructs.TanimotoSimilarity(fps[d1], fps[d2]) > TANIMOTO_THRESH:
                G.add_edge(d1, d2)

    components = sorted(nx.connected_components(G), key=len)  # smallest first
    print(f"\nTanimoto>{TANIMOTO_THRESH} connected components: {len(components)} "
          f"(sizes: {[len(c) for c in components]})")

    target_n = int(TARGET_TEST_FRAC * len(unique_drugs))
    test_drugs, n = set(), 0
    for comp in components:
        if n >= target_n:
            break
        test_drugs |= comp
        n += len(comp)

    train_drugs = set(unique_drugs) - test_drugs
    # Verify: no cross pair should exceed threshold (sanity check, not just trust)
    max_cross = 0.0
    for d1 in test_drugs:
        for d2 in train_drugs:
            s = DataStructs.TanimotoSimilarity(fps[d1], fps[d2])
            max_cross = max(max_cross, s)
    print(f"New-Drug split: {len(test_drugs)} test / {len(train_drugs)} train drugs "
          f"({len(test_drugs)/len(unique_drugs)*100:.1f}% test)")
    print(f"VERIFIED max test-vs-train Tanimoto: {max_cross:.3f} "
          f"(must be <= {TANIMOTO_THRESH})")
    assert max_cross <= TANIMOTO_THRESH, "Split construction bug -- should be impossible"
    return test_drugs


def main():
    master = pool_all_triplets()
    genes = sorted(master["gene"].unique())

    # 1. New-Drug (Tanimoto<=0.5)
    test_drugs = build_tanimoto_components(master)
    master["new_drug_split"] = np.where(master["drug_name"].isin(test_drugs), "test", "train")

    # 2. New-Protein (LOKO), one column per held-out gene
    for held_out in genes:
        master[f"new_protein_holdout_{held_out}"] = np.where(
            master["gene"] == held_out, "test", "train")

    # 3. Both-New: LOKO gene AND drug in the New-Drug test set; everything else train;
    #    held-out gene's rows with a TRAIN-side drug are excluded (can't be used as
    #    either clean train or clean test under this combined regime)
    for held_out in genes:
        col = f"both_new_holdout_{held_out}"
        def assign(row, held_out=held_out):
            if row["gene"] != held_out:
                return "train"
            return "test" if row["drug_name"] in test_drugs else "exclude"
        master[col] = master.apply(assign, axis=1)
        n_test = (master[col] == "test").sum()
        n_excl = (master[col] == "exclude").sum()
        print(f"  both_new[{held_out}]: {n_test} test / {n_excl} excluded")

    # 4. Legacy 5-fold cold-drug (kept for continuity)
    unique_drugs = np.array(sorted(master["drug_name"].unique()))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    d2f = {}
    for fold, (_, te) in enumerate(kf.split(unique_drugs)):
        for d in unique_drugs[te]:
            d2f[d] = fold
    master["fold_cold_drug"] = master["drug_name"].map(d2f)

    master.to_csv(OUT_MASTER, index=False)
    print(f"\nSaved -> {OUT_MASTER}")
    print("\nTrackable pairs per kinase:")
    print(master[master.is_trackable].groupby("gene")["is_trackable"].count()
          .sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
