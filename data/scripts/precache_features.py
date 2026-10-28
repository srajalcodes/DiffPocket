"""
12_precache_generalized.py
Same as 02_precache_features.py, parameterized by --gene / --uniprot.
Run per protein: python 12_precache_generalized.py --gene EGFR --uniprot P00533
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA3TO1 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
          "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
          "THR":"T","VAL":"V","TRP":"W","TYR":"Y"}


def download_structure(uniprot, pdb_path):
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    if pdb_path.exists() and pdb_path.stat().st_size > 1000:
        print(f"Found existing: {pdb_path}"); return
    r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}", timeout=15)
    if r.status_code == 200:
        d = r.json()
        if isinstance(d, list) and d and "pdbUrl" in d[0]:
            r2 = requests.get(d[0]["pdbUrl"], timeout=30)
            pdb_path.write_bytes(r2.content)
            print(f"Saved AlphaFold structure -> {pdb_path}"); return
    raise RuntimeError(f"Could not fetch AlphaFold structure for {uniprot}. "
                        f"Find a suitable PDB ID manually and place it at {pdb_path}.")


def parse_ca(pdb_path):
    coords, seq = [], []
    for line in open(pdb_path, errors="ignore"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            rn = line[17:20].strip()
            if rn in AA3TO1:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                seq.append(AA3TO1[rn])
    return np.array(coords, dtype=np.float32), "".join(seq)



VOL_DELTA = {  # crude per-residue volume/polarity proxy, Å^3 relative to Gly
    'A':88,'R':173,'N':114,'D':111,'C':108,'Q':143,'E':138,'G':60,'H':153,
    'I':166,'L':166,'K':168,'M':162,'F':189,'P':112,'S':89,'T':116,'W':227,
    'Y':193,'V':140}

def perturb_coords_for_mutation(wt_coords, wt_seq, mut_seq, scale=0.5):
    """Locally displaces CA positions near the mutated residue(s) along
    the local normal, proportional to side-chain volume change -- a
    crude but REAL per-mutation structural proxy (not a full relax)."""
    coords = wt_coords.copy()
    diffs = [i for i,(a,b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
    for i in diffs:
        dv = (VOL_DELTA.get(mut_seq[i],120) - VOL_DELTA.get(wt_seq[i],120)) / 227.0
        for j in range(max(0,i-2), min(len(coords), i+3)):
            v1 = coords[max(j,1)] - coords[max(j,1)-1]
            v2 = coords[min(j+1,len(coords)-1)] - coords[min(j,len(coords)-2)]
            normal = np.cross(v1, v2); normal /= (np.linalg.norm(normal)+1e-8)
            falloff = 1.0 - abs(j-i)/3.0
            coords[j] = coords[j] + scale * dv * falloff * normal
    return coords    


def build_edges(coords, cutoff=10.0):
    N = len(coords); ei, ea, frames = [], [], []
    for i in range(N):
        v1 = coords[max(i,1)] - coords[max(i,1)-1]
        v2 = coords[min(i+1,N-1)] - coords[min(i,N-2)]
        ex = (v1+v2)/(np.linalg.norm(v1+v2)+1e-8)
        n = np.cross(v1, v2); nn = np.linalg.norm(n)
        ey = (n/nn) if nn > 1e-6 else np.array([-ex[1],ex[0],0.0])/(np.linalg.norm([-ex[1],ex[0],0.0])+1e-8)
        ez = np.cross(ex, ey)
        frames.append((ex,ey,ez))
    for i in range(N):
        ex,ey,ez = frames[i]
        for j in range(N):
            if i==j: continue
            dist = np.linalg.norm(coords[i]-coords[j])
            if dist <= cutoff or abs(i-j) <= 5:
                u = (coords[j]-coords[i])/(dist+1e-8)
                ei.append([i,j]); ea.append([dist/cutoff, float(u@ex), float(u@ey), float(u@ez)])
    return torch.tensor(ei).t().contiguous(), torch.tensor(ea, dtype=torch.float32)


def extract_esm2(seqs):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()
    out = {}
    for name, seq in tqdm(seqs.items(), desc="ESM-2"):
        _,_,tok = bc([(name, seq)]); tok = tok.to(DEVICE)
        with torch.no_grad():
            r = model(tok, repr_layers=[33])
            out[name] = r["representations"][33][0,1:-1,:].cpu().clone()
    return out


def featurize_drugs(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    out = {}
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        fp = np.zeros((1024,), dtype=np.float32); ConvertToNumpyArray(gen.GetFingerprint(mol), fp)
        d = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                      Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                      Descriptors.NumRotatableBonds(mol)], dtype=np.float32)
        out[s] = {"morgan": torch.tensor(fp), "descriptors": torch.tensor(d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--uniprot", required=True)
    args = ap.parse_args()
    gene = args.gene.lower()

    triplets_file = Path(f"data/processed/{gene}_triplets.csv")
    pdb_path = Path(f"data/structures/{gene}_structure.pdb")
    out_file = Path(f"data/processed/{gene}_features.pt")

    download_structure(args.uniprot, pdb_path)
    df = pd.read_csv(triplets_file)
    print(f"Loaded {len(df)} triplets for {gene.upper()}")

    seqs = {"WT": df["wt_sequence"].iloc[0]}
    for mut, g in df.groupby("mutation"):
        seqs[mut] = g["mut_sequence"].iloc[0]
    print(f"Sequences: {list(seqs.keys())}")

    ca, full_seq = parse_ca(pdb_path)
    wt = seqs["WT"]
    pos = full_seq.find(wt)
    if pos == -1:
        for l in range(50,10,-5):
            pos = full_seq.find(wt[:l])
            if pos != -1: break
    kd_start = pos if pos != -1 else 0
    print(f"kd_start={kd_start}") 
    kd_coords = ca[kd_start:kd_start+len(wt)]
    if len(kd_coords) < len(wt): kd_coords = ca[:len(wt)]
    print(f"Aligned: {len(kd_coords)} CA atoms")

    graphs = {"WT": build_edges(kd_coords)}
    for mut, mseq in seqs.items():
        if mut != "WT":
            graphs[mut] = build_edges(perturb_coords_for_mutation(kd_coords, wt, mseq))
    print(f"Built distinct SE(3) graphs for: {list(graphs.keys())}")

    esm_feats = extract_esm2(seqs)
    drug_feats = featurize_drugs(df["smiles"].unique())

    torch.save({"ca_coords": torch.tensor(kd_coords), "graphs": graphs,
                "esm_features": esm_feats, "drug_features": drug_feats,
                "kd_start": kd_start}, out_file)
    print(f"SUCCESS -> {out_file}")


if __name__ == "__main__":
    main()
