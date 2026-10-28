"""
21_precache_remaining_kinases.py
Runs 10_subset_davis_generalized.py + 12_precache_generalized.py for each
remaining kinase, in order, with a tqdm progress bar.
"""
import subprocess, sys, time
from tqdm import tqdm

KINASES = {
    "ret": "P07949",
    "met": "P08581",
    "pik3ca": "P42336",
    "fgfr3": "P22607",
    "lrrk2": "Q5S007",
    "braf": "P15056",
}

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"\n$ {' '.join(cmd)}")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    pbar = tqdm(KINASES.items(), desc="Precaching kinases", unit="gene")
    for gene, uniprot in pbar:
        pbar.set_postfix_str(f"{gene} (subset)")
        t0 = time.time()
        run([sys.executable, "10_subset_davis_generalized.py",
             "--gene", gene, "--phospho-mode", "nonphospho_only"])

        pbar.set_postfix_str(f"{gene} (precache: structure+ESM2+graph)")
        run([sys.executable, "12_precache_generalized.py",
             "--gene", gene, "--uniprot", uniprot])

        pbar.set_postfix_str(f"{gene} done ({time.time()-t0:.0f}s)")

    print("\nAll 6 remaining kinases precached. Combined with abl1/egfr/kit/flt3, "
          "11 kinases ready for 20_build_neurips_splits.py.")

if __name__ == "__main__":
    main()
