"""
28_partial_correlation_ligand_signal.py
Resolves Reviewer Critique #3:
Proves that DiffPocket learns compound-specific interactions beyond mutation averages.
Computes:
  1. Per-mutation stratified Pearson r
  2. Partial correlation controlling for mutation identity: r(pred, true | mut_mean)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

RESULTS_DIR = Path("results")
pred_file = RESULTS_DIR / "predictions" / "predictions_diffpocket_setting1.csv"
df = pd.read_csv(pred_file)

# 1. Compute empirical mutation mean from training data or pooled test
mut_means = df.groupby("mutation")["ddG_kcal_mol"].transform("mean")
df["mut_mean"] = mut_means

# Residuals after removing mutation effect
res_true = df["ddG_kcal_mol"] - df["mut_mean"]
res_pred = df["pred_ddg"] - df.groupby("mutation")["pred_ddg"].transform("mean")

# Partial correlation
r_partial, p_val = pearsonr(res_true, res_pred)
r_overall, _ = pearsonr(df["ddG_kcal_mol"], df["pred_ddg"])

print("=== SETTING 1: LIGAND-SPECIFIC SIGNAL ANALYSIS ===")
print(f"Overall Pearson r                              : {r_overall:.4f}")
print(f"Partial Pearson r (controlling for mutation ID): {r_partial:.4f} (p = {p_val:.2e})")
print("\nPer-Mutation Stratified Correlation:")
strat_rows = []
for mut, grp in df.groupby("mutation"):
    if len(grp) >= 5 and np.std(grp["ddG_kcal_mol"]) > 0 and np.std(grp["pred_ddg"]) > 0:
        r_m, _ = pearsonr(grp["ddG_kcal_mol"], grp["pred_ddg"])
        strat_rows.append({"mutation": mut, "N": len(grp), "stratified_r": round(r_m, 3)})

strat_df = pd.DataFrame(strat_rows).sort_values("N", ascending=False)
print(strat_df.to_string(index=False))

out_df = pd.DataFrame([{
    "overall_r": round(r_overall, 4),
    "partial_r_controlling_mutation": round(r_partial, 4),
    "p_value": p_val
}])
out_df.to_csv(RESULTS_DIR / "28_partial_correlation_summary.csv", index=False)
print("\nSaved -> results/28_partial_correlation_summary.csv")