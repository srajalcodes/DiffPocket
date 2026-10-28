# DiffPocket: Differential Pocket Encoding for Mutant-Selective Binding Affinity Prediction

Official PyTorch implementation of **DiffPocket**, a structure-aware differential graph neural network designed to predict somatic mutation-induced binding affinity shifts (\(\Delta\Delta G\)) in human kinases without empirical docking or structure relaxation.

## 📌 Architectural Core

Standard structure-based drug–target interaction (DTI) models evaluate wild-type (WT) and mutant complexes independently and subtract their predictions post hoc. Because a point mutation perturbs less than 0.4% of a kinase domain, the unperturbed sequence background can dominate the learned representations. This may cause WT and mutant predictions to become highly correlated (\(R_p > 0.97\)), making their difference difficult to predict reliably.

DiffPocket addresses this challenge through differential pocket encoding:

1. **Shared structural representation:** Maps WT and mutant sequences onto a shared, rigid, SE(3)-invariant Cα contact graph with ESM-2 (650M) node features.
2. **Explicit residue-level differencing:** Computes per-residue difference representations, \(\mathbf{h}_{\mathrm{mut},i} - \mathbf{h}_{\mathrm{WT},i}\), to emphasize mutation-associated changes.
3. **Compound-aware prediction:** Fuses pooled pocket perturbation vectors with 2D chemical descriptors for direct \(\Delta\Delta G\) regression.

The central design principle is to represent the difference between WT and mutant protein states explicitly rather than relying exclusively on the subtraction of two independently predicted affinities.

## 🚀 Installation

### 1. Create the environment

```bash
conda create -n diffpocket python=3.10 -y
conda activate diffpocket
```

### 2. Install PyTorch

The following command installs PyTorch packages for CUDA 12.1. Ensure that your system and GPU drivers are compatible with this configuration.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

From the repository root, run:

```bash
pip install -r requirements.txt
pip install -e .
```

## 🔬 Reproducing Paper Experiments

Experiments use predefined random seeds, with `seed=42` for the canonical split.

Run the following commands from the repository root. The script names and paths below assume the corresponding files are present in the repository.

### 1. Within-Kinase 5-Fold Cross-Validation

**Associated results:** Figure 3 and Table 3

Evaluates generalization to held-out Bemis–Murcko scaffolds across EGFR, KIT, FLT3, and ABL1.

```bash
python experiments/run_within_kinase.py
```

### 2. Loss Objective vs. Architecture Control

**Associated results:** Table 6

Compares DiffPocket with an Independent-GAT baseline trained using a Siamese loss to investigate the contribution of the architecture relative to the training objective.

```bash
python experiments/run_siamese_control.py
```

### 3. Component and Structural Ablations

Investigates the contribution of explicit subtraction and the use of unperturbed rigid coordinates.

```bash
python experiments/run_subtraction_ablation.py
python experiments/run_ca_ablation.py
```

### 4. Pan-Kinase Benchmarks

Evaluates performance on novel chemical scaffolds and in Leave-One-Kinase-Out (LOKO) transfer experiments.

The novel-scaffold setting uses a reported Tanimoto similarity threshold of \(\leq 0.434\).

```bash
python experiments/run_pan_kinase_benchmark.py
```

### 5. Generate Publication Figures

Generate the benchmark parity plots and all-kinase parity plots.

```bash
python figures/plot_benchmark_parity.py
python figures/plot_all_kinases_parity.py
```

## 📊 Benchmark Results

The following results summarize the supplied benchmark values. Pearson correlation is reported as \(R_p\), and RMSE is reported in kcal/mol.

### Within-Kinase Evaluation

| Target | Test Samples (\(N\)) | Independent-GAT (\(R_p\) / RMSE) | DeepDTA (\(R_p\) / RMSE) | DiffPocket (\(R_p\) / RMSE) |
|:---|---:|---:|---:|---:|
| EGFR | 306 | 0.034 / 0.857 | 0.028 / 0.857 | **0.577 / 0.698** |
| KIT | 268 | 0.461 / 1.492 | 0.380 / 1.485 | **0.539 / 1.130** |
| FLT3 | 242 | 0.064 / 0.983 | 0.051 / 0.980 | **0.232 / 0.936** |
| ABL1 | 185 | 0.465 / 1.063 | 0.412 / 0.967 | **0.548 / 0.858** |

Higher Pearson correlation indicates stronger linear agreement with experimental measurements. Lower RMSE indicates smaller prediction errors.

### Generalization Evaluation

| Evaluation Setting | Metric(s) | Independent-GAT | DeepDTA | DiffPocket |
|:---|:---|---:|---:|---:|
| Novel scaffolds (\(\text{Tanimoto} \leq 0.43\)) | Pearson \(R_p\) / second reported metric | 0.122 / 0.359 | 0.230 / 0.401 | **0.597 / 0.639** |
| LOKO transfer (mean) | Pearson \(R_p\) | 0.039 | -0.077 | **0.209** |

**Metric clarification:** The supplied results label the second novel-scaffold metric as AUPR, but the value alone is insufficient to establish that interpretation. Confirm the metric against the original evaluation output before final publication. Also verify that the novel-scaffold threshold is consistently reported as 0.434 or 0.43.

## 🧪 Evaluation Protocol

For reproducible and fair comparisons, the evaluation should specify:

- **Data splits:** Identify the exact training, validation, and test partitions.
- **Scaffold separation:** Confirm that held-out Bemis–Murcko scaffolds do not appear in the corresponding training partition.
- **Baseline comparability:** Evaluate all models on the same test examples and report the same metrics.
- **Random seeds:** Record the seeds used for data splitting, model initialization, and training.
- **Metric definitions:** Document the calculation of Pearson correlation, RMSE, and any additional reported metrics.
- **Generalization settings:** Distinguish within-kinase scaffold evaluation from Leave-One-Kinase-Out transfer.

The benchmark values above are reproduced from the supplied results and should be checked against the final evaluation logs before publication.

## 📁 Repository Structure

The following is an indicative layout based on the experiment commands documented above. Adjust it to match the actual repository.

```text
DiffPocket/
├── experiments/
│   ├── run_within_kinase.py
│   ├── run_siamese_control.py
│   ├── run_subtraction_ablation.py
│   ├── run_ca_ablation.py
│   └── run_pan_kinase_benchmark.py
├── figures/
│   ├── plot_benchmark_parity.py
│   └── plot_all_kinases_parity.py
├── requirements.txt
├── README.md
└── setup.py
```

## 📖 Citation

If you use DiffPocket in your research, please cite the associated work.

The following BibTeX entry is based on the supplied citation information. Confirm the publication venue, year, and final bibliographic details before using it in a published manuscript.

```bibtex
@article{tiwari2026diffpocket,
  title   = {DiffPocket: Differential Pocket Encoding for Mutant-Selective Binding Affinity Prediction},
  author  = {Tiwari, Srajal and Tiwari, Swati and Sharma, Dolly},
  journal = {Research in Computational Molecular Biology (RECOMB)},
  year    = {2027}
}
```

## 📜 License

Add the license governing this repository before distributing or reusing the code. If no license file is present, the repository's reuse terms should not be assumed.

## 🙏 Acknowledgments

DiffPocket builds on ideas from protein language models, graph neural networks, and molecular representation learning for structure-based drug discovery.

## 📬 Contact

For questions, support, or collaboration inquiries, please contact **Srajal Tiwari** at [st735@snu.edu.in](mailto:st735@snu.edu.in).