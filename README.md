# Deep Learning-Based Genomic Prediction with Multi-Scale Attention Pooling and LD-Aware Encoding

This repository contains a collaborative implementation and evaluation of an EBMGP-style deep genomic-prediction pipeline. The project combines Elastic Net-selected SNP subsets, linkage-disequilibrium-aware tokenization, BERT-style embeddings, a 1D convolutional backbone, and alternative pooling strategies for quantitative phenotype prediction.

> This is Tasfia Zaman's public fork of the collaborative project developed with Md. Rezaur Rahman Bhuiyan and Jareen Tasneem Khondaker. The upstream repository is [`killerexcipio/443_proj`](https://github.com/killerexcipio/443_proj).

[Read the term project report](Bio_Term_Project_Report_Our_Copy.pdf)

## Project scope

The project adapts and empirically evaluates an EBMGP-style workflow across **three species and eleven quantitative traits**. It is an implementation and reproduction project; the original EBMGP framework and its methodological references are described in the report.

The workflow starts from Elastic Net (EN)-selected SNP subsets, constructs LD-aware categorical SNP representations, and combines BERT-style token, position, and segment/type embeddings. A 1D convolutional regression backbone compares alternative pooling operators. Evaluation uses 5-fold cross-validation and matched classical and machine-learning baselines.

### Collaborative contributions

The team:

- Implemented the downstream pipeline from EN-selected SNP subsets through LD-aware tokenization, convolution/pooling, and scalar phenotype regression.
- Evaluated the pipeline across three species and eleven traits.
- Reproduced LD-threshold ablation and pooling-strategy comparison at a primary SNP budget of `T = 5000`.
- Benchmarked against classical/ML baselines using the same folds and selected feature subsets.

### Processed-data dependency

The project uses processed genotype/phenotype files and precomputed Elastic Net SNP rankings associated with the original EBMGP setup. The team did **not** rerun the complete raw-data quality-control, imputation, and EN feature-selection workflow or generate those upstream processed datasets and rankings. The implemented work begins from the selected/processed inputs and covers downstream representation, modeling, ablation, evaluation, and benchmarking.

These external data and ranking files are not bundled in this repository. Reproducing the reported experiments requires the same processed inputs; the report describes their provenance.

## Datasets

| Dataset | Samples | SNPs after upstream QC | Quantitative traits |
|---|---:|---:|---|
| Rice | 413 | 36,901 | SW, FLW, PH, AC, SNPP |
| Sorghum | 451 | 36,468 | MO, YLD, HT |
| Holstein Bulls | 1,508 | 44,074 | MS, NMSP, VE |

Sample sizes and post-QC SNP counts follow the EBMGP dataset summary reproduced in the report. There are **11 traits across 3 species**. Each sample is an individual, and each prediction task targets one standardized quantitative phenotype.

## Methods

### SNP selection and input preparation

The notebook reads precomputed EN ranking files, ranks markers by the supplied `cs` values, selects the Top-T subset, and restores marker order for sequence construction. The primary experimental budget is **`T = 5000`**. This applies existing rankings rather than fitting the complete upstream EN selection pipeline from raw genomic data.

The downstream dataset loader also mean-fills missing phenotype values and standardizes the target before constructing the fold datasets, as implemented in the notebook.

### LD-aware representation

Genotype states use `H` (major allele), `M` (heterozygous), and `L` (minor allele). Adjacent-marker LD is computed as an R²-style statistic from genotype values. The implementation assigns:

| LD/boundary label | Convention in this notebook |
|---|---|
| `J` | Adjacent-marker R² is at or above the LD threshold |
| `Y` | Adjacent-marker R² is below the threshold |
| `N` | Chromosome boundary |

Genotype state and LD label combine into categorical SNP tokens:

```text
HJ  HY  HN  LJ  LY  LN  MJ  MY  MN
```

The LD-label stream also supplies segment/type IDs. This documents the convention used by this implementation.

### BERT-style embeddings and convolutional regression

The model uses the BERT embedding structure: token embeddings, positional embeddings, and segment/type embeddings based on the LD-label stream are combined into a dense representation. These embeddings are randomly initialized and trained within the regression pipeline. No pretrained BERT language model or pretrained sentence transformer is used.

The sequence passes through five 1D convolutional blocks, interleaved with pooling and dropout. Adaptive average pooling reduces the final sequence representation, and a linear head predicts one scalar phenotype.

| Architecture setting | Value |
|---|---|
| SNP budget / maximum positions | 5000 |
| Token vocabulary | 9 SNP tokens + padding |
| Type vocabulary | 3 LD labels + padding |
| Embedding dimension | 64 |
| Convolutional kernels | 30, 3, 30, 3, 30 |
| Convolutional stride | 2 in each block |
| Pool sizes after successive blocks | 1, 2, 3, 2, 3 |
| Dropout after pooling | 0.3 |
| Prediction head | `AdaptiveAvgPool1d(1)` followed by `Linear(64, 1)` |

### Pooling and LD-threshold ablations

Four pooling choices provide alternative aggregation/downsampling strategies within the convolutional architecture:

| Choice | Role |
|---|---|
| MAP | Learned attention-style local feature aggregation |
| AVG | Average pooling |
| MAX | Max pooling |
| LIP | Local importance pooling with learned importance weights |

The pooling comparison uses `T = 5000` and LD threshold `0.8`. MAP is not uniformly the best-performing choice; performance varies by trait and species.

The LD-threshold ablation tests **`0.2`, `0.4`, `0.6`, and `0.8`** with MAP pooling at `T = 5000`, measuring sensitivity of predictive performance to the LD discretization threshold.

## Training and validation

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Loss | MAE / L1 loss |
| Learning rate | `5e-4` |
| Weight decay | `1e-5` |
| Batch size | 32 |
| Epochs | Up to 100 |
| Scheduler | `CosineAnnealingLR`, stepped during training |
| Cross-validation | 5-fold `KFold`, `shuffle=True`, `random_state=27` |

In the main ablation and matched-baseline workflow, indices `0–4` identify the held-out folds and corresponding EN ranking files. The main deep-training wrapper sets the initialization seed to `3407 + fold_index` and uses early stopping based on validation Pearson correlation (patience 15, minimum improvement `1e-4`). Metrics are aggregated across the five held-out folds.

Later notebook cells contain additional exploratory benchmarks and a separate training wrapper for figures. Their settings are retained in the notebook; the table above describes the main reported workflow.

## Baselines and evaluation metrics

The matched baseline comparison uses the same EN-selected feature subsets and fold splits as the deep model, with numerical `H/M/L → 0/1/2` genotype encoding:

| Implemented estimator | Interpretation |
|---|---|
| `RidgeCV` | Practical GBLUP-style linear-shrinkage proxy |
| `BayesianRidge` | Bayesian shrinkage proxy |
| `HistGradientBoostingRegressor` | Nonlinear histogram-gradient-boosting baseline |

Some saved notebook labels use the family names GBLUP, BayesB, and LightGBM. The estimators above are the actual implementations; those labels do not denote exact implementations of those methods.

**Pearson correlation** is the primary predictive-association metric. The project also uses **RMSE**, **R²**, and **MAE**, with MAE serving as the training loss.

## Results summary

Across the tested setup, pooling strategy produced larger performance differences than the exact LD threshold, while performance was comparatively stable across the tested LD thresholds.

The EBMGP-style model was particularly competitive on the rice traits. Classical linear/shrinkage baselines remained strong on sorghum and Holstein-bull traits, illustrating that the deep architecture did not uniformly dominate across datasets.

Detailed experimental results, ablations, figures, and numerical tables are available in the [term project report](Bio_Term_Project_Report_Our_Copy.pdf).

## Repository structure

```text
ld-aware-genomic-prediction/
├── Bio_Term_Project_Report_Our_Copy.pdf
├── ld_aware_genomic_prediction.ipynb
└── README.md
```

## Installation and running

Clone this fork:

```bash
git clone https://github.com/kitkiti/ld-aware-genomic-prediction.git
cd ld-aware-genomic-prediction
```

Install the notebook dependencies in your Python environment:

```bash
pip install jupyter numpy pandas scikit-learn scipy torch torchmetrics transformers einops matplotlib seaborn tqdm
```

The additional exploratory tree-benchmark cells also require `xgboost`:

```bash
pip install xgboost
```

Prepare the external processed inputs relative to the repository root:

```text
data/
├── rice_pheno.csv
├── ricerawgeno.csv
├── sorghum_pheno.csv
├── sorghumrawgeno.csv
├── bulls_pheno.csv
└── bullsrawgeno.csv
EN/
├── rice/
├── sorghum/
└── bulls/
```

The genotype filenames above are those expected by the notebook; they refer to the supplied processed inputs. EN files follow the pattern `EN/{species}/T5000{trait}{fold}.csv`, for example `EN/rice/T5000SW0.csv` for fold 0. These are ranking inputs with `cs` values, not newly fitted feature-selection outputs.

From the repository root, open:

```bash
jupyter notebook ld_aware_genomic_prediction.ipynb
```

Check the data paths and trait-column ordering in the configuration against your processed files before running the experiment cells. The auxiliary loader's `/path/to/data/{species}_pheno.csv` fallback is a placeholder for an alternate processed-data directory; configure it only if needed. The main experiments use the relative `data/` and `EN/` paths above. A CUDA-capable GPU can accelerate training; the main configuration falls back to CPU when CUDA is unavailable.

The notebook retains experiment outputs and includes ablation runs, matched baselines, and figure-generation cells. Full reproduction depends on the external data, precomputed rankings, and compatible package versions; this repository does not include a pinned environment or a complete raw-data preprocessing workflow.

## Authors

Md. Rezaur Rahman Bhuiyan  
Jareen Tasneem Khondaker  
Tasfia Zaman

## Acknowledgment

This project was developed as part of a computational biology / genomic prediction assignment and adapts ideas from the EBMGP genomic-prediction framework. The implementation focuses on reproducing and evaluating selected components of that workflow, including LD-aware SNP representation, convolutional prediction, pooling ablations, and matched baseline comparisons.

## License

No license is currently specified for this repository. Please contact the project authors before reusing or redistributing the code.
