# Field-Theoretic Inferencing for Overlapping Clustering

Rust implementation of the numerical overlapping-clustering methods used in the paper **Field-Theoretic Inferencing for Overlapping Clustering**.

The repository contains:

- Proposed QFT
- OKM
- NEO-style
- FCM-Overlap
- GMM-Overlap
- SVD-OSC-style
- OC-DD-style
- multi-label ARFF and CSV loaders
- the preprocessing rules used in the paper
- ONMI, example-based F1, membership Jaccard, exact membership-cardinality recovery, membership-cardinality MAE, mean membership count, and wall-clock timing
- one demo command that runs all methods

There are no dot-prefixed files or directories in this repository.

## Build

Install a recent Rust toolchain, then run:

```bash
cargo build --release
```

## Demo: run every method

The demo creates a synthetic overlapping-cluster problem, standardizes the features, runs Proposed QFT and every baseline, and prints the evaluation table.

```bash
cargo run --release -- demo
```

A larger demo can be requested directly:

```bash
cargo run --release -- demo --n 500 --d 20 --k 8 --seed 7
```

## Run all methods on a MULAN-style ARFF file

The ARFF reader supports dense and sparse numeric data and treats the final `label-count` attributes as the held-out binary labels.

Emotions or Yeast style preprocessing:

```bash
cargo run --release -- arff \
  --path data/emotions.arff \
  --label-count 6 \
  --preprocess standard \
  --method all
```

Enron style preprocessing:

```bash
cargo run --release -- arff \
  --path data/enron.arff \
  --label-count 53 \
  --preprocess idf-l2 \
  --method all
```

Run only Proposed QFT:

```bash
cargo run --release -- arff \
  --path data/yeast.arff \
  --label-count 14 \
  --preprocess standard \
  --method qft
```

Available method names are:

```text
all
qft
okm
neo
fcm
gmm
svd
ocdd
```

## Run on CSV files

Features and labels can be supplied as separate numeric CSV files. Binary label values greater than `0.5` are treated as active memberships.

```bash
cargo run --release -- csv \
  --features data/features.csv \
  --labels data/labels.csv \
  --clusters 6 \
  --preprocess standard \
  --method all
```

If the CSV files contain a header row, add:

```text
--headers
```

A feature-only CSV can also be clustered by omitting `--labels`; in that case the program prints mean membership count and runtime but not label-based metrics.

## Proposed QFT options

The field-theoretic solver exposes the Hamiltonian and numerical iteration parameters on the command line:

```text
--alpha <value>
--beta <value>
--mu <value>
--rho <value>
--damping <value>
--qft-max-iter <integer>
--qft-tolerance <value>
```

The default numerical values are:

```text
alpha = 1.0
beta = 1.0
mu = 0.5
rho = 1.0
damping = 0.5
qft-max-iter = 250
qft-tolerance = 1e-7
```

The implementation uses the second-order diagonal field-theoretic closure. It starts from K-means++ centroids, constructs the unnormalized occupation field

```text
m_ik = exp( -||x_i-c_k||^2 / (2 tau^2) )
```

with `tau` equal to the median nearest-centroid distance, iterates the fluctuation-corrected occupation equations, applies damping, then hardens at `m_ik >= 0.5`. If a row is empty after hardening, its maximum-occupation cluster is activated.

## Baseline implementations

The baseline modules implement the measured variants described in the paper rather than replacing them by unrelated library defaults.

- `src/baselines/okm.rs`
- `src/baselines/neo.rs`
- `src/baselines/fcm.rs`
- `src/baselines/gmm.rs`
- `src/baselines/svd_osc.rs`
- `src/baselines/ocdd.rs`
- `src/qft.rs`

The entry point is `src/main.rs`; shared evaluation code is in `src/metrics.rs`.

# Tables from the paper

## Table 1. Dominance/comparison relations for the QFT hierarchy

| Comparator | Quantity | QFT relation |
|---|---|---|
| Factorized/blocked VB | Cross-block connected covariance | One higher exact interaction order; Theorem 4 |
| Gaussian EP/ADF | First non-Gaussian connected cumulant | Explicit cubic graph; Theorem 5 |
| Laplace | Independent hbar loop order | Two-loop and higher terms; Theorem 6 |
| Bethe/BP/GBP | First omitted generalized-loop coefficient | Complete perturbative coefficient; Corollary 1 |
| TAP/Plefka | Interaction order | General order theorem, with pairwise second-order special case; Corollaries 2 and 3 |
| Gibbs/hybrid MCMC | Finite-sample MSE | Joint-limit MSE dominance; Theorem 7 |
| SMC | Finite-particle MSE | Joint-limit MSE dominance; Corollary 4 |
| Finite closures | Selected SD/Ward residuals | Higher-order exact-identity satisfaction; Theorem 8 |

## Table 2. Data sets used for overlapping-clustering evaluation

| Data set | n | d | K | Label cardinality | Label density |
|---|---:|---:|---:|---:|---:|
| Emotions | 593 | 72 | 6 | 1.8685 | 0.3114 |
| Yeast | 2417 | 103 | 14 | 4.2371 | 0.3026 |
| Enron | 1702 | 1001 | 53 | 3.3784 | 0.0637 |

## Table 3. Comparison implementations and hardening rules

| Method | Implementation used |
|---|---|
| OKM | K-means initialization; at most five active memberships per observation; a new membership is accepted only if the mean-prototype reconstruction error decreases by more than `1e-8`; 20 outer iterations; centroid least-squares system regularized by `1e-6 I`. |
| NEO-style | Nearest-cluster assignment plus a fixed global budget of `1.5 n` additional point-cluster assignments chosen by smallest unassigned squared distance; 30 outer iterations. |
| FCM-Overlap | Fuzzy c-means with exponent `m=2`, K-means initialization, and convergence tolerance `1e-5`. A cluster is activated when its fuzzy membership is at least `0.8` times the row maximum. |
| GMM-Overlap | Diagonal-covariance Gaussian mixture, one initialization, covariance regularizer `1e-5`, maximum 100 EM iterations. Components with posterior responsibility at least `0.2` times the row maximum are activated. |
| SVD-OSC-style | Cosine-affinity matrix from row-normalized features, negative affinities truncated to zero, symmetrized top-20-neighbour sparsification, randomized rank-K truncated SVD with five power iterations, absolute embedding, and a `0.5`-of-row-maximum activation threshold. |
| OC-DD-style | Euclidean 15-neighbour distances; bandwidth equal to the median 15th-neighbour distance; local Gaussian density; K density-distance attractors chosen greedily; activation score equal to attractor density times `exp[-(d/h)^2]`; threshold `0.55` of the row maximum. |

## Table 4. ONMI higher is better

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 0.0861 | 0.0122 | 0.0207 |
| NEO-style | 0.0946 | 0.0249 | 0.0834 |
| FCM-Overlap | 0.0000 | 0.0000 | 0.0000 |
| GMM-Overlap | 0.0906 | 0.0196 | 0.0190 |
| SVD-OSC-style | 0.1148 | 0.0102 | 0.0162 |
| OC-DD-style | 0.0651 | 0.0107 | 0.0483 |
| **Proposed QFT** | **0.1283** | **0.0300** | **0.0952** |

## Table 5. Example-based F1 higher is better

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 0.4702 | 0.3969 | 0.2490 |
| NEO-style | 0.4568 | 0.2706 | 0.1790 |
| FCM-Overlap | 0.4638 | 0.4534 | 0.1185 |
| GMM-Overlap | 0.3999 | 0.1760 | 0.1351 |
| SVD-OSC-style | 0.4700 | 0.3132 | 0.2631 |
| OC-DD-style | 0.4502 | 0.4136 | 0.3843 |
| **Proposed QFT** | **0.4911** | **0.4720** | **0.4092** |

## Table 6. Membership Jaccard higher is better

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 0.3610 | 0.2686 | 0.1664 |
| NEO-style | 0.3500 | 0.1906 | 0.1219 |
| FCM-Overlap | 0.3114 | 0.3026 | 0.0637 |
| GMM-Overlap | 0.3184 | 0.1114 | 0.0900 |
| SVD-OSC-style | 0.3819 | 0.2102 | 0.1726 |
| OC-DD-style | 0.3499 | 0.2782 | 0.2528 |
| **Proposed QFT** | **0.4027** | **0.3260** | **0.2734** |

## Table 7. Exact membership-cardinality recovery higher is better

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 0.3339 | 0.1200 | 0.1716 |
| NEO-style | 0.1990 | 0.0505 | 0.1240 |
| FCM-Overlap | 0.0000 | 0.0000 | 0.0000 |
| GMM-Overlap | 0.3035 | 0.0174 | 0.1152 |
| SVD-OSC-style | 0.3862 | 0.1328 | 0.1533 |
| OC-DD-style | 0.3170 | 0.0844 | 0.0429 |
| **Proposed QFT** | **0.4125** | **0.1543** | **0.1935** |

## Table 8. Membership-cardinality MAE lower is better

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 0.8449 | 1.9160 | 1.7303 |
| NEO-style | 1.6071 | 2.6181 | 3.3273 |
| FCM-Overlap | 4.1315 | 9.7629 | 49.6216 |
| GMM-Overlap | 0.8634 | 3.2180 | 2.3784 |
| SVD-OSC-style | 0.7639 | 2.2019 | 2.5053 |
| OC-DD-style | 0.9680 | 3.4030 | 4.3390 |
| **Proposed QFT** | **0.7135** | **1.7648** | **1.5924** |

## Table 9. Predicted mean number of memberships

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| Ground truth | 1.8685 | 4.2371 | 3.3784 |
| OKM | 1.9342 | 3.4100 | 2.7197 |
| NEO-style | 2.5008 | 2.5002 | 2.5000 |
| FCM-Overlap | 6.0000 | 14.0000 | 53.0000 |
| GMM-Overlap | 1.0084 | 1.0207 | 1.0000 |
| SVD-OSC-style | 1.5430 | 2.5681 | 4.1751 |
| OC-DD-style | 2.0506 | 5.8734 | 7.3425 |
| Proposed QFT | 1.8695 | 4.2250 | 3.3820 |

## Table 10. Wall-clock clustering time in seconds lower is better

The paper reports baseline and QFT rows from separate execution environments.

| Method | Emotions | Yeast | Enron |
|---|---:|---:|---:|
| OKM | 1.040 | 8.610 | 19.651 |
| NEO-style | 0.032 | 0.354 | 1.410 |
| FCM-Overlap | 0.029 | **0.202** | 1.013 |
| GMM-Overlap | **0.018** | 0.819 | 1.844 |
| SVD-OSC-style | 0.166 | 1.386 | 1.056 |
| OC-DD-style | 0.129 | 0.225 | **0.253** |
| Proposed QFT | 0.050 | 0.300 | 0.465 |

## Source layout

```text
Cargo.toml
README.md
src
  main.rs
  lib.rs
  data.rs
  metrics.rs
  preprocess.rs
  qft.rs
  synthetic.rs
  utils.rs
  baselines
    mod.rs
    okm.rs
    neo.rs
    fcm.rs
    gmm.rs
    svd_osc.rs
    ocdd.rs
```
