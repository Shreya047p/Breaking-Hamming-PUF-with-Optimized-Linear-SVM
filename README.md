# Breaking the Hamming PUF with Optimized Linear SVM

**CS771: Introduction to Machine Learning, IIT Kanpur — Assignment 1**
Team Deep_Coders (Group 27): Neeraj Kumar, Shubhangam Raj, Shreya Pazare, Kavita Kumari, Arkajyoti Santra

Modeled a Hamming PUF's challenge-response behavior via its additive delay model and trained an optimized linear SVM to predict unseen responses with high accuracy, exposing its vulnerability to ML-based modeling attacks and motivating non-linear constructions (e.g., XOR-PUFs) for stronger security.

## Problem

A **Physical Unclonable Function (PUF)** exploits manufacturing randomness to produce device-specific responses to challenges. The **Hamming PUF** studied here uses a 32-bit secret word `s ∈ {0,1}^32` and a secret threshold `t`. Given a 32-bit challenge `c`, it:

1. Computes `m = c ⊕ s`
2. Computes partial Hamming weights `h_e` (even-indexed bits) and `h_o` (odd-indexed bits)
3. Returns response `r(c) = 1[h_e · h_o ≥ t]`

The goal: learn a **linear classifier** from challenge-response pairs (CRPs) that predicts `r(c)` accurately — i.e., *break* the PUF despite it being designed to be unclonable.

## Approach

- **Feature map derivation:** Since `h_e` and `h_o` are linear in the even/odd challenge bits respectively, the product `h_e · h_o` (and hence the decision boundary) is a **degree-2 polynomial in `c`** containing only even–odd cross terms. This yields a minimal, PUF-instance-independent feature map:

  ```
  φ(c) = [c_0, ..., c_31, c_i·c_j for all even i, odd j]  ∈ R^288
  ```

  (32 raw bits + 16×16 = 256 even-odd cross terms), which is provably sufficient to make the PUF linearly separable — far smaller than the exponential `O(2^n)` blow-up of enumerating all bit subsets, and smaller than the naive degree-2 polynomial expansion (528 features).

- **Classifier:** `LinearSVC` (hinge loss) trained on the 288-dimensional mapped features, with hyperparameters tuned via controlled experiments (loss type, `C`, `tol`) and benchmarked against `LogisticRegression`.

- **Implementation:** `my_map(X)` — fully vectorized NumPy transform (32-dim binary input → 288-dim mapped output) using broadcasting outer products, running in ~0.03s on 7500 samples. `my_params(...)` — returns fixed, pre-optimized hyperparameters for instant training.

## Results

| Dataset | Features | Train Acc | Test Acc | Train Time |
|---|---|---|---|---|
| Public | 288 | 99.90% | **99.80%** | 1.85s |
| Secret | 288 | 99.90% | **99.80%** | 1.85s |
| Dummy baseline (raw 32-dim) | 32 | 95.68% | 94.64% | 0.03s |

**+5.16% test accuracy improvement** over the raw-feature baseline, driven almost entirely by the engineered feature map (confirmed to be independent of hyperparameter choice — see report).

### Key findings

- The even-odd 288-dim feature map alone accounts for ≈+5% accuracy gain over raw features, regardless of `C`.
- `hinge` loss outperforms `squared_hinge` by 0.4% test accuracy (though `squared_hinge` trains 9× faster).
- `C = 2.0` gives the best accuracy/generalization tradeoff; test accuracy is insensitive to solver tolerance (`tol`).
- The 288-dim map is highly sample-efficient: only **2000 CRPs** are needed for ≥95% accuracy, and **5000 CRPs** for ≥99% accuracy (vs. more samples required for the naive 528-dim map).

Full derivations, all hyperparameter sweep plots, and the learning-curve analysis are in [`report/CS771_Assignment1_Report.pdf`](report/CS771_Assignment1_Report.pdf).

## Repo structure

```
hamming-puf-svm-attack/
├── data/              # CRP datasets (add your train/test .npz or .csv here)
├── src/
│   └── submit.py      # my_map() feature transform + my_params() hyperparameters
├── report/
│   └── CS771_Assignment1_Report.pdf
├── results/           # accuracy/learning-curve plots, logs
├── requirements.txt
└── README.md
```

## Usage

```python
from src.submit import my_map, my_params
from sklearn.svm import LinearSVC
import numpy as np

X_train = np.load("data/train_X.npy")   # (n, 32) binary challenge matrix
y_train = np.load("data/train_y.npy")   # (n,) responses

X_map = my_map(X_train)                 # -> (n, 288)
params = my_params(X_map, X_train, y_train)

clf = LinearSVC(**params)
clf.fit(X_map, y_train)
```

## Setup

```bash
pip install -r requirements.txt
```

## Course credit

Project for **CS771: Introduction to Machine Learning**, IIT Kanpur (2024). Shared here for portfolio purposes.
