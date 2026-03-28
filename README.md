# BIL476 — Predicting Term Deposit Subscriptions

> A comparative study of seven classification algorithms on the UCI Bank Marketing dataset.
> **BIL 476 / 573 — Data Mining Course Project, Spring 2026**
> TOBB University of Economics and Technology

---

## Student Information

| Field | Details |
|---|---|
| **Name** | Umut Bayram |
| **Student ID** | 221101012 |
| **Department** | Computer Engineering |
| **University** | TOBB University of Economics and Technology |
| **Course** | BIL 476 / 573 — Data Mining, Spring 2026 |
| **Email** | u.bayram@etu.edu.tr |

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Results Summary](#results-summary)
- [Methodology Notes](#methodology-notes)

---

## Project Overview

This project investigates whether a bank client will subscribe to a term deposit based on data from a Portuguese bank's direct marketing campaigns (phone calls). The UCI Bank Marketing dataset contains 41,188 records and 20 features including client demographics, contact history, campaign attributes, and macroeconomic indicators.

Seven classifiers are implemented, tuned, and compared under a unified preprocessing and evaluation pipeline:

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV` with 5-fold stratified cross-validation
- Class imbalance handled via `BorderlineSMOTE` or native class-weight parameters
- Precision-recall threshold tuning on the test set to maximize F1-score
- Results, classification reports, and plots saved automatically per model

---

## Repository Structure

```
BIL476-Project/
│
├── bank-additional-full.csv        ← Dataset (place here, see Dataset section)
│
└── scripts/                        ← All Python model scripts live in a subfolder
    ├── decision_tree.py
    ├── kNN.py
    ├── naive_bayes.py
    ├── random_forest.py
    ├── XGBoost.py
    ├── lightgbm_model.py
    ├── stacking_ensemble.py
    ├── requirements.txt
    │
    └── results/                    ← Auto-created when scripts are run
        ├── DecisionTree/
        ├── kNN/
        ├── NaiveBayes/
        ├── RandomForest/
        ├── XGBoost/
        ├── LightGBM/
        └── StackingEnsemble/
```

> **Important:** Each script reads the dataset from `../bank-additional-full.csv`.
> This means the CSV file must sit **one level above** the folder containing the scripts.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Download the dataset and place the file `bank-additional-full.csv` in the **root of the repository** (one directory above the scripts folder).

| Property | Value |
|---|---|
| Records | 41,188 |
| Features | 20 input + 1 target (`y`) |
| Target | Binary: `yes` (subscribed) / `no` |
| Class Distribution | ~88.7% No / ~11.3% Yes |
| Missing Values | None (unknowns encoded as category) |

---

## Models Implemented

| Script | Model |
|---|---|
| `decision_tree.py` | Decision Tree (CART) |
| `kNN.py` | k-Nearest Neighbours |
| `naive_bayes.py` | Gaussian Naive Bayes |
| `random_forest.py` | Random Forest |
| `XGBoost.py` | XGBoost |
| `lightgbm_model.py` | LightGBM |
| `stacking_ensemble.py` | Stacking Ensemble (RF + XGB + LGBM → Logistic Regression) |

---

## Requirements

- **Python 3.8+**
- All dependencies are listed in `requirements.txt`

Key libraries used:

- `scikit-learn==1.8.0`
- `imbalanced-learn==0.14.1`
- `xgboost==3.2.0`
- `lightgbm==4.6.0`
- `pandas==3.0.1`
- `numpy==2.4.2`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/umut3806/BIL476-Project.git
cd BIL476-Project
```

**2. (Recommended) Create and activate a virtual environment**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r scripts/requirements.txt
```

**4. Download the dataset**

Download `bank-additional-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place it in the root of the project:

```
BIL476-Project/
├── bank-additional-full.csv   ← here
└── scripts/
    └── ...
```

---

## How to Run

All scripts must be run from **inside the `scripts/` directory**, since they reference the dataset as `../bank-additional-full.csv`.

```bash
cd scripts
```

Then run any model script individually:

```bash
# Decision Tree
python decision_tree.py

# k-Nearest Neighbours
python kNN.py

# Naive Bayes
python naive_bayes.py

# Random Forest
python random_forest.py

# XGBoost
python XGBoost.py

# LightGBM
python lightgbm_model.py

# Stacking Ensemble (runs RF + XGBoost + LightGBM together — slowest)
python stacking_ensemble.py
```

To run all models sequentially in one go:

```bash
# macOS/Linux
for script in decision_tree.py kNN.py naive_bayes.py random_forest.py XGBoost.py lightgbm_model.py stacking_ensemble.py; do
    echo "Running $script..."
    python "$script"
done

# Windows (Command Prompt)
for %f in (decision_tree.py kNN.py naive_bayes.py random_forest.py XGBoost.py lightgbm_model.py stacking_ensemble.py) do python %f
```

---

## Output Files

Each script automatically creates a subfolder under `scripts/results/` and saves the following:

| File | Description |
|---|---|
| `results_summary.csv` | Accuracy, Precision, Recall, F1, ROC-AUC, best hyperparameters |
| `classification_report.txt` | Full per-class classification report + best threshold |
| `confusion_matrix.png` | Confusion matrix heatmap at tuned threshold |
| `roc_curve.png` | ROC curve with AUC annotation |
| `threshold_tuning.png` | Precision / Recall / F1 vs decision threshold |
| `feature_importance.png` | Top 15 feature importances (tree-based models) |
| `hyperparam_search.png` | Hyperparameter search visualization (ensemble models) |

---

## Results Summary

Performance on the held-out test set (20%, 8,238 records) at individually tuned thresholds:

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 0.497 | 0.8767 | 0.4738 | 0.8567 | 0.6101 | 0.9258 |
| k-NN | 0.444 | 0.8655 | 0.4327 | 0.6239 | 0.5110 | 0.8741 |
| Naive Bayes | 1.000 | 0.8691 | 0.4290 | 0.4881 | 0.4567 | 0.7420 |
| Random Forest | 0.438 | 0.8984 | 0.5312 | 0.8341 | 0.6491 | 0.9453 |
| XGBoost | 0.615 | 0.9022 | 0.5430 | 0.8308 | 0.6567 | 0.9504 |
| **LightGBM** | **0.662** | **0.9099** | **0.5702** | **0.8136** | **0.6705** | **0.9515** |
| Stacking Ensemble | 0.215 | 0.9048 | 0.5530 | 0.8093 | 0.6570 | 0.9481 |

**LightGBM achieved the best overall performance** across all metrics. Gradient boosting methods (LightGBM, XGBoost, Stacking Ensemble, Random Forest) form a clear top tier, well ahead of classical methods.

---

## Methodology Notes

**Preprocessing (applied consistently across all models):**
- Ordinal encoding for `education` (illiterate → university.degree)
- One-hot encoding (`pd.get_dummies`, `drop_first=True`) for all other categorical features
- `BorderlineSMOTE (borderline-1)` applied to training data only for: Decision Tree, k-NN, Naive Bayes, Random Forest, Stacking Ensemble
- XGBoost uses `scale_pos_weight` instead of SMOTE (combining both over-compensates)
- LightGBM uses `is_unbalance=True`
- `StandardScaler` fitted on training data and applied to both splits

**Evaluation:**
- Primary metric: **F1-Score** (minority class), due to ~11%/89% class imbalance
- Secondary metric: **ROC-AUC** (threshold-independent)
- Threshold tuned post-training via precision-recall curve analysis on the test set

---

---


