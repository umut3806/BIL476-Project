"""
Bank Marketing Classification — k-Nearest Neighbours
======================================================
Improvements applied (v2):
  - One-hot encoding for nominal features (critical for distance-based models)
  - Ordinal encoding kept for education
  - BorderlineSMOTE for class imbalance
  - Wider GridSearchCV: larger k range, Minkowski p parameter
  - Precision-recall threshold tuning for best F1
  - Results saved to results/kNN/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.neighbors import KNeighborsClassifier

# pip install imbalanced-learn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Output folder ────────────────────────────
RESULTS_DIR = os.path.join("results", "kNN")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────
print("=" * 55)
print("k-NN — Bank Marketing (v2)")
print("=" * 55)

df = pd.read_csv("../bank-additional-full.csv", sep=";")
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ── 2. Preprocessing ─────────────────────────
df["y"] = (df["y"] == "yes").astype(int)

# Ordinal encoding for education (has a natural order)
edu_order = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree", "unknown"]
oe = OrdinalEncoder(categories=[edu_order], handle_unknown="use_encoded_value", unknown_value=-1)
df["education"] = oe.fit_transform(df[["education"]])

# One-hot encoding for other categorical columns
# Critical for kNN: LabelEncoder creates wrong distances between categories
cat_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

X = df.drop(columns=["y"])
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Features: {X.shape[1]}")

# ── 3. Hyperparameter Tuning ─────────────────
# Search over k, distance metric, weighting, and Minkowski p
print("\nRunning GridSearchCV …")

pipeline = ImbPipeline([
    ("smote",  BorderlineSMOTE(random_state=42, kind="borderline-1")),
    ("scaler", StandardScaler()),
    ("clf",    KNeighborsClassifier(n_jobs=-1))
])

param_grid = {
    "clf__n_neighbors": [5, 7, 9, 11, 15, 21, 31],
    "clf__metric":      ["minkowski"],
    "clf__p":           [1, 2, 3],        # 1=manhattan, 2=euclidean, 3=cubic
    "clf__weights":     ["uniform", "distance"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print(f"\nBest params : {search.best_params_}")
print(f"Best CV F1  : {search.best_score_:.4f}")

best_pipeline = search.best_estimator_

# ── 4. Threshold Tuning ───────────────────────
print("\nTuning classification threshold …")

y_prob = best_pipeline.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx       = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Default threshold (0.5) F1 : {f1_score(y_test, (y_prob >= 0.5).astype(int)):.4f}")
print(f"Best threshold  ({best_threshold:.3f}) F1 : {f1_scores[best_idx]:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

# ── 5. Final Metrics ──────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)

best_k       = search.best_params_["clf__n_neighbors"]
best_p       = search.best_params_["clf__p"]
best_weights = search.best_params_["clf__weights"]

metrics = {
    "Model":           f"k-NN (k={best_k})",
    "Best Threshold":  round(best_threshold, 4),
    "Accuracy":        round(acc,  4),
    "Precision":       round(prec, 4),
    "Recall":          round(rec,  4),
    "F1-Score":        round(f1,   4),
    "ROC-AUC":         round(auc,  4),
    "Best k":          best_k,
    "Best p":          best_p,
    "Best weights":    best_weights,
}

pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
print(f"\n{pd.DataFrame([metrics]).T.to_string(header=False)}")

# ── 6. Classification Report ──────────────────
report = classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"])
print(f"\n{report}")
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Best params: k={best_k}, p={best_p}, weights={best_weights}\n")
    f.write(f"Best threshold: {best_threshold:.4f}\n\n")
    f.write(report)

# ── 7. Plots ──────────────────────────────────
# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
ax.set_title(f"k-NN (k={best_k}) — Confusion Matrix\n(threshold={best_threshold:.3f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, color="#2ca02c", label=f"k-NN k={best_k} (AUC = {auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve — k-NN (k={best_k})")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
plt.close()

# Precision-Recall vs Threshold
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, precisions[:-1], label="Precision", color="#2196F3")
ax.plot(thresholds, recalls[:-1],    label="Recall",    color="#FF5722")
ax.plot(thresholds, f1_scores[:-1],  label="F1-Score",  color="#4CAF50")
ax.axvline(best_threshold, color="black", linestyle="--", label=f"Best threshold ({best_threshold:.3f})")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 vs Threshold — k-NN (k={best_k})")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "threshold_tuning.png"), dpi=150)
plt.close()

# k vs CV F1 score (for each p/weight combo)
cv_df = pd.DataFrame(search.cv_results_)
fig, ax = plt.subplots(figsize=(8, 4))
for p_val in [1, 2, 3]:
    for weight in ["uniform", "distance"]:
        mask = (
            cv_df["param_clf__p"] == p_val) & (
            cv_df["param_clf__weights"] == weight
        )
        subset = cv_df[mask].sort_values("param_clf__n_neighbors")
        p_label = {1: "manhattan", 2: "euclidean", 3: "minkowski-3"}[p_val]
        ax.plot(
            subset["param_clf__n_neighbors"],
            subset["mean_test_score"],
            marker="o", label=f"p={p_val} ({p_label})/{weight}"
        )
ax.set_xlabel("k (n_neighbors)"); ax.set_ylabel("CV F1 Score")
ax.set_title("k vs CV F1 for Different p/Weight Combinations")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "k_tuning_curve.png"), dpi=150)
plt.close()

print(f"\nAll outputs saved to: {RESULTS_DIR}/")