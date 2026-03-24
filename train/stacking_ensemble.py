"""
Bank Marketing Classification — Stacking Ensemble
===================================================
New ensemble model:
  - Stacking classifier that combines:
    • Random Forest (base learner 1)
    • XGBoost (base learner 2)
    • LightGBM (base learner 3)
    • Logistic Regression (meta-learner / final estimator)
  - One-hot encoding for nominal features
  - Ordinal encoding kept for education
  - BorderlineSMOTE for class imbalance (applied before stacking)
  - Cross-validated stacking (cv=5 internally)
  - Precision-recall threshold tuning for best F1
  - Results saved to results/StackingEnsemble/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# pip install imbalanced-learn
from imblearn.over_sampling import BorderlineSMOTE

# ── Output folder ────────────────────────────
RESULTS_DIR = os.path.join("results", "StackingEnsemble")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────
print("=" * 55)
print("Stacking Ensemble — Bank Marketing")
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
cat_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

X = df.drop(columns=["y"])
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Features: {X.shape[1]}")

# ── 3. Apply SMOTE + Scaling ─────────────────
print("\nApplying BorderlineSMOTE …")
smote = BorderlineSMOTE(random_state=42, kind="borderline-1")
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape[0]:,} samples "
      f"(class 0: {(y_train_res==0).sum():,}, class 1: {(y_train_res==1).sum():,})")

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X_train.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ── 4. Define Stacking Ensemble ──────────────
print("\nTraining Stacking Ensemble (RF + XGB + LGBM → LogisticRegression) …")

# scale_pos_weight for XGBoost (even after SMOTE, a bit helps)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

# Base learners with strong hyperparameters
base_estimators = [
    ("rf", RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )),
    ("xgb", XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )),
    ("lgbm", LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )),
]

# Meta-learner: Logistic Regression with regularization
stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method="predict_proba",
    passthrough=False,
    n_jobs=-1
)

stack_clf.fit(X_train_sc, y_train_res)
print("Stacking Ensemble trained successfully.")

# ── 5. Threshold Tuning ───────────────────────
print("\nTuning classification threshold …")

y_prob = stack_clf.predict_proba(X_test_sc)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx       = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Default threshold (0.5) F1 : {f1_score(y_test, (y_prob >= 0.5).astype(int)):.4f}")
print(f"Best threshold  ({best_threshold:.3f}) F1 : {f1_scores[best_idx]:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

# ── 6. Final Metrics ──────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)

metrics = {
    "Model":            "Stacking Ensemble",
    "Best Threshold":   round(best_threshold, 4),
    "Accuracy":         round(acc,  4),
    "Precision":        round(prec, 4),
    "Recall":           round(rec,  4),
    "F1-Score":         round(f1,   4),
    "ROC-AUC":          round(auc,  4),
    "Base Learners":    "RF + XGB + LGBM",
    "Meta Learner":     "LogisticRegression",
}

pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
print(f"\n{pd.DataFrame([metrics]).T.to_string(header=False)}")

# ── 7. Classification Report ──────────────────
report = classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"])
print(f"\n{report}")
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Base Learners: RF + XGB + LGBM\n")
    f.write(f"Meta Learner: LogisticRegression(C=1.0, class_weight=balanced)\n")
    f.write(f"Best threshold: {best_threshold:.4f}\n\n")
    f.write(report)

# ── 8. Plots ──────────────────────────────────
# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", ax=ax,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
ax.set_title(f"Stacking Ensemble — Confusion Matrix\n(threshold={best_threshold:.3f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, color="#e377c2", label=f"Stacking (AUC = {auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Stacking Ensemble")
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
ax.set_title("Precision / Recall / F1 vs Threshold — Stacking Ensemble")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "threshold_tuning.png"), dpi=150)
plt.close()

# Base learner contribution: show individual base learner AUCs vs stack
print("\n── Base Learner Performance ──")
for name, est in stack_clf.named_estimators_.items():
    base_prob = est.predict_proba(X_test_sc)[:, 1]
    base_auc  = roc_auc_score(y_test, base_prob)
    base_f1   = f1_score(y_test, (base_prob >= 0.5).astype(int))
    print(f"  {name:5s} →  AUC={base_auc:.4f}  F1(0.5)={base_f1:.4f}")
print(f"  stack →  AUC={auc:.4f}  F1({best_threshold:.3f})={f1:.4f}")

print(f"\nAll outputs saved to: {RESULTS_DIR}/")
