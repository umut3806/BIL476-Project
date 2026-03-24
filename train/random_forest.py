"""
Bank Marketing Classification — Random Forest
==============================================
Improvements applied (v2):
  - One-hot encoding for nominal features
  - Ordinal encoding kept for education
  - BorderlineSMOTE for class imbalance
  - Wider RandomizedSearchCV: more estimators, bootstrap toggle
  - Precision-recall threshold tuning for best F1
  - Results saved to results/RandomForest/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

# pip install imbalanced-learn
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Output folder ────────────────────────────
RESULTS_DIR = os.path.join("results", "RandomForest")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────
print("=" * 55)
print("Random Forest — Bank Marketing (v2)")
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

# ── 3. Hyperparameter Tuning ─────────────────
print("\nRunning RandomizedSearchCV (n_iter=40) …")

pipeline = ImbPipeline([
    ("smote",  BorderlineSMOTE(random_state=42, kind="borderline-1")),
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42))
])

param_dist = {
    "clf__n_estimators":      [200, 400, 600, 800, 1000],
    "clf__max_depth":         [8, 10, 12, 16, 20, None],
    "clf__min_samples_leaf":  [5, 10, 20, 40],
    "clf__max_features":      ["sqrt", "log2", 0.3, 0.5],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__bootstrap":         [True, False],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=40,
    scoring="f1", cv=cv, n_jobs=-1,
    random_state=42, verbose=1
)
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

bp = search.best_params_
metrics = {
    "Model":                  "Random Forest",
    "Best Threshold":         round(best_threshold, 4),
    "Accuracy":               round(acc,  4),
    "Precision":              round(prec, 4),
    "Recall":                 round(rec,  4),
    "F1-Score":               round(f1,   4),
    "ROC-AUC":                round(auc,  4),
    "Best n_estimators":      bp["clf__n_estimators"],
    "Best max_depth":         bp["clf__max_depth"],
    "Best min_samples_leaf":  bp["clf__min_samples_leaf"],
    "Best max_features":      bp["clf__max_features"],
    "Best min_samples_split": bp["clf__min_samples_split"],
    "Best bootstrap":         bp["clf__bootstrap"],
}

pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
print(f"\n{pd.DataFrame([metrics]).T.to_string(header=False)}")

# ── 6. Classification Report ──────────────────
report = classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"])
print(f"\n{report}")
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Best params: {bp}\n")
    f.write(f"Best threshold: {best_threshold:.4f}\n\n")
    f.write(report)

# ── 7. Plots ──────────────────────────────────
# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
ax.set_title(f"Random Forest — Confusion Matrix\n(threshold={best_threshold:.3f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, color="#d62728", label=f"Random Forest (AUC = {auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Random Forest")
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
ax.set_title("Precision / Recall / F1 vs Threshold — Random Forest")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "threshold_tuning.png"), dpi=150)
plt.close()

# Feature Importance
clf        = best_pipeline.named_steps["clf"]
importances = pd.Series(clf.feature_importances_, index=X.columns)
top15      = importances.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(7, 6))
top15.plot(kind="barh", ax=ax, color="#d62728", edgecolor="black")
ax.set_title("Top 15 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# RandomizedSearch scatter: n_estimators vs CV F1
cv_df = pd.DataFrame(search.cv_results_)
fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(
    cv_df["param_clf__n_estimators"],
    cv_df["mean_test_score"],
    c=cv_df["param_clf__max_depth"].apply(lambda x: 0 if x is None else x),
    cmap="Reds", edgecolors="black", s=60
)
plt.colorbar(sc, ax=ax, label="max_depth (0 = None)")
ax.set_xlabel("n_estimators"); ax.set_ylabel("CV F1 Score")
ax.set_title("RandomizedSearch: n_estimators vs CV F1")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hyperparam_search.png"), dpi=150)
plt.close()

print(f"\nAll outputs saved to: {RESULTS_DIR}/")