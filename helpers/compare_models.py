"""
Bank Marketing Classification — Combined Comparison
=====================================================
Reads results_summary.csv from each model's results folder
and produces a unified comparison table + plots for the report.

Expected folder structure:
  results/
    DecisionTree/results_summary.csv
    NaiveBayes/results_summary.csv
    kNN/results_summary.csv
    RandomForest/results_summary.csv
    XGBoost/results_summary.csv
    LightGBM/results_summary.csv
    StackingEnsemble/results_summary.csv

Output saved to: overall_results/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config ───────────────────────────────────
MODELS = {
    "Decision Tree":      "train/results/DecisionTree/results_summary.csv",
    "Naive Bayes":        "train/results/NaiveBayes/results_summary.csv",
    "k-NN":               "train/results/kNN/results_summary.csv",
    "Random Forest":      "train/results/RandomForest/results_summary.csv",
    "XGBoost":            "train/results/XGBoost/results_summary.csv",
    "LightGBM":           "train/results/LightGBM/results_summary.csv",
    "Stacking Ensemble":  "train/results/StackingEnsemble/results_summary.csv",
}

COLORS = {
    "Decision Tree":      "#1f77b4",
    "Naive Bayes":        "#ff7f0e",
    "k-NN":               "#2ca02c",
    "Random Forest":      "#d62728",
    "XGBoost":            "#9467bd",
    "LightGBM":           "#17becf",
    "Stacking Ensemble":  "#e377c2",
}

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

OUT_DIR = os.path.join("overall_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load all results ───────────────────────
print("=" * 55)
print("Combined Comparison — Bank Marketing")
print("=" * 55)

rows = []
for model_name, csv_path in MODELS.items():
    if not os.path.exists(csv_path):
        print(f"[WARNING] Missing: {csv_path}  — skipping.")
        continue
    row = pd.read_csv(csv_path).iloc[0].to_dict()
    row["Model"] = model_name
    rows.append(row)

df = pd.DataFrame(rows).set_index("Model")
print("\nLoaded results for:", list(df.index))

# ── 2. Unified metrics table ──────────────────
metric_df = df[METRICS + ["Best Threshold"]].copy()
metric_df = metric_df.astype(float).round(4)

# Highlight best per metric
print("\n── Metrics Table ──")
print(metric_df.to_string())

# Save as CSV (clean, ready for LaTeX/Word table)
metric_df.to_csv(os.path.join(OUT_DIR, "combined_metrics.csv"))
print(f"\nSaved: {OUT_DIR}/combined_metrics.csv")

# ── 3. Ranked summary table (sorted by F1) ────
ranked = metric_df.sort_values("F1-Score", ascending=False).copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))
ranked.to_csv(os.path.join(OUT_DIR, "ranked_summary.csv"))
print(f"Saved: {OUT_DIR}/ranked_summary.csv")
print("\n── Ranked by F1-Score ──")
print(ranked.to_string())

# ── 4. Grouped bar chart — all metrics ────────
fig, ax = plt.subplots(figsize=(15, 5))
x       = np.arange(len(METRICS))
n       = len(metric_df)
width   = 0.11
offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

for i, (model, row) in enumerate(metric_df[METRICS].iterrows()):
    color = COLORS.get(model, "#aaaaaa")
    bars = ax.bar(
        x + offsets[i] * width,
        row.values,
        width,
        label=model,
        color=color,
        edgecolor="black",
        linewidth=0.5
    )
    # annotate bars with value
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=5.5
        )

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Classifier Comparison — All Metrics (Tuned Thresholds)", fontsize=13)
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "grouped_bar_all_metrics.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/grouped_bar_all_metrics.png")

# ── 5. Heatmap — metrics table ────────────────
fig, ax = plt.subplots(figsize=(9, 5))
heatmap_data = metric_df[METRICS].astype(float)
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    ax=ax, vmin=0.4, vmax=1.0,
    cbar_kws={"label": "Score"}
)
ax.set_title("Performance Heatmap — All Classifiers", fontsize=13)
ax.set_xlabel("Metric"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_metrics.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/heatmap_metrics.png")

# ── 6. F1-Score bar chart (main takeaway) ─────
f1_sorted  = metric_df["F1-Score"].sort_values(ascending=True)
bar_colors = [COLORS.get(m, "#aaaaaa") for m in f1_sorted.index]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(f1_sorted.index, f1_sorted.values,
               color=bar_colors, edgecolor="black", linewidth=0.6)
for bar in bars:
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}", va="center", fontsize=10)
ax.set_xlim(0, 0.85)
ax.set_xlabel("F1-Score (Yes class)", fontsize=11)
ax.set_title("F1-Score Comparison — All Classifiers", fontsize=13)
ax.axvline(f1_sorted.max(), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_comparison.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/f1_comparison.png")

# ── 7. Radar / Spider chart ───────────────────
radar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model, row in metric_df[radar_metrics].iterrows():
    color = COLORS.get(model, "#aaaaaa")
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, lw=2, color=color, label=model)
    ax.fill(angles, values, alpha=0.05, color=color)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Radar Chart — Classifier Comparison", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "radar_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/radar_chart.png")

# ── 8. Precision vs Recall scatter ───────────
fig, ax = plt.subplots(figsize=(8, 6))
for model, row in metric_df.iterrows():
    color = COLORS.get(model, "#aaaaaa")
    ax.scatter(row["Recall"], row["Precision"],
               color=color, s=120, zorder=3, edgecolors="black", linewidth=0.7)
    ax.annotate(model, (row["Recall"], row["Precision"]),
                textcoords="offset points", xytext=(8, 4), fontsize=8)

ax.set_xlabel("Recall (Yes class)", fontsize=11)
ax.set_ylabel("Precision (Yes class)", fontsize=11)
ax.set_title("Precision vs Recall — All Classifiers", fontsize=13)
# Auto-range based on actual data
r_min = max(0, metric_df["Recall"].min() - 0.05)
r_max = min(1, metric_df["Recall"].max() + 0.05)
p_min = max(0, metric_df["Precision"].min() - 0.05)
p_max = min(1, metric_df["Precision"].max() + 0.1)
ax.set_xlim(r_min, r_max); ax.set_ylim(p_min, p_max)
ax.grid(alpha=0.3)

# iso-F1 curves
for f1_val in [0.50, 0.55, 0.60, 0.65, 0.70]:
    r = np.linspace(0.01, 1.0, 300)
    p = (f1_val * r) / (2 * r - f1_val + 1e-9)
    mask = (p > 0) & (p <= 1)
    ax.plot(r[mask], p[mask], "k--", linewidth=0.6, alpha=0.4)
    ax.text(r[mask][-1] + 0.01, p[mask][-1], f"F1={f1_val:.2f}", fontsize=7, alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "precision_recall_scatter.png"), dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/precision_recall_scatter.png")

# ── 9. Print final summary ────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(ranked[["Rank"] + METRICS + ["Best Threshold"]].to_string())
print(f"\nBest model by F1  : {ranked['F1-Score'].idxmax()}")
print(f"Best model by AUC : {ranked['ROC-AUC'].idxmax()}")
print(f"\nAll outputs saved to: {OUT_DIR}/")