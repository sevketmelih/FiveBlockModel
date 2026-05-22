#!/usr/bin/env python3
"""
Task 4 — Publication-grade visualization & analytics utilities.

Student 210911028 — The Visualization & Analytics Guru
All figures saved at 300 DPI (IEEE/Springer print standard).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Publication constants
# ---------------------------------------------------------------------------
PUBLICATION_DPI = 300
SAVE_KWARGS = {
    "dpi": PUBLICATION_DPI,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
}

PALETTE = ["#003f5c", "#bc5090", "#ffa600", "#ff6361", "#58508d"]
MODEL_SHORT = {
    "Model A": "A (Full)",
    "Model B": "B (No DAE)",
    "Model C": "C (No Attn)",
    "Model D": "D (Base)",
}
RUSH_HOUR_BANDS = ((7, 9), (17, 20))  # Beijing morning / evening peaks
WINDOW_SIZE = 24
POOL_FACTOR = 2  # MaxPool1D(pool_size=2) after CNN


def apply_publication_style() -> None:
    """Matplotlib + seaborn theme for academic figures."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.15,
        rc={
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.35,
        },
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)


def _short_model_name(scenario: str) -> str:
    for prefix, label in MODEL_SHORT.items():
        if scenario.startswith(prefix):
            return label
    m = re.match(r"Model ([A-D])", scenario)
    return m.group(0) if m else scenario[:20]


def _format_metric(val: float, kind: str) -> str:
    if kind == "r2":
        return f"{val:.4f}"
    if kind == "mse":
        return f"{val:.2f}"
    return f"{val:.2f}"


# ---------------------------------------------------------------------------
# Markdown tables (Clean vs Noisy ablation)
# ---------------------------------------------------------------------------
def build_ablation_markdown(
    metrics_clean: Union[pd.DataFrame, List[dict]],
    metrics_noisy: Union[pd.DataFrame, List[dict]],
) -> str:
    """Build publication-ready Markdown for clean and noisy test scenarios."""
    df_c = pd.DataFrame(metrics_clean)
    df_n = pd.DataFrame(metrics_noisy)

    def pivot_metrics(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "Model": _short_model_name(r["Scenario"]),
                    "MSE": r["MSE (ug/m^3)^2"],
                    "MAE": r["MAE (ug/m^3)"],
                    "R²": r["R2 Score"],
                }
            )
        return pd.DataFrame(rows)

    pc, pn = pivot_metrics(df_c), pivot_metrics(df_n)
    merged = pc.merge(pn, on="Model", suffixes=("_clean", "_noisy"))
    merged["ΔR²"] = merged["R²_noisy"] - merged["R²_clean"]
    merged["ΔMAE"] = merged["MAE_noisy"] - merged["MAE_clean"]

    lines = [
        "# Ablation Study — Quantitative Results (Publication Tables)",
        "",
        "**Dataset:** Zhang et al. (2017) — Aotizhongxin, chronological test split (15%).  ",
        "**Forecast horizon:** $T+1$ hour ahead from 24-hour multivariate windows.",
        "",
        "## Table 1 — Clean Test Set (No Sensor Corruption)",
        "",
        "| Model | MSE (µg/m³)² | MAE (µg/m³) | R² |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for _, r in pc.iterrows():
        lines.append(
            f"| **{r['Model']}** | {_format_metric(r['MSE'], 'mse')} | "
            f"{_format_metric(r['MAE'], 'mae')} | {_format_metric(r['R²'], 'r2')} |"
        )

    lines.extend(
        [
            "",
            "## Table 2 — Noisy Test Set (Gaussian Corruption on Auxiliary Sensors)",
            "",
            f"Test-time noise: $\\sigma = 0.12$ on scaled auxiliary channels (columns 1:).",
            "",
            "| Model | MSE (µg/m³)² | MAE (µg/m³) | R² |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    for _, r in pn.iterrows():
        lines.append(
            f"| **{r['Model']}** | {_format_metric(r['MSE'], 'mse')} | "
            f"{_format_metric(r['MAE'], 'mae')} | {_format_metric(r['R²'], 'r2')} |"
        )

    lines.extend(
        [
            "",
            "## Table 3 — Clean vs. Noisy Comparison (Robustness)",
            "",
            "| Model | R² (Clean) | R² (Noisy) | ΔR² | MAE (Clean) | MAE (Noisy) | ΔMAE |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in merged.iterrows():
        lines.append(
            f"| **{r['Model']}** | {_format_metric(r['R²_clean'], 'r2')} | "
            f"{_format_metric(r['R²_noisy'], 'r2')} | {r['ΔR²']:+.4f} | "
            f"{_format_metric(r['MAE_clean'], 'mae')} | {_format_metric(r['MAE_noisy'], 'mae')} | "
            f"{r['ΔMAE']:+.2f} |"
        )

    # DAE headline comparison
    a_c = merged[merged["Model"].str.contains("A")]
    b_c = merged[merged["Model"].str.contains("B")]
    if not a_c.empty and not b_c.empty:
        da_clean = a_c["R²_clean"].iloc[0] - b_c["R²_clean"].iloc[0]
        da_noisy = a_c["R²_noisy"].iloc[0] - b_c["R²_noisy"].iloc[0]
        lines.extend(
            [
                "",
                "## Table 4 — DAE Block Contribution (Model A vs. Model B)",
                "",
                "| Comparison | Clean ΔR² (A − B) | Noisy ΔR² (A − B) | Interpretation |",
                "| :--- | :---: | :---: | :--- |",
                f"| DAE advantage | {da_clean:+.4f} | {da_noisy:+.4f} | "
                + (
                    "DAE improves robustness under sensor noise."
                    if da_noisy > da_clean
                    else "Comparable on clean data."
                )
                + " |",
            ]
        )

    lines.append("")
    lines.append("*Generated by Task 4 visualization pipeline (Student 210911028).*")
    return "\n".join(lines)


def write_ablation_markdown_tables(
    metrics_clean: Union[pd.DataFrame, List[dict]],
    metrics_noisy: Union[pd.DataFrame, List[dict]],
    path: str = "outputs/ablation_tables.md",
) -> str:
    md = build_ablation_markdown(metrics_clean, metrics_noisy)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------
def plot_validation_loss_curves(
    histories: Dict[str, dict],
    path: str = "outputs/ablation_loss_curves.png",
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, hist) in enumerate(histories.items()):
        key = "val_loss" if "val_loss" in hist else "val_forecast_output_loss"
        ax.plot(
            hist[key],
            label=_short_model_name(name),
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.5,
        )
    ax.set_title("Ablation Study: Validation Loss Curves", pad=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    fig.tight_layout()
    save_figure(fig, path)


def plot_prediction_timeseries(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    path: str = "outputs/prediction_timeseries_72h.png",
    start: int = 200,
    length: int = 72,
) -> None:
    """72-hour forecast comparison (line plot)."""
    apply_publication_style()
    sl = slice(start, start + length)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        y_true[sl],
        label="Ground Truth",
        color="#1a1a1a",
        linewidth=2.5,
        linestyle="--",
        zorder=5,
    )
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax.plot(
            y_pred[sl],
            label=_short_model_name(name),
            color=PALETTE[i % len(PALETTE)],
            linewidth=2,
            alpha=0.9,
        )
    ax.set_title("PM₂.₅ Forecast vs. Ground Truth (72-Hour Test Window)", pad=14)
    ax.set_xlabel("Test Sample Index (consecutive hours)")
    ax.set_ylabel("PM₂.₅ (µg/m³)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, path)


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label: str,
    path: str,
    condition: str = "Clean",
) -> None:
    """Single-model scatter with 1:1 reference and R² annotation."""
    apply_publication_style()
    r2 = r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.25,
        s=18,
        c=PALETTE[0],
        edgecolors="none",
        rasterized=True,
    )
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lo, hi = lims[0] - margin, lims[1] + margin
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed PM₂.₅ (µg/m³)")
    ax.set_ylabel("Predicted PM₂.₅ (µg/m³)")
    ax.set_title(f"Actual vs. Predicted — {model_label} ({condition})", pad=14)
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, path)


def plot_scatter_grid_all_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    path: str = "outputs/prediction_scatter_all_models.png",
    condition: str = "Clean",
) -> None:
    """2×2 scatter panel for Models A–D."""
    apply_publication_style()
    n = len(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    lims = [
        min(y_true.min(), min(p.min() for p in predictions.values())),
        max(y_true.max(), max(p.max() for p in predictions.values())),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lo, hi = lims[0] - margin, lims[1] + margin

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        r2 = r2_score(y_true, y_pred)
        ax.scatter(
            y_true,
            y_pred,
            alpha=0.2,
            s=12,
            c=PALETTE[i % len(PALETTE)],
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{_short_model_name(name)}  ($R^2$={r2:.4f})", fontsize=13)
        if i >= 2:
            ax.set_xlabel("Observed PM₂.₅ (µg/m³)")
        if i % 2 == 0:
            ax.set_ylabel("Predicted PM₂.₅ (µg/m³)")

    fig.suptitle(f"Actual vs. Predicted — Ablation Models ({condition} Test)", fontsize=16, y=1.02)
    fig.tight_layout()
    save_figure(fig, path)


def _pooled_to_clock_hours(
    target_hours: np.ndarray,
    n_pooled_steps: int,
) -> np.ndarray:
    """Map each pooled attention index to clock hour (0–23) in the input window."""
    hours = np.zeros(n_pooled_steps, dtype=int)
    for p in range(n_pooled_steps):
        orig_start = p * POOL_FACTOR
        lag_hours = WINDOW_SIZE - orig_start - 1
        hours[p] = lag_hours  # offset from target; aggregated per-sample below
    return hours


def aggregate_attention_by_clock_hour(
    attention_weights: np.ndarray,
    target_hours: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean attention vs. clock hour (0–23) across test samples.

    attention_weights: (n_samples, T_pooled)
    target_hours: hour-of-day (0–23) for each forecast target
    """
    n_pooled = attention_weights.shape[1]
    hour_sum = np.zeros(24, dtype=np.float64)
    hour_count = np.zeros(24, dtype=np.float64)

    for idx in range(attention_weights.shape[0]):
        t_hour = int(target_hours[idx]) % 24
        for p in range(n_pooled):
            orig_start = p * POOL_FACTOR
            lag = WINDOW_SIZE - orig_start - 1
            clock = (t_hour - lag) % 24
            hour_sum[clock] += attention_weights[idx, p]
            hour_count[clock] += 1

    with np.errstate(invalid="ignore"):
        mean_attn = np.divide(
            hour_sum, hour_count, out=np.zeros_like(hour_sum), where=hour_count > 0
        )
    return mean_attn, hour_count


def plot_attention_sample_heatmap(
    attention_weights: np.ndarray,
    path: str = "outputs/attention_weights_map.png",
    n_samples: int = 50,
) -> None:
    """Sample × pooled-timestep heatmap (seaborn)."""
    apply_publication_style()
    data = attention_weights[:n_samples, :]
    n_steps = data.shape[1]
    lag_labels = [
        f"{WINDOW_SIZE - p * POOL_FACTOR - 1}h"
        for p in range(n_steps)
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = LinearSegmentedColormap.from_list(
        "pub_magma", ["#f0f4f8", "#bc5090", "#003f5c", "#1a0000"]
    )
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": "Attention weight α"},
        xticklabels=lag_labels,
        yticklabels=5,
        linewidths=0,
    )
    ax.set_title(
        "Self-Attention Weights — Model A (First 50 Test Samples)",
        pad=14,
    )
    ax.set_xlabel("Lag before forecast (within 24h window)")
    ax.set_ylabel("Test sample index")
    fig.tight_layout()
    save_figure(fig, path)


def plot_attention_hour_of_day(
    attention_weights: np.ndarray,
    target_hours: np.ndarray,
    path: str = "outputs/attention_hour_of_day.png",
) -> np.ndarray:
    """
    Mean attention profile by clock hour with rush-hour bands highlighted.
    Returns mean attention array (length 24).
    """
    apply_publication_style()
    mean_attn, counts = aggregate_attention_by_clock_hour(
        attention_weights, target_hours
    )
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(hours, mean_attn, color=PALETTE[0], alpha=0.85, edgecolor="white", width=0.85)

    for h_start, h_end in RUSH_HOUR_BANDS:
        ax.axvspan(
            h_start - 0.5,
            h_end + 0.5,
            alpha=0.18,
            color=PALETTE[2],
            label="Rush-hour band" if h_start == 7 else None,
        )

    peak_h = int(np.argmax(mean_attn))
    ax.axvline(peak_h, color=PALETTE[3], linestyle=":", linewidth=2, label=f"Peak hour ({peak_h}:00)")

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=0)
    ax.set_xlabel("Clock hour (local time)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(
        "Temporal Focus of Self-Attention by Hour of Day (Model A, Test Set)",
        pad=14,
    )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, path)
    return mean_attn


def plot_attention_rush_hour_comparison(
    mean_attn: np.ndarray,
    path: str = "outputs/attention_rush_hour_comparison.png",
) -> None:
    """Bar comparison: rush hours vs. off-peak mean attention."""
    apply_publication_style()
    rush_mask = np.zeros(24, dtype=bool)
    for h0, h1 in RUSH_HOUR_BANDS:
        for h in range(h0, h1 + 1):
            rush_mask[h] = True

    categories = ["Rush hours\n(07–09, 17–20)", "Off-peak\n(other hours)"]
    values = [mean_attn[rush_mask].mean(), mean_attn[~rush_mask].mean()]
    counts = [rush_mask.sum(), (~rush_mask).sum()]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(
        categories,
        values,
        color=[PALETTE[2], PALETTE[0]],
        edgecolor="white",
        width=0.55,
    )
    for bar, val, n in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            f"{val:.4f}\n(n={n}h)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Rush Hour vs. Off-Peak Attention (Model A)", pad=14)
    ratio = values[0] / max(values[1], 1e-9)
    ax.text(
        0.5,
        0.92,
        f"Rush / off-peak ratio = {ratio:.2f}×",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="#fff8e6", alpha=0.9),
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_clean_vs_noisy_metrics_bar(
    metrics_clean: Union[pd.DataFrame, List[dict]],
    metrics_noisy: Union[pd.DataFrame, List[dict]],
    path: str = "outputs/ablation_clean_vs_noisy_r2.png",
) -> None:
    """Grouped bar chart of R² under clean vs. noisy conditions."""
    apply_publication_style()
    df_c = pd.DataFrame(metrics_clean)
    df_n = pd.DataFrame(metrics_noisy)
    labels = [_short_model_name(s) for s in df_c["Scenario"]]
    r2_clean = df_c["R2 Score"].values
    r2_noisy = df_n["R2 Score"].values

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, r2_clean, width, label="Clean test", color=PALETTE[0], edgecolor="white")
    ax.bar(x + width / 2, r2_noisy, width, label="Noisy test", color=PALETTE[3], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation: Forecast $R^2$ Under Clean vs. Sensor-Noise Conditions", pad=14)
    ax.legend(loc="lower left", frameon=True)
    ax.axhline(0.85, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    fig.tight_layout()
    save_figure(fig, path)


def plot_dae_forecast_comparison(
    y_true: np.ndarray,
    y_pred_a_clean: np.ndarray,
    y_pred_a_noisy: np.ndarray,
    y_pred_b_clean: np.ndarray,
    y_pred_b_noisy: np.ndarray,
    path: str = "outputs/dae_clean_vs_noisy_forecast.png",
    start: int = 200,
    length: int = 72,
) -> None:
    apply_publication_style()
    sl = slice(start, start + length)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true[sl], "k--", linewidth=2.5, label="Ground truth")
    ax.plot(y_pred_a_clean[sl], color=PALETTE[0], linewidth=2, label="Model A (clean)")
    ax.plot(
        y_pred_a_noisy[sl],
        color=PALETTE[0],
        linewidth=2,
        linestyle=":",
        label="Model A (noisy)",
    )
    ax.plot(y_pred_b_clean[sl], color=PALETTE[1], linewidth=2, label="Model B (clean)")
    ax.plot(
        y_pred_b_noisy[sl],
        color=PALETTE[1],
        linewidth=2,
        linestyle=":",
        label="Model B (noisy)",
    )
    ax.set_title("DAE Robustness: PM₂.₅ Forecast Under Sensor Noise (72h Window)", pad=14)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("PM₂.₅ (µg/m³)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_figure(fig, path)


def plot_noise_sweep(
    sweep_df: pd.DataFrame,
    path: str = "outputs/noise_sweep_a_vs_b.png",
) -> None:
    """Model A vs B R² across noise levels."""
    apply_publication_style()
    pivot = sweep_df.pivot(index="Noise_Std", columns="Model", values="R2 Score")
    a_col = [c for c in pivot.columns if "Model A" in str(c)][0]
    b_col = [c for c in pivot.columns if "Model B" in str(c)][0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        pivot.index,
        pivot[a_col],
        marker="o",
        linewidth=2.5,
        color=PALETTE[0],
        label="Model A (DAE)",
    )
    ax.plot(
        pivot.index,
        pivot[b_col],
        marker="s",
        linewidth=2.5,
        color=PALETTE[1],
        label="Model B (No DAE)",
    )
    ax.fill_between(
        pivot.index,
        pivot[a_col],
        pivot[b_col],
        where=(pivot[a_col] >= pivot[b_col]),
        alpha=0.15,
        color=PALETTE[0],
        interpolate=True,
    )
    ax.set_xlabel("Test noise σ (auxiliary sensors)")
    ax.set_ylabel("$R^2$")
    ax.set_title("Noise Robustness Sweep: Model A vs. Model B", pad=14)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, path)


def generate_all_publication_figures(
    histories: Dict[str, dict],
    y_true_clean: np.ndarray,
    test_predictions_clean: Dict[str, np.ndarray],
    metrics_clean: List[dict],
    metrics_noisy: List[dict],
    attention_weights: Optional[np.ndarray] = None,
    target_hours: Optional[np.ndarray] = None,
    sweep_df: Optional[pd.DataFrame] = None,
    y_pred_a_noisy: Optional[np.ndarray] = None,
    y_pred_b_noisy: Optional[np.ndarray] = None,
    y_pred_a_clean: Optional[np.ndarray] = None,
    y_pred_b_clean: Optional[np.ndarray] = None,
) -> None:
    """Orchestrate full Task 4 figure + table export."""
    apply_publication_style()
    os.makedirs("outputs", exist_ok=True)

    write_ablation_markdown_tables(metrics_clean, metrics_noisy)

    if histories:
        plot_validation_loss_curves(histories)
    plot_prediction_timeseries(y_true_clean, test_predictions_clean)
    plot_scatter_grid_all_models(y_true_clean, test_predictions_clean)

    for name, y_pred in test_predictions_clean.items():
        short = _short_model_name(name).replace(" ", "_").replace("(", "").replace(")", "")
        plot_actual_vs_predicted_scatter(
            y_true_clean,
            y_pred,
            _short_model_name(name),
            f"outputs/prediction_scatter_{short}.png",
        )

    # Legacy filename expected by teammates
    key_a = next(k for k in test_predictions_clean if k.startswith("Model A"))
    plot_actual_vs_predicted_scatter(
        y_true_clean,
        test_predictions_clean[key_a],
        "Model A (Full)",
        "outputs/prediction_scatter_plot.png",
    )

    plot_clean_vs_noisy_metrics_bar(metrics_clean, metrics_noisy)

    if attention_weights is not None:
        plot_attention_sample_heatmap(attention_weights)
        if target_hours is not None:
            mean_attn = plot_attention_hour_of_day(attention_weights, target_hours)
            plot_attention_rush_hour_comparison(mean_attn)

    if (
        y_pred_a_clean is not None
        and y_pred_a_noisy is not None
        and y_pred_b_clean is not None
        and y_pred_b_noisy is not None
    ):
        plot_dae_forecast_comparison(
            y_true_clean,
            y_pred_a_clean,
            y_pred_a_noisy,
            y_pred_b_clean,
            y_pred_b_noisy,
        )

    if sweep_df is not None and not sweep_df.empty:
        plot_noise_sweep(sweep_df)

    print("[VISUALIZATION] Publication figures (300 DPI) saved under outputs/")


def regenerate_tables_from_csv(
    clean_path: str = "outputs/ablation_metrics_clean.csv",
    noisy_path: str = "outputs/ablation_metrics_noisy.csv",
) -> str:
    """Regenerate Markdown tables from existing CSV metrics (no retraining)."""
    df_c = pd.read_csv(clean_path)
    df_n = pd.read_csv(noisy_path)
    return write_ablation_markdown_tables(df_c.to_dict("records"), df_n.to_dict("records"))


def regenerate_static_plots_from_csv() -> None:
    """Bar and noise-sweep plots from saved CSVs (no model weights required)."""
    apply_publication_style()
    df_c = pd.read_csv("outputs/ablation_metrics_clean.csv")
    df_n = pd.read_csv("outputs/ablation_metrics_noisy.csv")
    plot_clean_vs_noisy_metrics_bar(df_c, df_n)
    sweep_path = "outputs/noise_sweep_a_vs_b.csv"
    if os.path.exists(sweep_path):
        plot_noise_sweep(pd.read_csv(sweep_path))
    regenerate_tables_from_csv()
    print("[VISUALIZATION] Static plots/tables regenerated from CSV.")
