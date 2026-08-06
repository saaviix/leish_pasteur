"""
climate_response.py
===================
Analyse des correlations entre le climat et la presence de P. sergenti.
Lit les donnees climatiques par commune et les posteriors bayesiens.
Produit des courbes de reponse, heatmaps et importance des variables.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "data_prep"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from data_prep import config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def _load_communes_climate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Communes climate CSV not found: {path}")
    return pd.read_csv(path)


def _load_posterior(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Posterior CSV not found: {path}")
    return pd.read_csv(path)


def _prepare_merged(climate_df: pd.DataFrame, posterior_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = [c for c in ["province", "province_id", "commune"] if c in climate_df.columns and c in posterior_df.columns]
    if not join_cols:
        common = list(set(climate_df.columns) & set(posterior_df.columns))
        if not common:
            raise ValueError("No common columns between climate and posterior data.")
        join_cols = common[:1]
    merged = pd.merge(climate_df, posterior_df, on=join_cols, how="inner")
    return merged


def plot_psi_vs_temp(merged: pd.DataFrame, output_path: Path) -> None:
    if "psi_mean" not in merged.columns or "temp_mean" not in merged.columns:
        logger.warning("Required columns missing for psi vs temp plot.")
        return
    x = merged["temp_mean"].dropna()
    y = merged.loc[x.index, "psi_mean"].dropna()
    if len(x) < 5:
        logger.warning("Not enough data points for psi vs temp regression.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ci = 1.96 * std_err

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.plot(x_line, y_line, color="tab:red", label=f"Linear fit (R²={r_value**2:.2f})")
    ax.fill_between(x_line, y_line - ci, y_line + ci, color="tab:red", alpha=0.2)
    ax.set_title("P. sergenti Occupancy vs Mean Temperature", fontsize=14)
    ax.set_xlabel("Mean Temperature (°C)", fontsize=12)
    ax.set_ylabel("Mean Occupancy (ψ)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved psi vs temp plot to %s", output_path)


def plot_psi_vs_precip(merged: pd.DataFrame, output_path: Path) -> None:
    col = "precip_annual" if "precip_annual" in merged.columns else "precipitation"
    if "psi_mean" not in merged.columns or col not in merged.columns:
        logger.warning("Required columns missing for psi vs precip plot.")
        return
    x = merged[col].dropna()
    y = merged.loc[x.index, "psi_mean"].dropna()
    if len(x) < 5:
        logger.warning("Not enough data points for psi vs precip regression.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ci = 1.96 * std_err

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.plot(x_line, y_line, color="tab:blue", label=f"Linear fit (R²={r_value**2:.2f})")
    ax.fill_between(x_line, y_line - ci, y_line + ci, color="tab:blue", alpha=0.2)
    ax.set_title("P. sergenti Occupancy vs Annual Precipitation", fontsize=14)
    ax.set_xlabel("Annual Precipitation (mm)", fontsize=12)
    ax.set_ylabel("Mean Occupancy (ψ)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved psi vs precip plot to %s", output_path)


def plot_psi_vs_arid(merged: pd.DataFrame, output_path: Path) -> None:
    if "psi_mean" not in merged.columns or "arid_index" not in merged.columns:
        logger.warning("Required columns missing for psi vs arid_index plot.")
        return
    x = merged["arid_index"].dropna()
    y = merged.loc[x.index, "psi_mean"].dropna()
    if len(x) < 5:
        logger.warning("Not enough data points for psi vs arid_index regression.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ci = 1.96 * std_err

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.plot(x_line, y_line, color="tab:green", label=f"Linear fit (R²={r_value**2:.2f})")
    ax.fill_between(x_line, y_line - ci, y_line + ci, color="tab:green", alpha=0.2)
    ax.set_title("P. sergenti Occupancy vs Aridity Index", fontsize=14)
    ax.set_xlabel("Aridity Index", fontsize=12)
    ax.set_ylabel("Mean Occupancy (ψ)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved psi vs arid_index plot to %s", output_path)


def plot_correlation_heatmap(merged: pd.DataFrame, output_path: Path) -> None:
    covars = ["psi_mean", "psi_sd", "temp_mean", "precip_annual", "arid_index"]
    present = [c for c in covars if c in merged.columns]
    if len(present) < 3:
        logger.warning("Not enough covariates for correlation heatmap.")
        return

    corr = merged[present].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Climate Covariate Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", output_path)


def plot_variable_importance(merged: pd.DataFrame, output_path: Path) -> None:
    covars = ["temp_mean", "precip_annual", "arid_index"]
    present = [c for c in covars if c in merged.columns]
    if "psi_mean" not in merged.columns or len(present) < 2:
        logger.warning("Not enough data for variable importance plot.")
        return

    importances = []
    for cov in present:
        x = merged[cov].dropna()
        y = merged.loc[x.index, "psi_mean"].dropna()
        if len(x) < 5:
            continue
        _, _, r_value, _, _ = stats.linregress(x, y)
        importances.append({"variable": cov, "r_squared": r_value**2})

    if not importances:
        logger.warning("No valid regressions for variable importance.")
        return

    imp_df = pd.DataFrame(importances).sort_values("r_squared", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp_df["variable"], imp_df["r_squared"], color="steelblue")
    ax.set_xlabel("R² (explained variance)", fontsize=12)
    ax.set_title("Variable Importance for ψ Prediction", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved variable importance plot to %s", output_path)


def main() -> None:
    config.ensure_dirs()
    logger.info("Running climate response analysis...")

    try:
        climate_df = _load_communes_climate(config.COMMUNES_CLIMATE)
    except FileNotFoundError:
        logger.error("Communes climate CSV not available: %s", config.COMMUNES_CLIMATE)
        return
    try:
        posterior_df = _load_posterior(config.POSTERIOR_CSV)
    except FileNotFoundError:
        logger.error("Posterior CSV not available: %s", config.POSTERIOR_CSV)
        return

    try:
        merged = _prepare_merged(climate_df, posterior_df)
    except ValueError as exc:
        logger.error("Failed to merge datasets: %s", exc)
        return

    plot_psi_vs_temp(merged, config.FIGURES / "climate_response_psi_vs_temp.png")
    plot_psi_vs_precip(merged, config.FIGURES / "climate_response_psi_vs_precip.png")
    plot_psi_vs_arid(merged, config.FIGURES / "climate_response_psi_vs_arid.png")
    plot_correlation_heatmap(merged, config.FIGURES / "climate_response_correlation_heatmap.png")
    plot_variable_importance(merged, config.FIGURES / "climate_response_variable_importance.png")

    logger.info("Climate response analysis completed.")


if __name__ == "__main__":
    main()
