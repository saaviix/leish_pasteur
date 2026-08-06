"""
figures.py
==========
Generation centralisee de toutes les figures de publication.
Appelle chaque module d'analyse et produit egalement les diagnostics
de modele, le classement des provinces et le graphique de qualite
d'inference des provinces manquantes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "data_prep"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from data_prep import config

from analysis import seasonality, climate_response, projections, spatial

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def plot_model_diagnostics(output_path: Path) -> None:
    try:
        posterior_df = pd.read_csv(config.POSTERIOR_CSV)
    except FileNotFoundError:
        logger.warning("Posterior CSV not found; skipping model diagnostics.")
        return

    param_col = "parameter" if "parameter" in posterior_df.columns else posterior_df.columns[0]
    mean_col = "mean" if "mean" in posterior_df.columns else posterior_df.columns[-1]
    sd_col = "sd" if "sd" in posterior_df.columns else None

    params = posterior_df[param_col].astype(str).tolist()
    means = posterior_df[mean_col].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(range(len(params)), means, color="steelblue")
    axes[0].set_xticks(range(len(params)))
    axes[0].set_xticklabels(params, rotation=45, ha="right")
    axes[0].set_title("Posterior Mean by Parameter", fontsize=14)
    axes[0].set_ylabel("Value", fontsize=12)

    if sd_col and sd_col in posterior_df.columns:
        sds = posterior_df[sd_col].values
        axes[1].errorbar(range(len(params)), means, yerr=sds, fmt="o",
                         color="tab:red", ecolor="gray", capsize=4)
    else:
        axes[1].scatter(range(len(params)), means, color="tab:red")
    axes[1].set_xticks(range(len(params)))
    axes[1].set_xticklabels(params, rotation=45, ha="right")
    axes[1].set_title("Posterior Uncertainty (±SD)", fontsize=14)
    axes[1].set_ylabel("Value", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved model diagnostics to %s", output_path)


def plot_province_ranking(output_path: Path) -> None:
    try:
        posterior_df = pd.read_csv(config.POSTERIOR_CSV)
    except FileNotFoundError:
        logger.warning("Posterior CSV not found; skipping province ranking.")
        return

    psi_col = None
    for c in ["psi_mean", "psi", "mean"]:
        if c in posterior_df.columns:
            psi_col = c
            break
    if psi_col is None:
        psi_col = posterior_df.columns[-1]

    sorted_df = posterior_df.sort_values(psi_col, ascending=True).tail(30)
    labels = sorted_df.iloc[:, 0].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(labels, sorted_df[psi_col].values, color="teal")
    ax.set_title("Top 30 Provinces by Estimated ψ", fontsize=14)
    ax.set_xlabel("Mean Occupancy (ψ)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved province ranking to %s", output_path)


def plot_gap_inference_quality(output_path: Path) -> None:
    try:
        province_df = pd.read_csv(config.PROVINCE_TABLE)
        posterior_df = pd.read_csv(config.POSTERIOR_CSV)
    except FileNotFoundError:
        logger.warning("Required files not found; skipping gap inference quality plot.")
        return

    psi_col = None
    for c in ["psi_mean", "psi", "mean"]:
        if c in posterior_df.columns:
            psi_col = c
            break
    if psi_col is None:
        psi_col = posterior_df.columns[-1]

    sd_col = None
    for c in ["psi_sd", "sd"]:
        if c in posterior_df.columns:
            sd_col = c
            break

    merged = province_df.reset_index(drop=True).join(posterior_df.reset_index(drop=True))
    if sd_col and sd_col in merged.columns:
        x = merged[psi_col].fillna(0)
        y = merged[sd_col].fillna(0)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Mean ψ", fontsize=12)
        ax.set_ylabel("SD of ψ", fontsize=12)
        ax.set_title("Inference Quality: ψ Uncertainty vs Mean", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        logger.info("Saved gap inference quality plot to %s", output_path)
    else:
        logger.warning("No SD column found; skipping gap inference quality plot.")


def generate_all_figures() -> None:
    config.ensure_dirs()
    logger.info("Generating all publication-quality figures...")

    try:
        seasonality.main()
    except Exception as exc:
        logger.warning("Seasonality figures failed: %s", exc)

    try:
        climate_response.main()
    except Exception as exc:
        logger.warning("Climate response figures failed: %s", exc)

    try:
        projections.main()
    except Exception as exc:
        logger.warning("Projections figures failed: %s", exc)

    try:
        spatial.main()
    except Exception as exc:
        logger.warning("Spatial figures failed: %s", exc)

    plot_model_diagnostics(config.FIGURES / "diagnostics_model.png")
    plot_province_ranking(config.FIGURES / "ranking_provinces.png")
    plot_gap_inference_quality(config.FIGURES / "gap_inference_quality.png")

    logger.info("All figures generation completed.")


if __name__ == "__main__":
    generate_all_figures()
