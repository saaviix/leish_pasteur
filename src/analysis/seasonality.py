"""
seasonality.py
==============
Analyse des motifs saisonniers de P. sergenti et des cas LCT.
Lit la base de donnees climatiques journalieres et le CSV LCT nettoye.
Produit des figures haute resolution (300 dpi) dans outputs/figures/.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3
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


def _load_climate_db(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Climate database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM climate", conn)
    conn.close()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df


def _load_lct(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"LCT CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df


def plot_seasonal_lct_incidence(df_lct: pd.DataFrame, output_path: Path) -> None:
    monthly = (
        df_lct.groupby("month", dropna=True)["cases"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
        .reset_index()
    )
    monthly["month_name"] = monthly["month"].map(
        lambda m: pd.to_datetime(f"2000-{m:02d}-01").strftime("%b")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=monthly, x="month_name", y="cases", palette="Blues_d", ax=ax)
    ax.set_title("Monthly LCT Case Incidence (Aggregated)", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Cases", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved seasonal LCT incidence chart to %s", output_path)


def plot_seasonal_climate(df_climate: pd.DataFrame, output_path: Path) -> None:
    if "temp_mean" not in df_climate.columns or "precip" not in df_climate.columns:
        logger.warning("Expected columns 'temp_mean'/'precip' not found in climate data.")
        return

    monthly_climate = (
        df_climate.groupby("month", dropna=True)
        .agg(temp_mean=("temp_mean", "mean"), precip=("precip", "mean"))
        .reindex(range(1, 13), fill_value=np.nan)
        .reset_index()
    )
    monthly_climate["month_name"] = monthly_climate["month"].map(
        lambda m: pd.to_datetime(f"2000-{m:02d}-01").strftime("%b")
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    sns.lineplot(data=monthly_climate, x="month_name", y="temp_mean",
                 color="tab:red", marker="o", ax=ax1, label="Mean Temperature")
    sns.lineplot(data=monthly_climate, x="month_name", y="precip",
                 color="tab:blue", marker="s", ax=ax2, label="Precipitation")

    ax1.set_title("Monthly Temperature and Precipitation Patterns", fontsize=14)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=12, color="tab:red")
    ax2.set_ylabel("Precipitation (mm)", fontsize=12, color="tab:blue")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved seasonal climate chart to %s", output_path)


def plot_climate_case_correlation(df_climate: pd.DataFrame, df_lct: pd.DataFrame,
                                  output_path: Path) -> None:
    if "temp_mean" not in df_climate.columns or "precip" not in df_climate.columns:
        logger.warning("Expected columns not found; skipping correlation plot.")
        return

    climate_monthly = (
        df_climate.groupby(["year", "month"], dropna=True)
        .agg(temp_mean=("temp_mean", "mean"), precip=("precip", "mean"))
        .reset_index()
    )
    lct_monthly = (
        df_lct.groupby(["year", "month"], dropna=True)["cases"]
        .sum()
        .reset_index()
    )

    merged = pd.merge(climate_monthly, lct_monthly, on=["year", "month"], how="inner")
    if merged.empty:
        logger.warning("No overlapping data between climate and LCT; skipping correlation plot.")
        return

    corr = merged[["temp_mean", "precip", "cases"]].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                ax=ax, square=True)
    ax.set_title("Climate–Case Correlation by Month", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved climate–case correlation heatmap to %s", output_path)


def main() -> None:
    config.ensure_dirs()
    logger.info("Running seasonality analysis...")

    try:
        df_climate = _load_climate_db(config.CLIMATE_DB)
    except FileNotFoundError:
        logger.error("Climate DB not available: %s", config.CLIMATE_DB)
        return
    try:
        df_lct = _load_lct(config.LCT_CSV)
    except FileNotFoundError:
        logger.error("LCT CSV not available: %s", config.LCT_CSV)
        return

    plot_seasonal_lct_incidence(df_lct, config.FIGURES / "seasonality_lct_incidence.png")
    plot_seasonal_climate(df_climate, config.FIGURES / "seasonality_climate.png")
    plot_climate_case_correlation(df_climate, df_lct,
                                  config.FIGURES / "seasonality_climate_case_corr.png")

    logger.info("Seasonality analysis completed.")


if __name__ == "__main__":
    main()
