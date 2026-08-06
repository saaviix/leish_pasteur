"""
projections.py
==============
Projections climatiques futures (2030, 2050, 2070, 2100) pour la presence
de P. sergenti. Applique des deltas de temperature et de precipitation aux
covariables courantes selon des scenarios SSP, puis re-estime psi de
maniere approchee (sans re-executer le modele Bayesien complet).
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

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


SCENARIOS = {
    "SSP126": {"temp_delta": 1.5, "precip_delta": 0.05},
    "SSP585": {"temp_delta": 4.0, "precip_delta": -0.10},
}

PROJECTION_YEARS = [2030, 2050, 2070, 2100]


def _load_communes_climate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Communes climate CSV not found: {path}")
    return pd.read_csv(path)


def _load_posterior(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Posterior CSV not found: {path}")
    return pd.read_csv(path)


def _build_coef(posterior_df: pd.DataFrame, climate_df: pd.DataFrame) -> dict:
    """
    Derive approximate coefficients from the posterior psi values and current climate.
    Since the posterior CSV contains per-province psi (not raw parameter samples),
    we fit an empirical logistic regression: psi ~ temp + precip + aridity.
    """
    from scipy import stats as sp_stats
    
    # Merge psi with climate
    merged = posterior_df.merge(climate_df, on="province", how="inner")
    
    if len(merged) < 5:
        raise ValueError(f"Not enough provinces with both psi and climate data: {len(merged)}")
    
    coef = {"alpha": 0.0}
    
    # Fit logistic regression for temp
    if "temp_mean" in merged.columns and merged["temp_mean"].notna().any():
        x = merged["temp_mean"].values
        y = merged["psi_mean"].values
        # Logistic regression via scipy
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (y < 1)
        if mask.sum() >= 3:
            logit_y = np.log(y[mask] / (1 - y[mask]))
            slope, intercept, _, _, _ = sp_stats.linregress(x[mask], logit_y)
            coef["beta_temp"] = slope
            coef["alpha"] = intercept
    
    # Fit for precip
    if "precip_annual" in merged.columns and merged["precip_annual"].notna().any():
        x = merged["precip_annual"].values
        y = merged["psi_mean"].values
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (y < 1)
        if mask.sum() >= 3:
            logit_y = np.log(y[mask] / (1 - y[mask]))
            slope, _, _, _, _ = sp_stats.linregress(x[mask], logit_y)
            if "beta_precip" not in coef:
                coef["beta_precip"] = 0.0
            coef["beta_precip"] += slope
    
    # Fit for aridity
    if "arid_index" in merged.columns and merged["arid_index"].notna().any():
        x = merged["arid_index"].values
        y = merged["psi_mean"].values
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (y < 1)
        if mask.sum() >= 3:
            logit_y = np.log(y[mask] / (1 - y[mask]))
            slope, _, _, _, _ = sp_stats.linregress(x[mask], logit_y)
            coef["beta_arid"] = slope
    
    logger.info("Derived projection coefficients: %s", {k: f"{v:.3f}" for k, v in coef.items() if k != "alpha"})
    return coef


def _apply_deltas(climate_df: pd.DataFrame, scenario: dict, year: int) -> pd.DataFrame:
    base_year = 2021
    factor = max(0.0, (year - base_year) / (2100 - base_year))

    df = climate_df.copy()

    for col, delta in scenario.items():
        if col == "temp_delta":
            if "temp_mean" in df.columns:
                df["temp_mean_proj"] = df["temp_mean"] + delta * factor
            if "temp_min" in df.columns:
                df["temp_min_proj"] = df["temp_min"] + delta * factor
            if "temp_max" in df.columns:
                df["temp_max_proj"] = df["temp_max"] + delta * factor
        elif col == "precip_delta":
            for pcol in ["precip", "precip_annual"]:
                if pcol in df.columns:
                    df[pcol + "_proj"] = df[pcol] * (1 + delta * factor)
    return df


def _estimate_psi(df: pd.DataFrame, coef: dict) -> pd.Series:
    logit_psi = df.get("alpha", 0)
    if "temp_mean_proj" in df.columns and "beta_temp" in coef:
        logit_psi = logit_psi + coef["beta_temp"] * df["temp_mean_proj"]
    elif "temp_mean" in df.columns and "beta_temp" in coef:
        logit_psi = logit_psi + coef["beta_temp"] * df["temp_mean"]
    if "precip_proj" in df.columns and "beta_precip" in coef:
        logit_psi = logit_psi + coef["beta_precip"] * df["precip_proj"]
    elif "precip_annual" in df.columns and "beta_precip" in coef:
        logit_psi = logit_psi + coef["beta_precip"] * df["precip_annual"]
    if "arid_index" in df.columns and "beta_arid" in coef:
        logit_psi = logit_psi + coef["beta_arid"] * df["arid_index"]
    psi = 1.0 / (1.0 + np.exp(-logit_psi))
    return psi.clip(0, 1)


def _run_scenario(climate_df: pd.DataFrame, coef: dict,
                  scenario_name: str, scenario_params: dict) -> pd.DataFrame:
    rows = []
    for year in PROJECTION_YEARS:
        df = _apply_deltas(climate_df, scenario_params, year)
        df["psi_projected"] = _estimate_psi(df, coef)
        df["scenario"] = scenario_name
        df["year"] = year
        rows.append(df[["scenario", "year", "psi_projected"] + [c for c in df.columns if c.endswith("_proj")]].copy())
    return pd.concat(rows, ignore_index=True)


def plot_projection_time_series(projections: pd.DataFrame, output_path: Path) -> None:
    summary = projections.groupby(["scenario", "year"])["psi_projected"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=summary, x="year", y="psi_projected", hue="scenario",
                 marker="o", ax=ax, palette=["tab:green", "tab:red"])
    ax.set_title("Projected Mean ψ by Scenario", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Occupancy (ψ)", fontsize=12)
    ax.legend(title="Scenario")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved projection time series to %s", output_path)


def plot_scenario_comparison(projections: pd.DataFrame, output_path: Path) -> None:
    summary = projections.groupby(["scenario", "year"])["psi_projected"].mean().reset_index()
    pivot = summary.pivot(index="year", columns="scenario", values="psi_projected")
    if "SSP126" in pivot.columns and "SSP585" in pivot.columns:
        pivot["delta"] = pivot["SSP585"] - pivot["SSP126"]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=pivot.index.astype(str), y=pivot["delta"], palette="coolwarm", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Scenario Comparison: SSP585 − SSP126 ψ Difference", fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Δ ψ (pessimistic − optimistic)", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        logger.info("Saved scenario comparison plot to %s", output_path)


def main() -> None:
    config.ensure_dirs()
    logger.info("Running climate projections...")

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
        coef = _build_coef(posterior_df, climate_df)
    except ValueError as exc:
        logger.error("Failed to derive projection coefficients: %s", exc)
        return

    all_projections = []
    for scenario_name, scenario_params in SCENARIOS.items():
        try:
            proj = _run_scenario(climate_df, coef, scenario_name, scenario_params)
            all_projections.append(proj)
        except Exception as exc:
            logger.warning("Scenario %s failed: %s", scenario_name, exc)

    if not all_projections:
        logger.error("No scenarios completed successfully.")
        return

    projections = pd.concat(all_projections, ignore_index=True)
    out_csv = config.PROCESSED / "future_projections.csv"
    projections.to_csv(out_csv, index=False)
    logger.info("Saved future projections CSV to %s", out_csv)

    plot_projection_time_series(projections, config.FIGURES / "projections_time_series.png")
    plot_scenario_comparison(projections, config.FIGURES / "projections_scenario_comparison.png")

    logger.info("Projections analysis completed.")


if __name__ == "__main__":
    main()
