"""
azilal_lag_timeseries.py
========================
Courbes dual-axis : cas (barres) vs climat (ligne) pour différents lags.
Style inspiré de l'exemple : température retardée qui suit les pics de cas.
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

LAGS = [0, 2, 4, 6]
CLIMATE_VARS = {
    "temp_moy": ("Température (°C)", "tab:red"),
    "precip_mm": ("Précipitation (mm)", "tab:blue"),
    "humidite_pct": ("Humidité (%)", "tab:green"),
}


# ---------------------------------------------------------------------------
# 1. Coordonnées Azilal
# ---------------------------------------------------------------------------
def get_azilal_coords() -> tuple[float, float]:
    pt = pd.read_csv(config.PROVINCE_TABLE)
    row = pt[pt["province"].str.strip().str.lower() == "azilal"]
    if row.empty:
        raise ValueError("Azilal introuvable dans province_table.csv")
    lat = float(row.iloc[0]["lat"])
    lon = float(row.iloc[0]["lon"])
    logger.info("Azilal : lat=%.4f, lon=%.4f", lat, lon)
    return lat, lon


# ---------------------------------------------------------------------------
# 2. Extraction climat ERA5
# ---------------------------------------------------------------------------
def extract_azilal_climate(lat: float, lon: float) -> pd.DataFrame:
    monthly_files = sorted(config.RAW.glob("era5_morocco_*_monthly.nc"))
    if not monthly_files:
        raise FileNotFoundError(f"Aucun fichier era5_morocco_*_monthly.nc dans {config.RAW}")

    rows = []
    for nc_path in monthly_files:
        year = int(nc_path.stem.split("_")[-2])
        ds = xr.open_dataset(nc_path)

        lat_name = [c for c in ds.dims if c.startswith("lat")][0]
        lon_name = [c for c in ds.dims if c.startswith("lon")][0]
        time_name = [c for c in ds.dims if "time" in c or "valid" in c][0]

        def pick(candidates):
            for c in candidates:
                if c in ds.data_vars:
                    return c
            return None

        t2m_name = pick(["t2m", "2m_temperature", "temperature_2m"])
        d2m_name = pick(["d2m", "2m_dewpoint_temperature", "dewpoint_2m"])
        tp_name = pick(["tp", "total_precipitation"])

        if not all([t2m_name, d2m_name, tp_name]):
            raise ValueError(f"Variables ERA5 manquantes dans {nc_path.name}")

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        iy = int(np.abs(lats - lat).argmin())
        ix = int(np.abs(lons - lon).argmin())

        t2m = ds[t2m_name].values[:, iy, ix] - 273.15
        d2m = ds[d2m_name].values[:, iy, ix] - 273.15
        tp = ds[tp_name].values[:, iy, ix]

        times = pd.to_datetime(ds[time_name].values)
        days_in_month = np.array([t.days_in_month for t in times])

        def es(temp_c):
            return 6.112 * math.exp((17.62 * temp_c) / (243.12 + temp_c))

        rh = np.array([
            100 * es(d2m[m]) / es(t2m[m]) if not (math.isnan(d2m[m]) or math.isnan(t2m[m])) else np.nan
            for m in range(len(t2m))
        ])
        rh = np.clip(rh, 0, 100)

        precip_mm = tp * days_in_month * 1000

        for m_idx, ts in enumerate(times):
            rows.append({
                "annee": int(ts.year),
                "mois": int(ts.month),
                "temp_moy": round(float(t2m[m_idx]), 2) if not math.isnan(t2m[m_idx]) else np.nan,
                "precip_mm": round(float(precip_mm[m_idx]), 2) if not math.isnan(precip_mm[m_idx]) else np.nan,
                "humidite_pct": round(float(rh[m_idx]), 1) if not math.isnan(rh[m_idx]) else np.nan,
            })

        ds.close()

    df = pd.DataFrame(rows)
    df = df.groupby(["annee", "mois"], as_index=False).mean(numeric_only=True)
    df = df.sort_values(["annee", "mois"]).reset_index(drop=True)
    logger.info("Climat Azilal : %d mois (%d-%d)", len(df), df["annee"].min(), df["annee"].max())
    return df


# ---------------------------------------------------------------------------
# 3. Cas LCT Azilal
# ---------------------------------------------------------------------------
def extract_azilal_cases() -> pd.DataFrame:
    lct = pd.read_csv(config.LCT_CSV)
    az = lct[lct["Province"].str.strip().str.lower() == "azilal"].copy()
    az = az.dropna(subset=["Mois_Diagnostic", "Annee_Source"])
    az["Mois_Diagnostic"] = az["Mois_Diagnostic"].astype(int)
    az["Annee_Source"] = az["Annee_Source"].astype(int)

    monthly = az.groupby(["Annee_Source", "Mois_Diagnostic"]).size().reset_index(name="cas")
    monthly = monthly.rename(columns={"Annee_Source": "annee", "Mois_Diagnostic": "mois"})
    logger.info("Cas Azilal : %d mois", len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# 4. Fusion + lags
# ---------------------------------------------------------------------------
def build_lagged_dataset(cases: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(cases, climate, on=["annee", "mois"], how="outer")
    df = df.sort_values(["annee", "mois"]).reset_index(drop=True)
    df["cas"] = df["cas"].fillna(0)

    df["date"] = pd.to_datetime(df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01")
    df = df.set_index("date").sort_index()

    for lag in LAGS:
        for var in CLIMATE_VARS:
            df[f"{var}_lag{lag}"] = df[var].shift(-lag)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 5. Graphiques dual-axis style demande
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sauvé : %s", path)


def compute_corr(df: pd.DataFrame, var: str, lag: int) -> tuple[float, float]:
    col = f"{var}_lag{lag}"
    sub = df[["cas", col]].dropna()
    if len(sub) < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(sub["cas"], sub[col])
    return r, p


def plot_dual_axis_grid(df: pd.DataFrame):
    """
    Grille 2x2 : pour chaque lag, cas (barres) vs température (ligne).
    Style inspiré de l'exemple utilisateur.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, lag in enumerate(LAGS):
        ax = axes[idx]
        col = f"temp_moy_lag{lag}"
        r, p = compute_corr(df, "temp_moy", lag)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        # Cas (barres)
        ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.7, width=20, label="Cas observés")
        ax.set_ylabel("Cas / mois", color="tab:gray", fontsize=11)
        ax.tick_params(axis="y", labelcolor="tab:gray")
        ax.set_ylim(0, df["cas"].max() * 1.1)

        # Température retardée (ligne)
        ax2 = ax.twinx()
        ax2.plot(df["date"], df[col], color="tab:red", linewidth=2, label=f"Temp. (lag {lag})")
        ax2.set_ylabel("Température retardée (°C)", color="tab:red", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0, df["temp_moy"].max() * 1.2)

        # Titre avec stats
        ax.set_title(f"Lag {lag} mois — r = {r:.3f}, p = {p:.4f} {sig}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Légende combinée
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.suptitle("Azilal — Cas vs température retardée (2009-2020)", fontsize=14, fontweight="bold")
    _save(fig, "graphcycle_dual_temp_lags.png")


def plot_dual_axis_single(df: pd.DataFrame, var: str, lag: int):
    """
    Plot dual-axis pour une variable et un lag donné.
    """
    label, color = CLIMATE_VARS[var]
    col = f"{var}_lag{lag}"
    r, p = compute_corr(df, var, lag)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Cas vs {label} retardée (lag {lag} mois) — Azilal", fontsize=14, fontweight="bold")

    # Cas (barres)
    ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.7, width=20, label="Cas observés")
    ax.set_ylabel("Cas / mois", color="tab:gray", fontsize=11)
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_ylim(0, df["cas"].max() * 1.15)

    # Climat (ligne)
    ax2 = ax.twinx()
    ax2.plot(df["date"], df[col], color=color, linewidth=2.5, label=f"{label} (lag {lag})")
    ax2.set_ylabel(f"{label} retardée", color=color, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, df[var].max() * 1.2)

    # Stats
    ax.set_title(f"Corrélation : r = {r:.3f}, p = {p:.4f} {sig}", fontsize=11)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    _save(fig, f"graphcycle_dual_{var}_lag{lag}.png")


def plot_cross_correlation(df: pd.DataFrame):
    """
    Cross-corrélation entre cas et chaque variable climatique
    pour identifier le lag optimal.
    """
    lags_range = range(0, 13)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (var, (label, color)) in enumerate(CLIMATE_VARS.items()):
        ax = axes[idx]
        corrs = []
        pvals = []
        for lag in lags_range:
            col = f"{var}_lag{lag}"
            sub = df[["cas", col]].dropna()
            if len(sub) < 5:
                corrs.append(np.nan)
                pvals.append(np.nan)
                continue
            r, p = stats.pearsonr(sub["cas"], sub[col])
            corrs.append(r)
            pvals.append(p)

        ax.bar(lags_range, corrs, color=color, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (mois)")
        ax.set_ylabel("Corrélation Pearson")
        ax.set_title(f"{label}\nCross-corrélation avec les cas")
        ax.grid(axis="y", alpha=0.3)

        # Annoter le lag max
        corrs_arr = np.array(corrs)
        valid = ~np.isnan(corrs_arr)
        if valid.any():
            best_idx = np.argmax(np.abs(corrs_arr[valid]))
            best_lag = list(np.array(lags_range)[valid])[best_idx]
            best_r = corrs_arr[valid][best_idx]
            ax.axvline(best_lag, color="red", linestyle="--", linewidth=1.5, label=f"Max |r| (lag {best_lag})")
            ax.legend()

    fig.suptitle("Cross-corrélation : cas vs variables climatiques — Azilal", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_cross_correlation.png")


def plot_target_year_focus(df: pd.DataFrame, year: int):
    """
    Focus sur une année cible : cas + température retardée (tous lags).
    """
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                    "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    x = np.arange(12)

    sub = df[df["annee"] == year].sort_values("mois")
    if sub.empty:
        logger.warning(f"Année {year} absente")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Azilal — {year} : Cas vs température retardée", fontsize=14, fontweight="bold")

    # Cas (barres)
    ax.bar(x, sub["cas"].values, color="tab:gray", alpha=0.6, width=0.4, label="Cas observés")
    ax.set_ylabel("Cas / mois", color="tab:gray", fontsize=11)
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_ylim(0, sub["cas"].max() * 1.2)

    # Température pour chaque lag
    ax2 = ax.twinx()
    for lag in LAGS:
        col = f"temp_moy_lag{lag}"
        vals = sub[col].values
        ax2.plot(x, vals, linewidth=2.5, label=f"Lag {lag}")

    ax2.set_ylabel("Température (°C)", color="tab:red", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, df["temp_moy"].max() * 1.2)

    # Stats
    stats_text = "Corrélations :\n"
    for lag in LAGS:
        col = f"temp_moy_lag{lag}"
        ssub = sub[["cas", col]].dropna()
        if len(ssub) >= 4:
            r, p = stats.pearsonr(ssub["cas"], ssub[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            stats_text += f"  lag {lag}: r={r:.2f}{sig}\n"

    ax2.set_title(stats_text.strip(), fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, f"graphcycle_focus_{year}.png")


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("RÉSUMÉ : CAS vs TEMPÉRATURE RETARDÉE — AZILAL")
    print("=" * 75)
    print(f"{'Lag':>5} {'r':>7} {'R²':>7} {'p-value':>10} {'Signif':>7}")
    print("-" * 75)
    for lag in LAGS:
        r, p = compute_corr(df, "temp_moy", lag)
        r2 = r ** 2 if not np.isnan(r) else np.nan
        sig = "OUI ***" if p < 0.001 else "OUI **" if p < 0.01 else "OUI *" if p < 0.05 else "NON"
        print(f"{lag:>5} {r:>7.3f} {r2:>7.3f} {p:>10.4f} {sig:>7}")
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    config.ensure_dirs()

    logger.info("=== Azilal Dual-Axis Lag Analysis ===")

    lat, lon = get_azilal_coords()
    climate = extract_azilal_climate(lat, lon)
    cases = extract_azilal_cases()
    df = build_lagged_dataset(cases, climate)

    # Sauvegarder
    out_csv = config.OUTPUTS / "processed" / "azilal_climate_cases_lags.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Dataset : %s", out_csv)

    # Résumé console
    print_summary(df)

    # Graphiques
    plot_dual_axis_grid(df)
    plot_cross_correlation(df)

    for var in CLIMATE_VARS:
        for lag in LAGS:
            plot_dual_axis_single(df, var, lag)

    for year in [2010, 2015, 2020]:
        plot_target_year_focus(df, year)

    logger.info("=== Terminé ===")
    print(f"\nFigures dans : {config.FIGURES}")


if __name__ == "__main__":
    main()
