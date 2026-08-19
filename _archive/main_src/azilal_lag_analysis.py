"""
azilal_lag_analysis.py
======================
Analyse Azilal : lien retardé cas vs climat.

Sorties :
  - Courbes temporelles par lag (cas superposé à T, precip, humidité)
  - Tableau résumé des corrélations par lag
  - Exemples sur années cibles : 2010, 2015, 2020
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
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

LAGS = [0, 2, 4, 6]
TARGET_YEARS = [2010, 2015, 2020]
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
        raise ValueError("Province 'Azilal' introuvable dans province_table.csv")
    lat = float(row.iloc[0]["lat"])
    lon = float(row.iloc[0]["lon"])
    logger.info("Azilal coords : lat=%.4f, lon=%.4f", lat, lon)
    return lat, lon


# ---------------------------------------------------------------------------
# 2. Extraction climat ERA5 pour Azilal
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
    logger.info("Climat Azilal extrait : %d mois (%d-%d)", len(df), df["annee"].min(), df["annee"].max())
    return df


# ---------------------------------------------------------------------------
# 3. Cas LCT Azilal par mois
# ---------------------------------------------------------------------------
def extract_azilal_cases() -> pd.DataFrame:
    lct = pd.read_csv(config.LCT_CSV)
    az = lct[lct["Province"].str.strip().str.lower() == "azilal"].copy()
    az = az.dropna(subset=["Mois_Diagnostic", "Annee_Source"])
    az["Mois_Diagnostic"] = az["Mois_Diagnostic"].astype(int)
    az["Annee_Source"] = az["Annee_Source"].astype(int)

    monthly = az.groupby(["Annee_Source", "Mois_Diagnostic"]).size().reset_index(name="cas")
    monthly = monthly.rename(columns={"Annee_Source": "annee", "Mois_Diagnostic": "mois"})
    logger.info("Cas Azilal : %d lignes (mois avec au moins 1 cas)", len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# 4. Fusion et lags
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
# 5. Statistiques par lag
# ---------------------------------------------------------------------------
def compute_lag_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for lag in LAGS:
        for var in CLIMATE_VARS:
            col = f"{var}_lag{lag}"
            sub = df[["cas", col]].dropna()
            if len(sub) < 5:
                continue
            r, p = stats.pearsonr(sub["cas"], sub[col])
            records.append({
                "lag": lag,
                "variable": var,
                "label": CLIMATE_VARS[var][0],
                "r": round(r, 3),
                "p": round(p, 4),
                "n": len(sub),
            })
    return pd.DataFrame(records).sort_values(["variable", "lag"])


# ---------------------------------------------------------------------------
# 6. Graphiques
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée : %s", path)


def plot_lag_time_series(df: pd.DataFrame, stats_df: pd.DataFrame):
    """
    Pour chaque lag, 3 subplots : cas vs T, precip, humidité.
    Les courbes climatiques sont décalées de `lag` mois dans le passé.
    """
    for lag in LAGS:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Azilal — Cas vs climat (lag {lag} mois)", fontsize=14, fontweight="bold")

        for idx, (var, (label, color)) in enumerate(CLIMATE_VARS.items()):
            ax = axes[idx]
            clim_col = f"{var}_lag{lag}"

            # Cas
            ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.6, width=20, label="Cas")
            ax.set_ylabel("Cas", color="tab:gray")
            ax.tick_params(axis="y", labelcolor="tab:gray")
            ax.grid(True, alpha=0.3)

            # Climat décalée
            ax2 = ax.twinx()
            ax2.plot(df["date"], df[clim_col], color=color, linewidth=2, label=f"{label} (lag {lag})")
            ax2.set_ylabel(label, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            # Légende combinée
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

            # Annotation stats
            s = stats_df[(stats_df["variable"] == var) & (stats_df["lag"] == lag)]
            if len(s):
                r, p = s.iloc[0]["r"], s.iloc[0]["p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax2.set_title(f"r = {r:.3f}, p = {p:.4f} {sig}", fontsize=10)

        axes[-1].set_xlabel("Date")
        _save(fig, f"graphcycle_lag{lag}_timeseries.png")


def plot_target_years(df: pd.DataFrame, stats_df: pd.DataFrame):
    """
    Exemples sur 3 années cibles : 2010, 2015, 2020.
    Pour chaque année, superpose cas et climat sur 12 mois.
    """
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                    "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    x = np.arange(12)

    for year in TARGET_YEARS:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Azilal — {year} : Cas vs climat (lag 0, 2, 4, 6 mois)", fontsize=14, fontweight="bold")

        for idx, (var, (label, color)) in enumerate(CLIMATE_VARS.items()):
            ax = axes[idx]
            ax.set_ylabel("Cas", color="tab:gray")
            ax.tick_params(axis="y", labelcolor="tab:gray")
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            ax2.set_ylabel(label, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            # Cas de l'année
            sub_year = df[df["annee"] == year].sort_values("mois")
            if sub_year.empty:
                continue

            ax.bar(x, sub_year["cas"].values, color="tab:gray", alpha=0.6, width=0.4, label="Cas")

            # Climat pour chaque lag
            for lag in LAGS:
                col = f"{var}_lag{lag}"
                vals = sub_year[col].values
                ax2.plot(x, vals, linewidth=2, label=f"Lag {lag}")

            # Stats pour l'année (corrélation sur l'année complète)
            corr_text = ""
            for lag in LAGS:
                col = f"{var}_lag{lag}"
                ssub = sub_year[["cas", col]].dropna()
                if len(ssub) >= 4:
                    r, p = stats.pearsonr(ssub["cas"], ssub[col])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    corr_text += f"  lag{lag}: r={r:.2f}{sig}\n"

            ax2.set_title(f"{label}\n{corr_text.strip()}", fontsize=9)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(month_names)
        axes[-1].set_xlabel("Mois")
        _save(fig, f"graphcycle_target_{year}.png")


def plot_correlation_summary(stats_df: pd.DataFrame):
    """Tableau résumé + barres des R² par lag."""
    # Tableau console
    print("\n" + "=" * 75)
    print("RÉSUMÉ DES CORRÉLATIONS PAR LAG — AZILAL")
    print("=" * 75)
    print(f"{'Variable':<25} {'Lag':>5} {'r':>7} {'R²':>7} {'p-value':>10} {'Signif':>7}")
    print("-" * 75)
    for _, row in stats_df.iterrows():
        sig = "OUI ***" if row["p"] < 0.001 else "OUI **" if row["p"] < 0.01 else "OUI *" if row["p"] < 0.05 else "NON"
        r2 = row["r"] ** 2
        print(f"{row['label']:<25} {row['lag']:>5} {row['r']:>7.3f} {r2:>7.3f} {row['p']:>10.4f} {sig:>7}")
    print("=" * 75 + "\n")

    # Graphique barres R²
    stats_df["r_squared"] = stats_df["r"] ** 2
    stats_df["significant"] = stats_df["p"] < 0.05

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {var: CLIMATE_VARS[var][1] for var in CLIMATE_VARS}
    sns.barplot(data=stats_df, x="lag", y="r_squared", hue="variable", palette=palette, ax=ax)
    ax.set_title("R² par variable et lag — Azilal", fontsize=13)
    ax.set_xlabel("Lag (mois)")
    ax.set_ylabel("R²")
    ax.legend(title="Variable", loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    _save(fig, "graphcycle_r2_summary.png")


def plot_all_years_overview(df: pd.DataFrame):
    """Vue d'ensemble : cas mensuels sur toutes les années avec les 3 années cibles surlignées."""
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                    "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    monthly_clim = df.groupby("mois")[["cas", "temp_moy", "precip_mm", "humidite_pct"]].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (col, title, color) in enumerate([
        ("cas", "Cas moyens par mois", "tab:gray"),
        ("temp_moy", "Température moyenne (°C)", "tab:red"),
        ("precip_mm", "Précipitation moyenne (mm)", "tab:blue"),
        ("humidite_pct", "Humidité relative moyenne (%)", "tab:green"),
    ]):
        ax = axes[idx]
        ax.bar(monthly_clim["mois"], monthly_clim[col], color=color, alpha=0.8)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, rotation=45)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.4)

        # Surligner les mois des années cibles
        for yr in TARGET_YEARS:
            sub_yr = df[(df["annee"] == yr)][["mois", col]].dropna()
            if not sub_yr.empty:
                ax.scatter(sub_yr["mois"], sub_yr[col], color="black", s=50, zorder=5, marker="o")

    fig.suptitle("Profil mensuel moyen — Azilal (2009-2020)\nPoints noirs : années 2010, 2015, 2020",
                 fontsize=12, fontweight="bold")
    _save(fig, "graphcycle_overview.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    config.ensure_dirs()

    logger.info("=== Azilal Lag Analysis — Démarrage ===")

    lat, lon = get_azilal_coords()
    climate = extract_azilal_climate(lat, lon)
    cases = extract_azilal_cases()
    df = build_lagged_dataset(cases, climate)

    out_csv = config.OUTPUTS / "processed" / "azilal_climate_cases_lags.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Dataset sauvegardé : %s", out_csv)

    stats_df = compute_lag_stats(df)
    plot_correlation_summary(stats_df)
    plot_lag_time_series(df, stats_df)
    plot_target_years(df, stats_df)
    plot_all_years_overview(df)

    logger.info("=== Azilal Lag Analysis — Terminé ===")
    print(f"\nToutes les figures sont dans : {config.FIGURES}")


if __name__ == "__main__":
    main()
