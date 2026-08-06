"""
graphcycle.py
=============
Analyse Azilal : lien retardé entre climat et cas de leishmaniose.

Étapes :
  1. Extraire les coordonnées d'Azilal depuis province_table.csv
  2. Extraire les séries climatiques ERA5 mensuelles pour Azilal
  3. Agréger les cas LCT par année x mois
  4. Pour chaque lag (0, 2, 4, 6 mois) :
     - scatter cas vs climat + fit linéaire + R²
     - calcul corrélation Pearson
  5. Heatmap de corrélation par variable x lag
  6. Courbes de réponse lissées (lowess) cas vs climat
  7. Séries temporelles superposées (cas, T, precip, humidité)

Sorties : figures dans outputs/figures/graphcycle_*.png
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
CLIMATE_VARS = {
    "temp_moy": ("Température moyenne (°C)", "tab:red"),
    "precip_mm": ("Précipitation (mm)", "tab:blue"),
    "humidite_pct": ("Humidité relative (%)", "tab:green"),
}


# ---------------------------------------------------------------------------
# 1. Coordonnées Azilal
# ---------------------------------------------------------------------------
def get_azilal_coords() -> tuple[float, float]:
    """Retourne (lat, lon) d'Azilal depuis province_table.csv."""
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
    """
    Extrait la série climatique mensuelle Azilal depuis les NetCDF ERA5.
    Retourne un DataFrame [annee, mois, temp_moy, precip_mm, humidite_pct].
    """
    # Chercher les fichiers mensuels année par année
    monthly_files = sorted(config.RAW.glob("era5_morocco_*_monthly.nc"))
    if not monthly_files:
        raise FileNotFoundError(
            f"Aucun fichier era5_morocco_*_monthly.nc dans {config.RAW}"
        )

    rows = []
    for nc_path in monthly_files:
        year = int(nc_path.stem.split("_")[-2])
        ds = xr.open_dataset(nc_path)

        # Détection dynamique des noms de variables
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

        # Nearest neighbor
        iy = int(np.abs(lats - lat).argmin())
        ix = int(np.abs(lons - lon).argmin())

        t2m = ds[t2m_name].values[:, iy, ix] - 273.15  # K -> °C
        d2m = ds[d2m_name].values[:, iy, ix] - 273.15
        tp = ds[tp_name].values[:, iy, ix]  # m/jour

        # Jours dans chaque mois
        times = pd.to_datetime(ds[time_name].values)
        days_in_month = np.array([t.days_in_month for t in times])

        # Humidité relative (formule Magnus-Tetens)
        def es(temp_c):
            return 6.112 * math.exp((17.62 * temp_c) / (243.12 + temp_c))

        rh = np.array([100 * es(d2m[m]) / es(t2m[m]) if not (math.isnan(d2m[m]) or math.isnan(t2m[m])) else np.nan
                       for m in range(len(t2m))])
        rh = np.clip(rh, 0, 100)

        # Précipitation totale mensuelle (m/jour -> mm/mois)
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
    # Agréger par année x mois (moyenne si plusieurs fichiers se chevauchent)
    df = df.groupby(["annee", "mois"], as_index=False).mean(numeric_only=True)
    df = df.sort_values(["annee", "mois"]).reset_index(drop=True)
    logger.info("Climat Azilal extrait : %d mois (%d-%d)", len(df), df["annee"].min(), df["annee"].max())
    return df


# ---------------------------------------------------------------------------
# 3. Cas LCT Azilal par mois
# ---------------------------------------------------------------------------
def extract_azilal_cases() -> pd.DataFrame:
    """Retourne un DataFrame [annee, mois, cas] pour Azilal."""
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
# 4. Fusion et création des lags
# ---------------------------------------------------------------------------
def build_lagged_dataset(cases: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne cas et climat, puis crée les colonnes décalées pour chaque lag.
    Pour un lag de k mois, la climat du mois M est associée aux cas du mois M+k.
    """
    # Merge complet (toutes les combinaisons année x mois présentes dans les deux)
    df = pd.merge(cases, climate, on=["annee", "mois"], how="outer")
    df = df.sort_values(["annee", "mois"]).reset_index(drop=True)
    df["cas"] = df["cas"].fillna(0)

    # Créer un index temps pour le décalage propre
    df["date"] = pd.to_datetime(df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01")
    df = df.set_index("date").sort_index()

    for lag in LAGS:
        for var in CLIMATE_VARS:
            df[f"{var}_lag{lag}"] = df[var].shift(-lag)  # climat décalée vers le passé

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 5. Statistiques par lag
# ---------------------------------------------------------------------------
def compute_lag_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule corrélation Pearson et p-value pour chaque (variable x lag)."""
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


def plot_scatter_by_lag(df: pd.DataFrame, stats_df: pd.DataFrame):
    """Scatter plots cas vs climat pour chaque lag (grille 3x4)."""
    vars_list = list(CLIMATE_VARS.keys())
    fig, axes = plt.subplots(len(vars_list), len(LAGS), figsize=(16, 9))
    axes = np.atleast_2d(axes)

    for i, var in enumerate(vars_list):
        color = CLIMATE_VARS[var][1]
        for j, lag in enumerate(LAGS):
            ax = axes[i, j]
            col = f"{var}_lag{lag}"
            sub = df[["cas", col]].dropna()
            if len(sub) < 5:
                ax.set_visible(False)
                continue

            ax.scatter(sub[col], sub["cas"], alpha=0.7, edgecolors="k", linewidth=0.5, color=color)

            # Fit linéaire
            slope, intercept, r_value, p_value, _ = stats.linregress(sub[col], sub["cas"])
            x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="tab:red", linewidth=1.5)

            # R² et p-value
            s = stats_df[(stats_df["variable"] == var) & (stats_df["lag"] == lag)]
            r2 = s["r"].values[0] ** 2 if len(s) else np.nan
            pval = s["p"].values[0] if len(s) else np.nan
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax.set_title(f"Lag {lag} mois\nR²={r2:.2f} {sig}", fontsize=10)
            ax.set_xlabel(CLIMATE_VARS[var][0])
            ax.set_ylabel("Nombre de cas")
            ax.grid(True, alpha=0.4)

    fig.suptitle("Cas de leishmaniose vs variables climatiques — Azilal (2009-2020)", fontsize=14, fontweight="bold")
    _save(fig, "graphcycle_scatter_lags.png")


def plot_heatmap_corr_lag(stats_df: pd.DataFrame):
    """Heatmap des corrélations par variable x lag."""
    pivot = stats_df.pivot(index="variable", columns="lag", values="r")
    pivot.index = [CLIMATE_VARS[v][0] for v in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, linewidths=0.5, ax=ax, cbar_kws={"label": "Corrélation Pearson"})
    ax.set_title("Corrélation cas vs climat par lag — Azilal", fontsize=13)
    ax.set_xlabel("Lag (mois)")
    ax.set_ylabel("Variable climatique")
    _save(fig, "graphcycle_heatmap_lag.png")


def plot_response_curves(df: pd.DataFrame):
    """Courbes de réponse lissées (lowess) pour chaque variable climatique."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, var in enumerate(CLIMATE_VARS):
        ax = axes[idx]
        color = CLIMATE_VARS[var][1]

        for lag in LAGS:
            col = f"{var}_lag{lag}"
            sub = df[["cas", col]].dropna().sort_values(col)
            if len(sub) < 10:
                continue
            # Lowess
            lowess = stats.nonparametric.lowess(sub["cas"].values, sub[col].values, frac=0.6)
            ax.plot(lowess[:, 0], lowess[:, 1], label=f"Lag {lag}", linewidth=2)

        ax.set_xlabel(CLIMATE_VARS[var][0])
        ax.set_ylabel("Cas (lissé)")
        ax.set_title(CLIMATE_VARS[var][0])
        ax.legend(title="Lag")
        ax.grid(True, alpha=0.4)

    fig.suptitle("Courbes de réponse : cas vs climat (lowess) — Azilal", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_response_curves.png")


def plot_time_series(df: pd.DataFrame):
    """Séries temporelles superposées : cas, T, precip, humidité."""
    df_ts = df.dropna(subset=["temp_moy", "precip_mm", "humidite_pct"]).copy()
    if df_ts.empty:
        logger.warning("Données insuffisantes pour la série temporelle.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Cas
    ax = axes[0]
    ax.bar(df_ts["date"], df_ts["cas"], color="tab:gray", alpha=0.8)
    ax.set_ylabel("Cas")
    ax.set_title("Cas de leishmaniose — Azilal (2009-2020)")
    ax.grid(True, alpha=0.4)

    # Température
    ax = axes[1]
    ax.plot(df_ts["date"], df_ts["temp_moy"], color="tab:red", linewidth=1.5)
    ax.set_ylabel("Temp (°C)")
    ax.grid(True, alpha=0.4)

    # Précipitation
    ax = axes[2]
    ax.bar(df_ts["date"], df_ts["precip_mm"], color="tab:blue", alpha=0.7)
    ax.set_ylabel("Precip (mm)")
    ax.grid(True, alpha=0.4)

    # Humidité
    ax = axes[3]
    ax.plot(df_ts["date"], df_ts["humidite_pct"], color="tab:green", linewidth=1.5)
    ax.set_ylabel("Humidité (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, "graphcycle_time_series.png")


def plot_monthly_climatology(df: pd.DataFrame):
    """Profil climatologique mensuel (moyenne sur toutes les années)."""
    monthly_clim = df.groupby("mois")[["cas", "temp_moy", "precip_mm", "humidite_pct"]].mean().reset_index()
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    monthly_clim["mois_nom"] = monthly_clim["mois"].map(lambda x: month_names[x - 1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (col, title, color) in enumerate([
        ("cas", "Cas moyens par mois", "tab:gray"),
        ("temp_moy", "Température moyenne (°C)", "tab:red"),
        ("precip_mm", "Précipitation moyenne (mm)", "tab:blue"),
        ("humidite_pct", "Humidité relative moyenne (%)", "tab:green"),
    ]):
        ax = axes[idx]
        ax.bar(monthly_clim["mois_nom"], monthly_clim[col], color=color, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Profil mensuel climatique et épidémiologique — Azilal (2009-2020)", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_monthly_climatology.png")


def plot_p_value_by_lag(stats_df: pd.DataFrame):
    """Barres des p-values par variable et lag (échelle -log10)."""
    stats_df["neg_log10_p"] = -np.log10(stats_df["p"].clip(lower=1e-10))
    stats_df["significant"] = stats_df["p"] < 0.05

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {True: "tab:green", False: "tab:gray"}
    sns.barplot(data=stats_df, x="lag", y="neg_log10_p", hue="variable",
                palette=[CLIMATE_VARS[v][1] for v in stats_df["variable"].unique()], ax=ax)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, label="p = 0.05")
    ax.set_title("Significativité des corrélations par lag — Azilal", fontsize=13)
    ax.set_xlabel("Lag (mois)")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(title="Variable", loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    _save(fig, "graphcycle_pvalue_lag.png")


# ---------------------------------------------------------------------------
# 7. Résumé console
# ---------------------------------------------------------------------------
def print_summary(stats_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES CORRÉLATIONS PAR LAG — AZILAL")
    print("=" * 70)
    print(f"{'Variable':<30} {'Lag':>5} {'r':>7} {'p-value':>10} {'Signif':>7}")
    print("-" * 70)
    for _, row in stats_df.iterrows():
        sig = "OUI ***" if row["p"] < 0.001 else "OUI **" if row["p"] < 0.01 else "OUI *" if row["p"] < 0.05 else "NON"
        print(f"{row['label']:<30} {row['lag']:>5} {row['r']:>7.3f} {row['p']:>10.4f} {sig:>7}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    config.ensure_dirs()

    logger.info("=== GraphCycle Azilal — Démarrage ===")

    # 1. Coordonnées
    lat, lon = get_azilal_coords()

    # 2. Climat
    climate = extract_azilal_climate(lat, lon)

    # 3. Cas
    cases = extract_azilal_cases()

    # 4. Fusion + lags
    df = build_lagged_dataset(cases, climate)

    # Sauvegarder le dataset consolidé
    out_csv = config.OUTPUTS / "processed" / "azilal_climate_cases_lags.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Dataset consolidé sauvegardé : %s", out_csv)

    # 5. Stats
    stats_df = compute_lag_stats(df)
    print_summary(stats_df)

    # 6. Graphiques
    plot_scatter_by_lag(df, stats_df)
    plot_heatmap_corr_lag(stats_df)
    plot_response_curves(df)
    plot_time_series(df)
    plot_monthly_climatology(df)
    plot_p_value_by_lag(stats_df)

    logger.info("=== GraphCycle Azilal — Terminé ===")
    print(f"\nToutes les figures sont dans : {config.FIGURES}")


if __name__ == "__main__":
    main()