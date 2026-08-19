"""
azilal_best_lag.py
==================
Graphiques interprétables : cas vs climat avec lag optimal.
Style : barres cas + ligne climat retardée (dual-axis).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

MAX_LAG = 12
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

    for lag in range(MAX_LAG + 1):
        for var in CLIMATE_VARS:
            df[f"{var}_lag{lag}"] = df[var].shift(-lag)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 5. Cross-corrélation
# ---------------------------------------------------------------------------
def cross_correlation(df: pd.DataFrame, var: str, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """Calcule r et p pour chaque lag."""
    records = []
    for lag in range(max_lag + 1):
        col = f"{var}_lag{lag}"
        sub = df[["cas", col]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub["cas"], sub[col])
        records.append({"lag": lag, "r": r, "p": p, "n": len(sub)})
    return pd.DataFrame(records)


def find_best_lag(cc_df: pd.DataFrame) -> tuple[int, float]:
    """Trouve le lag avec |r| maximal."""
    if cc_df.empty:
        return 0, np.nan
    cc_df = cc_df.copy()
    cc_df["abs_r"] = cc_df["r"].abs()
    best = cc_df.loc[cc_df["abs_r"].idxmax()]
    return int(best["lag"]), best["r"]


# ---------------------------------------------------------------------------
# 6. Graphiques
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sauvé : %s", path)


def plot_dual_axis(df: pd.DataFrame, var: str, lag: int, title_suffix: str = ""):
    """
    Dual-axis : cas (barres) vs variable retardée (ligne).
    Style exactement comme l'exemple utilisateur.
    """
    label, color = CLIMATE_VARS[var]
    col = f"{var}_lag{lag}"

    r, p = stats.pearsonr(df["cas"], df[col]) if df[[col]].dropna().shape[0] > 5 else (np.nan, np.nan)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Cas vs {label} retardée (lag {lag} mois) — Azilal{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Cas (barres grises)
    ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.7, width=20, label="Cas observés")
    ax.set_ylabel("Cas / mois", color="tab:gray", fontsize=11)
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_ylim(0, max(df["cas"].max() * 1.15, 10))

    # Variable retardée (ligne colorée)
    ax2 = ax.twinx()
    ax2.plot(df["date"], df[col], color=color, linewidth=2.5, label=f"{label} (lag {lag})")
    ax2.set_ylabel(f"{label} retardée", color=color, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, df[var].max() * 1.2)

    # Titre stats
    ax.set_title(f"Corrélation : r = {r:.3f}, p = {p:.4f} {sig}", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Légende combinée
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    _save(fig, f"graphcycle_bestlag_{var}.png")


def plot_cross_correlation(df: pd.DataFrame):
    """Cross-corrélation pour toutes les variables."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (var, (label, color)) in enumerate(CLIMATE_VARS.items()):
        ax = axes[idx]
        cc = cross_correlation(df, var)

        bars = ax.bar(cc["lag"], cc["r"], color=color, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        # Annoter le lag max
        best_lag, best_r = find_best_lag(cc)
        ax.axvline(best_lag, color="red", linestyle="--", linewidth=2, label=f"Max |r| = lag {best_lag}")
        ax.legend()

        ax.set_xlabel("Lag (mois)")
        ax.set_ylabel("Corrélation Pearson")
        ax.set_title(f"{label}\nCross-corrélation")
        ax.grid(axis="y", alpha=0.3)

        # Annotation valeur max
        ax.annotate(
            f"r = {best_r:.3f}",
            xy=(best_lag, best_r),
            xytext=(best_lag + 1, best_r + 0.05 if best_r > 0 else best_r - 0.05),
            fontsize=9,
            color="red",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    fig.suptitle("Cross-corrélation : cas vs variables climatiques — Azilal", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_cross_correlation.png")


def plot_lag_comparison(df: pd.DataFrame):
    """
    Compare R² pour chaque variable à travers tous les lags.
    Met en évidence le lag optimal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (var, (label, color)) in enumerate(CLIMATE_VARS.items()):
        ax = axes[idx]
        cc = cross_correlation(df, var)
        cc["r_squared"] = cc["r"] ** 2

        ax.bar(cc["lag"], cc["r_squared"], color=color, alpha=0.7)
        ax.set_xlabel("Lag (mois)")
        ax.set_ylabel("R²")
        ax.set_title(f"{label}\nVariance expliquée par lag")
        ax.grid(axis="y", alpha=0.3)

        # Ligne rouge au seuil p < 0.05
        # (approximatif : on colorie les barres significatives différemment)
        for _, row in cc.iterrows():
            if row["p"] < 0.05:
                ax.axvline(row["lag"], color="green", alpha=0.3, linewidth=3)

    fig.suptitle("R² par lag — Azilal (zones vertes = p < 0.05)", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_r2_by_lag.png")


def plot_event_studies(df: pd.DataFrame, n_events: int = 6):
    """
    Zoom sur les n plus grands pics de cas pour montrer le décalage.
    """
    # Trouver les pics
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(df["cas"].values, height=df["cas"].quantile(0.8), distance=6)
    peak_dates = df.iloc[peaks]["date"].values[:n_events]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, peak_date in enumerate(peak_dates):
        ax = axes[idx]
        peak_idx = df[df["date"] == peak_date].index[0]

        # Fenêtre de 6 mois avant à 3 mois après le pic
        start_idx = max(0, peak_idx - 6)
        end_idx = min(len(df), peak_idx + 4)
        window = df.iloc[start_idx:end_idx].copy()

        if window.empty:
            continue

        # Cas (barres)
        ax.bar(window["date"], window["cas"], color="tab:gray", alpha=0.7, width=15, label="Cas")
        ax.set_ylabel("Cas", color="tab:gray")
        ax.tick_params(axis="y", labelcolor="tab:gray")

        # Température lag 2
        ax2 = ax.twinx()
        ax2.plot(window["date"], window["temp_moy_lag2"], color="tab:red", linewidth=2.5, label="T. (lag 2)")
        ax2.set_ylabel("Température (°C)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Marquer le pic de cas
        ax.axvline(peak_date, color="black", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(f"Pic : {pd.Timestamp(peak_date).strftime('%b %Y')}")
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.suptitle("Études d'événements : pics de cas + température retardée (lag 2) — Azilal", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_event_studies.png")


def plot_summary_table(df: pd.DataFrame):
    """Affiche un tableau récapitulatif dans la console."""
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES CROSS-CORRÉLATIONS — AZILAL")
    print("=" * 80)

    for var, (label, _) in CLIMATE_VARS.items():
        cc = cross_correlation(df, var)
        if cc.empty:
            continue
        best_lag, best_r = find_best_lag(cc)
        best_row = cc[cc["lag"] == best_lag].iloc[0]
        sig = "***" if best_row["p"] < 0.001 else "**" if best_row["p"] < 0.01 else "*" if best_row["p"] < 0.05 else "ns"

        print(f"\n{label}")
        print(f"  Lag optimal : {best_lag} mois")
        print(f"  r = {best_r:.3f}, p = {best_row['p']:.4f} {sig}")
        print(f"  R² = {best_r**2:.3f} ({best_r**2*100:.1f}% de variance expliquée)")

        # Top 3 lags
        top3 = cc.nlargest(3, "r", keep="all")[["lag", "r", "p"]].head(3)
        print("  Top 3 lags :")
        for _, row in top3.iterrows():
            sig2 = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "ns"
            print(f"    lag {int(row['lag'])}: r={row['r']:.3f}, p={row['p']:.4f} {sig2}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    config.ensure_dirs()

    logger.info("=== Azilal Best Lag Analysis ===")

    lat, lon = get_azilal_coords()
    climate = extract_azilal_climate(lat, lon)
    cases = extract_azilal_cases()
    df = build_lagged_dataset(cases, climate)

    # Sauvegarder
    out_csv = config.OUTPUTS / "processed" / "azilal_climate_cases_lags.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Dataset : %s", out_csv)

    # Résumé
    plot_summary_table(df)

    # Graphiques
    plot_cross_correlation(df)
    plot_lag_comparison(df)

    # Plot pour chaque variable à son lag optimal
    for var in CLIMATE_VARS:
        cc = cross_correlation(df, var)
        best_lag, _ = find_best_lag(cc)
        plot_dual_axis(df, var, best_lag, f" (lag optimal = {best_lag})")

    # Event studies
    plot_event_studies(df, n_events=6)

    logger.info("=== Terminé ===")
    print(f"\nFigures dans : {config.FIGURES}")


if __name__ == "__main__":
    main()
