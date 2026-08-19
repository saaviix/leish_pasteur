"""
azilal_temp_lag.py
==================
Graphique unique et propre : cas vs température retardée.
Style : barres grises = cas, ligne rouge = température (lag optimal).
Focus : identifier le lag où la température "précède" les pics de cas.
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


# ---------------------------------------------------------------------------
# 1. Coordonnées Azilal
# ---------------------------------------------------------------------------
def get_azilal_coords():
    pt = pd.read_csv(config.PROVINCE_TABLE)
    row = pt[pt["province"].str.strip().str.lower() == "azilal"]
    if row.empty:
        raise ValueError("Azilal introuvable dans province_table.csv")
    return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])


# ---------------------------------------------------------------------------
# 2. Extraction climat ERA5 pour Azilal
# ---------------------------------------------------------------------------
def extract_azilal_climate(lat, lon):
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

        def es(temp_c):
            return 6.112 * math.exp((17.62 * temp_c) / (243.12 + temp_c))

        rh = np.array([
            100 * es(d2m[m]) / es(t2m[m]) if not (math.isnan(d2m[m]) or math.isnan(t2m[m])) else np.nan
            for m in range(len(t2m))
        ])
        rh = np.clip(rh, 0, 100)

        for m_idx, ts in enumerate(times):
            # tp en ERA5 mensuel : m/jour (moyenne journalière du mois)
            # Pour obtenir le total mensuel : tp * nb_jours
            precip_mm = float(tp[m_idx]) * 1000 * ts.days_in_month if not math.isnan(tp[m_idx]) else np.nan

            rows.append({
                "annee": int(ts.year),
                "mois": int(ts.month),
                "temp_moy": round(float(t2m[m_idx]), 2) if not math.isnan(t2m[m_idx]) else np.nan,
                "precip_mm": round(precip_mm, 2),
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
def extract_azilal_cases():
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
def build_lagged_dataset(cases, climate):
    df = pd.merge(cases, climate, on=["annee", "mois"], how="outer")
    df = df.sort_values(["annee", "mois"]).reset_index(drop=True)
    df["cas"] = df["cas"].fillna(0)
    df["date"] = pd.to_datetime(df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01")
    df = df.set_index("date").sort_index()

    for lag in range(MAX_LAG + 1):
        df[f"temp_moy_lag{lag}"] = df["temp_moy"].shift(-lag)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 5. Cross-corrélation température
# ---------------------------------------------------------------------------
def cross_corr_temp(df):
    records = []
    for lag in range(MAX_LAG + 1):
        col = f"temp_moy_lag{lag}"
        sub = df[["cas", col]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub["cas"], sub[col])
        records.append({"lag": lag, "r": r, "p": p})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Graphiques
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sauvé : %s", path)


def plot_best_lag_dual(df):
    """
    Plot principal : cas (barres) vs température retardée au lag optimal.
    Style exactement comme l'exemple utilisateur.
    """
    cc = cross_corr_temp(df)
    cc["abs_r"] = cc["r"].abs()
    best_lag = int(cc.loc[cc["abs_r"].idxmax(), "lag"])
    best_r = cc.loc[cc["abs_r"].idxmax(), "r"]
    best_p = cc.loc[cc["abs_r"].idxmax(), "p"]

    col = f"temp_moy_lag{best_lag}"
    sig = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else "ns"

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Cas vs température retardée (lag {best_lag} mois) — Azilal", fontsize=14, fontweight="bold")

    # Cas (barres)
    ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.7, width=20, label="Cas observés")
    ax.set_ylabel("Cas / mois", color="tab:gray", fontsize=11)
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_ylim(0, df["cas"].max() * 1.15)

    # Température retardée (ligne)
    ax2 = ax.twinx()
    ax2.plot(df["date"], df[col], color="tab:red", linewidth=2.5, label=f"Température (lag {best_lag})")
    ax2.set_ylabel("Température retardée (°C)", color="tab:red", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, df["temp_moy"].max() * 1.2)

    # Stats
    ax.set_title(f"Corrélation : r = {best_r:.3f}, p = {best_p:.4f} {sig}  |  R² = {best_r**2:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    _save(fig, "graphcycle_temp_best_lag.png")

    # Impression console
    print(f"\nMeilleur lag pour la température : {best_lag} mois")
    print(f"  r = {best_r:.3f}, p = {best_p:.4f} {sig}")
    print(f"  R² = {best_r**2:.3f} ({best_r**2*100:.1f}% de variance expliquée)\n")


def plot_cross_correlation(df):
    """
    Cross-corrélation : montre à quel lag la température "colle" le mieux aux cas.
    """
    cc = cross_corr_temp(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:red" if p < 0.05 else "lightgray" for p in cc["p"]]
    ax.bar(cc["lag"], cc["r"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)

    best_lag = int(cc.loc[cc["r"].abs().idxmax(), "lag"])
    best_r = cc.loc[cc["r"].abs().idxmax(), "r"]
    ax.axvline(best_lag, color="blue", linestyle="--", linewidth=2, label=f"Lag optimal = {best_lag} mois")

    ax.set_xlabel("Lag (mois)", fontsize=11)
    ax.set_ylabel("Corrélation Pearson (r)", fontsize=11)
    ax.set_title("Cross-corrélation : cas vs température retardée — Azilal\nBarres rouges = p < 0.05", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "graphcycle_cross_corr_temp.png")


def plot_all_lags_grid(df):
    """
    Grille 3x4 : cas vs température pour lag 0,1,2,3,4,5,6,7,8,9,10,11.
    Permet de visualiser quel lag aligne le mieux les pics.
    """
    lags = list(range(0, MAX_LAG + 1))
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes = axes.flatten()

    cc = cross_corr_temp(df)
    cc_dict = cc.set_index("lag").to_dict("index")

    for idx, lag in enumerate(lags):
        ax = axes[idx]
        col = f"temp_moy_lag{lag}"

        ax.bar(df["date"], df["cas"], color="tab:gray", alpha=0.6, width=20, label="Cas")
        ax.set_ylabel("Cas", color="tab:gray", fontsize=9)
        ax.tick_params(axis="y", labelcolor="tab:gray", labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(df["date"], df[col], color="tab:red", linewidth=1.5, label=f"T. lag {lag}")
        ax2.set_ylabel("°C", color="tab:red", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=8)

        # Stats
        info = cc_dict.get(lag, {})
        r = info.get("r", np.nan)
        p = info.get("p", np.nan)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"Lag {lag} mois\nr={r:.2f} {sig}", fontsize=9)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)

    # Masquer les subplots vides
    for idx in range(len(lags), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cas vs température retardée — Tous les lags (0-12 mois) — Azilal", fontsize=14, fontweight="bold")
    _save(fig, "graphcycle_all_lags_grid.png")


def plot_event_studies(df):
    """
    Zoom sur les pics de cas les plus marquants.
    Montre clairement : la température monte, puis les cas montent.
    """
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(df["cas"].values, height=df["cas"].quantile(0.75), distance=6)
    peak_dates = df.iloc[peaks]["date"].values[:6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, peak_date in enumerate(peak_dates):
        ax = axes[idx]
        peak_idx = df[df["date"] == peak_date].index[0]
        start_idx = max(0, peak_idx - 6)
        end_idx = min(len(df), peak_idx + 4)
        window = df.iloc[start_idx:end_idx].copy()

        if window.empty:
            continue

        ax.bar(window["date"], window["cas"], color="tab:gray", alpha=0.7, width=15, label="Cas")
        ax.set_ylabel("Cas", color="tab:gray", fontsize=9)
        ax.tick_params(axis="y", labelcolor="tab:gray", labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(window["date"], window["temp_moy_lag2"], color="tab:red", linewidth=2, label="T. (lag 2)")
        ax2.set_ylabel("°C", color="tab:red", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=8)

        ax.axvline(peak_date, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"Pic : {pd.Timestamp(peak_date).strftime('%b %Y')}", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Études d'événements : pics de cas + température retardée (lag 2) — Azilal", fontsize=13, fontweight="bold")
    _save(fig, "graphcycle_event_studies.png")


def plot_summary_table(df):
    """Affiche le résumé des cross-corrélations dans la console."""
    cc = cross_corr_temp(df)
    cc["abs_r"] = cc["r"].abs()
    best_lag = int(cc.loc[cc["abs_r"].idxmax(), "lag"])
    best_r = cc.loc[cc["abs_r"].idxmax(), "r"]
    best_p = cc.loc[cc["abs_r"].idxmax(), "p"]

    print("\n" + "=" * 70)
    print("RÉSUMÉ : CAS vs TEMPÉRATURE RETARDÉE — AZILAL")
    print("=" * 70)
    print(f"{'Lag':>5} {'r':>7} {'|r|':>7} {'p-value':>10} {'Signif':>7}")
    print("-" * 70)
    for _, row in cc.iterrows():
        sig = "OUI ***" if row["p"] < 0.001 else "OUI **" if row["p"] < 0.01 else "OUI *" if row["p"] < 0.05 else "NON"
        marker = " <-- MEILLEUR" if int(row["lag"]) == best_lag else ""
        print(f"{int(row['lag']):>5} {row['r']:>7.3f} {abs(row['r']):>7.3f} {row['p']:>10.4f} {sig:>7}{marker}")
    print("=" * 70)
    sig = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else "ns"
    print(f"\n→ Lag optimal : {best_lag} mois")
    print(f"  r = {best_r:.3f}, p = {best_p:.4f} {sig}")
    print(f"  R² = {best_r**2:.3f} ({best_r**2*100:.1f}% de variance expliquée)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    config.ensure_dirs()

    logger.info("=== Azilal Temp Lag — Démarrage ===")

    lat, lon = get_azilal_coords()
    climate = extract_azilal_climate(lat, lon)
    cases = extract_azilal_cases()
    df = build_lagged_dataset(cases, climate)

    # Sauvegarder
    out_csv = config.OUTPUTS / "processed" / "azilal_climate_cases_lags.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Résumé
    plot_summary_table(df)

    # Graphiques
    plot_cross_correlation(df)
    plot_all_lags_grid(df)
    plot_best_lag_dual(df)
    plot_event_studies(df)

    logger.info("=== Terminé ===")
    print(f"Figures dans : {config.FIGURES}")


if __name__ == "__main__":
    main()
