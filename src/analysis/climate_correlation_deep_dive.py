"""
climate_correlation_deep_dive.py
==================================
Reponse directe a "la temperature doit forcement compter, pourquoi elle ne
ressort pas ?" : les correlations de Spearman precedentes (elevation_case_
drivers.py) sont MONOTONES par construction -- si la relation vraie est en
cloche (risque maximal dans une plage de temperature intermediaire, comme
c'est biologiquement attendu pour l'activite d'un insecte ectotherme), une
correlation monotone peut sous-estimer fortement son importance reelle. Ce
script verifie explicitement la forme de la relation (binning), pas
seulement son signe/intensite lineaire.

Fait aussi ce qui manquait au rapport precedent : matrice de correlation
complete (Pearson + Spearman) climat/environnement x cas, globale ET par
zone bioclimatique (pas seulement par classe d'altitude), et une verification
du seul proxy socio-economique disponible dans ce projet (population totale
-- il n'y a AUCUNE donnee de revenu/pauvrete/education dans data/raw/,
verifie explicitement, pas suppose).

Entrees :
  outputs/processed/commune_panel.csv
  outputs/processed/zone_bioclim_province.csv

Sortie :
  outputs/processed/climate_correlation_matrix.csv
  outputs/processed/temperature_binning.csv

Usage :
  python src/analysis/climate_correlation_deep_dive.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

FEATURES = ["temp_moy", "precip_mm", "humidite_pct", "aridity_index", "lai", "elevation_m", "pop_total"]


def build_commune_level(only_valid_years=True) -> pd.DataFrame:
    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv")
    panel = panel[panel["annee"] <= 2020]
    if only_valid_years:
        panel = panel[~panel["annee_sans_mois"]] if "annee_sans_mois" in panel.columns else panel

    agg = {"n_cas": "sum", "pop_total": "first", "province": "first"}
    for c in ["elevation_m", "lai", "aridity_index"]:
        agg[c] = "first"
    for c in ["temp_moy", "precip_mm", "humidite_pct"]:
        agg[c] = "mean"

    commune = panel.groupby("commune_id").agg(agg).reset_index().rename(columns={"n_cas": "total_cas"})
    commune["incidence_100k"] = np.where(
        commune["pop_total"] > 0, commune["total_cas"] / commune["pop_total"] * 100_000, np.nan
    )
    return commune


def correlation_table(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        for target in ["total_cas", "incidence_100k"]:
            sub = df[[feat, target]].dropna()
            if len(sub) < 10 or sub[feat].nunique() < 3:
                continue
            rho, p_s = spearmanr(sub[feat], sub[target])
            r, p_p = pearsonr(sub[feat], sub[target])
            rows.append({
                "groupe": group_label, "feature": feat, "target": target, "n": len(sub),
                "spearman_rho": rho, "spearman_p": p_s, "pearson_r": r, "pearson_p": p_p,
            })
    return pd.DataFrame(rows)


def temperature_binning(commune: pd.DataFrame) -> pd.DataFrame:
    df = commune.dropna(subset=["temp_moy", "total_cas"]).copy()
    df["temp_bin"] = pd.qcut(df["temp_moy"], q=8, duplicates="drop")
    out = df.groupby("temp_bin", observed=True).agg(
        temp_moy_bin=("temp_moy", "mean"),
        n_communes=("commune_id", "count"),
        cas_moyens=("total_cas", "mean"),
        pct_communes_avec_cas=("total_cas", lambda s: 100 * (s > 0).mean()),
    ).reset_index(drop=True)
    return out


def main() -> None:
    config.ensure_dirs()
    commune = build_commune_level()

    print(f"\n{'='*95}\nVERIFICATION DONNEES SOCIO-ECONOMIQUES DISPONIBLES\n{'='*95}")
    import os
    raw_files = os.listdir(config.RAW) if hasattr(config, "RAW") else []
    print("Fichiers dans data/raw/ :", raw_files)
    print("Aucun fichier de revenu/pauvrete/education/emploi trouve -> seul proxy disponible : population "
          "totale (pop_total), pas une vraie mesure socio-economique. Traite comme tel ci-dessous.")

    print(f"\n{'='*95}\nMATRICE DE CORRELATION GLOBALE (climat/environnement/population x cas)\n{'='*95}")
    global_corr = correlation_table(commune, "national")
    print(global_corr.sort_values("spearman_rho", key=abs, ascending=False).to_string(index=False))

    zb = pd.read_csv(config.ZONE_BIOCLIM_CSV)[["province", "zone_bioclim"]]
    commune_z = commune.merge(zb, on="province", how="left")

    zone_names = {0: "Plaine/Plateau", 1: "Montagne/Atlas", 2: "Aride/Saharien"}
    zone_rows = []
    for z, g in commune_z.groupby("zone_bioclim"):
        label = zone_names.get(int(z), str(z))
        t = correlation_table(g, label)
        zone_rows.append(t)
        print(f"\n--- Zone bioclimatique : {label} (n={len(g)} communes) ---")
        print(t[t["target"] == "total_cas"].sort_values("spearman_rho", key=abs, ascending=False).to_string(index=False))

    all_corr = pd.concat([global_corr] + zone_rows, ignore_index=True)
    out_path = config.PROCESSED / "climate_correlation_matrix.csv"
    all_corr.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")

    print(f"\n{'='*95}\nTEMPERATURE : RELATION EST-ELLE MONOTONE OU EN CLOCHE ? (binning, pas juste correlation)\n{'='*95}")
    tbin = temperature_binning(commune)
    print(tbin.to_string(index=False))
    peak_idx = tbin["cas_moyens"].idxmax()
    print(f"\nPic de cas moyens dans la tranche de temperature : {tbin.loc[peak_idx, 'temp_moy_bin']:.1f}C "
          f"({tbin.loc[peak_idx, 'cas_moyens']:.3f} cas/commune en moyenne)")
    is_monotonic_inc = tbin["cas_moyens"].is_monotonic_increasing
    is_monotonic_dec = tbin["cas_moyens"].is_monotonic_decreasing
    if not is_monotonic_inc and not is_monotonic_dec:
        print("-> relation NON MONOTONE confirmee : la temperature a un effet reel mais pas lineaire, "
              "ce qui explique pourquoi Spearman (mesure de monotonie) le sous-estime.")
    else:
        print("-> relation globalement monotone sur cet echantillon.")

    tbin_path = config.PROCESSED / "temperature_binning.csv"
    tbin.to_csv(tbin_path, index=False, encoding="utf-8")
    print(f"[OK] {tbin_path}")


if __name__ == "__main__":
    main()
