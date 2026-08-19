"""
bioclimatic_zoning.py
======================
Zonage bioclimatique/epidemiologique des provinces marocaines : classification
par K-means (k choisi par silhouette) sur le climat + l'environnement REELS
(ERA5), couvrant les 76 provinces (le climat est disponible partout, y
compris dans le Sud sans cas ni documentation vectorielle).

Le statut vecteur (P. sergenti) et le volume de cas LCT sont ajoutes ensuite
en SURCOUCHE (pas comme critere de clustering) pour l'interpretation et la
carte : c'est la reponse a "division administrative -> division
epidemiologique/ecologique basee sur le climat".

Migre et adapte depuis l'ancien script exploratoire
`main src/zonage_bioclimatique_final.py` (chemins ad hoc /home/claude/...
remplaces par config.py + les sorties du pipeline).

Entrees :
  outputs/processed/communes_climate.csv    (extract_climate.py)
  outputs/processed/climate_morocco.db      (extract_climate.py, table daily -> humidite)
  outputs/processed/environment_morocco.db  (build_environment.py, altitude)
  data/raw/communes_maroc_final.csv
  outputs/processed/lct_clean.csv (ou data/raw/leish_LCT.csv si absent)
  data/raw/phlebotomus_sergenti_par_province.csv

Sortie :
  outputs/processed/zone_bioclim_province.csv

Usage :
  python src/models/bioclimatic_zoning.py

Note : les jointures province se font ici sur le libelle brut (comme dans le
script d'origine). La Phase 2 de la refonte introduit un libelle de
region/province canonique normalise dans config.py — ce script devra l'adopter
a ce moment-la pour eviter les pertes de lignes par variantes orthographiques.
"""

import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

K_RANGE = range(2, 9)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.\n{hint}")


def load_climate_profile() -> pd.DataFrame:
    """Profil bioclimatique annuel par province : temperature, amplitude, precipitations, humidite."""
    _require(config.COMMUNES_CLIMATE, "Lance d'abord : python src/data_prep/extract_climate.py")
    clim = pd.read_csv(config.COMMUNES_CLIMATE)

    prov_clim = (
        clim.groupby("province")
        .agg(
            temp_moy_an=("temp_mean", "mean"),
            temp_max=("temp_mean", "max"),
            temp_min=("temp_mean", "min"),
            precip_totale_an=("precip_annual", "mean"),
        )
        .reset_index()
    )
    prov_clim["amplitude_thermique"] = prov_clim["temp_max"] - prov_clim["temp_min"]

    if config.CLIMATE_DB.exists():
        conn = sqlite3.connect(config.CLIMATE_DB)
        hum = pd.read_sql(
            "SELECT province, AVG(humidity) AS humidite_moy_an FROM climate GROUP BY province",
            conn,
        )
        conn.close()
        prov_clim = prov_clim.merge(hum, on="province", how="left")
    else:
        print("[WARN] climate_morocco.db introuvable -> humidite_moy_an = NaN "
              "(lance extract_climate.py)")
        prov_clim["humidite_moy_an"] = np.nan

    return prov_clim


def load_environment_profile() -> pd.DataFrame:
    _require(config.ENV_DB, "Lance d'abord : python src/data_prep/build_environment.py")
    conn = sqlite3.connect(config.ENV_DB)
    env = pd.read_sql(
        "SELECT province, AVG(altitude_m) AS elevation_m FROM environment GROUP BY province",
        conn,
    )
    conn.close()
    return env


def classify_vector(status) -> str:
    if pd.isna(status):
        return "non_documente"
    s = str(status).lower()
    if s.startswith("oui"):
        return "confirme"
    if "indirecte" in s or "ancienne" in s:
        return "indice_indirect"
    return "non_documente"


def build_zone_profile() -> pd.DataFrame:
    profile = load_climate_profile().merge(load_environment_profile(), on="province", how="left")

    feat_cols = ["temp_moy_an", "amplitude_thermique", "precip_totale_an", "humidite_moy_an", "elevation_m"]
    n_missing = int(profile[feat_cols].isna().any(axis=1).sum())
    if n_missing:
        print(f"[WARN] {n_missing}/{len(profile)} provinces avec une valeur manquante -> "
              f"imputees par la mediane de la colonne")
        profile[feat_cols] = profile[feat_cols].fillna(profile[feat_cols].median())

    X = StandardScaler().fit_transform(profile[feat_cols])

    # ---- k-means, selection de k par silhouette (sur toutes les provinces) ----
    best_k, best_score, scores = None, -1.0, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
        score = silhouette_score(X, km.labels_)
        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score

    print("Silhouette par k :", {k: round(v, 3) for k, v in scores.items()})
    print(f"-> k retenu = {best_k} (silhouette={best_score:.3f})")

    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
    profile["zone_bioclim"] = km_final.labels_

    # ---- surcouche : volume de cas + statut vecteur (pas dans le clustering) ----
    lct_path = config.PROCESSED / "lct_clean.csv"
    if not lct_path.exists():
        lct_path = config.LCT_CSV
    cases = pd.read_csv(lct_path)
    case_counts = cases.groupby("Province").size().rename("total_cas").reset_index()
    profile = (
        profile.merge(case_counts, left_on="province", right_on="Province", how="left")
        .drop(columns=["Province"])
    )
    profile["total_cas"] = profile["total_cas"].fillna(0).astype(int)

    vec = pd.read_csv(config.SERGENTI_CSV)
    vec.columns = [c.strip() for c in vec.columns]
    vec["statut_vecteur"] = vec["Statut_P_sergenti"].apply(classify_vector)
    profile = (
        profile.merge(vec[["Province", "statut_vecteur"]], left_on="province", right_on="Province", how="left")
        .drop(columns=["Province"])
    )
    profile["statut_vecteur"] = profile["statut_vecteur"].fillna("non_documente")

    # ---- centroide geographique par province (moyenne des communes), pour la carte ----
    communes = pd.read_csv(config.COMMUNES_CSV)
    centroids = communes.groupby("province")[["latitude", "longitude"]].mean().reset_index()
    profile = profile.merge(centroids, on="province", how="left")

    return profile


def main() -> None:
    config.ensure_dirs()
    profile = build_zone_profile()

    out_path = config.PROCESSED / "zone_bioclim_province.csv"
    profile.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")

    n_zones = profile["zone_bioclim"].nunique()
    print(f"\n{len(profile)} provinces classees en {n_zones} zones bioclimatiques\n")
    for z, g in profile.groupby("zone_bioclim"):
        print(
            f"Zone {z} (n={len(g)}): temp={g['temp_moy_an'].mean():.1f}C  "
            f"precip={g['precip_totale_an'].mean():.0f}mm/an  humid={g['humidite_moy_an'].mean():.0f}%  "
            f"elev={g['elevation_m'].mean():.0f}m  |  cas cumules={g['total_cas'].sum()}  |  "
            f"vecteur confirme dans {(g['statut_vecteur'] == 'confirme').sum()}/{len(g)} provinces"
        )


if __name__ == "__main__":
    main()
