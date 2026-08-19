"""
elevation_classification.py
============================
Classification des provinces marocaines par ALTITUDE SEULE (contrairement a
`bioclimatic_zoning.py` qui clusterise sur climat+altitude combines) :
K-means 1D sur `elevation_m`, k choisi par silhouette, puis etiquetage des
clusters par ordre croissant d'altitude (plaine -> montagne).

Entree :
  outputs/processed/zone_bioclim_province.csv (bioclimatic_zoning.py)

Sortie :
  outputs/processed/province_elevation_classes.csv

Usage :
  python src/analysis/elevation_classification.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

K_RANGE = range(2, 7)

# etiquettes par rang croissant d'altitude, utilisees selon le k retenu
LABELS_BY_K = {
    2: ["Plaine/Cotier", "Montagne"],
    3: ["Plaine/Cotier", "Plateau/Piemont", "Montagne"],
    4: ["Plaine/Cotier", "Plateau", "Moyenne montagne", "Haute montagne"],
    5: ["Plaine/Cotier", "Plateau bas", "Plateau/Piemont", "Moyenne montagne", "Haute montagne"],
    6: ["Plaine/Cotier", "Plateau bas", "Plateau/Piemont", "Moyenne montagne",
        "Haute montagne", "Tres haute montagne"],
}


def build_elevation_classes() -> pd.DataFrame:
    src = config.PROCESSED / "zone_bioclim_province.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} introuvable.\nLance d'abord : python src/models/bioclimatic_zoning.py")

    df = pd.read_csv(src)
    df = df.dropna(subset=["elevation_m"]).copy()

    X = df[["elevation_m"]].values

    best_k, best_score, scores = None, -1.0, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
        score = silhouette_score(X, km.labels_)
        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score

    print("Silhouette par k (altitude seule) :", {k: round(v, 3) for k, v in scores.items()})
    print(f"-> k retenu = {best_k} (silhouette={best_score:.3f})")

    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
    df["cluster_raw"] = km_final.labels_

    # ordonner les clusters par altitude moyenne croissante -> etiquette lisible
    order = (
        df.groupby("cluster_raw")["elevation_m"].mean().sort_values().index.tolist()
    )
    rank_map = {cluster_id: rank for rank, cluster_id in enumerate(order)}
    df["classe_altitude_rank"] = df["cluster_raw"].map(rank_map)

    labels = LABELS_BY_K.get(best_k, [f"Classe {i}" for i in range(best_k)])
    df["classe_altitude"] = df["classe_altitude_rank"].map(dict(enumerate(labels)))

    out_cols = [
        "province", "classe_altitude", "classe_altitude_rank", "elevation_m",
        "temp_moy_an", "precip_totale_an", "total_cas", "statut_vecteur",
        "zone_bioclim",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    result = df[out_cols].sort_values(["classe_altitude_rank", "elevation_m"])
    return result


def main() -> None:
    config.ensure_dirs()
    result = build_elevation_classes()

    out_path = config.PROCESSED / "province_elevation_classes.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")

    print(f"\n{len(result)} provinces classees par altitude en {result['classe_altitude'].nunique()} classes\n")
    for rank, g in result.groupby("classe_altitude_rank"):
        label = g["classe_altitude"].iloc[0]
        print(
            f"[{rank}] {label} (n={len(g)}): altitude {g['elevation_m'].min():.0f}-{g['elevation_m'].max():.0f}m "
            f"(moy {g['elevation_m'].mean():.0f}m)  |  cas cumules={g['total_cas'].sum()}  |  "
            f"vecteur confirme dans {(g['statut_vecteur'] == 'confirme').sum()}/{len(g)}"
        )
        exemples = ", ".join(g.sort_values('total_cas', ascending=False)['province'].head(5))
        print(f"      exemples : {exemples}")


if __name__ == "__main__":
    main()
