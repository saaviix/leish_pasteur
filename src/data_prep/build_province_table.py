"""
build_province_table.py
=======================
Construit la table par province qui alimente le modele bayesien :
  - centroides (lat/lon moyens) + region par province
  - signal epidemiologique  y_epi   (>=1 cas LCT autochtone)
  - signal entomologique     y_ento_hard / y_ento_soft (captures P. sergenti)
  - covariables climatiques   (temp_mean, precip_annual, aridite) agregees par province
  - graphe de voisinage spatial (Delaunay + haversine, pruning >300 km,
    reconnexion des noeuds isoles) pour le lissage spatial ICAR
  - covariables standardisees (lat_z, temp_z, precip_z, arid_z)

Entrees :
  data/raw/communes_maroc_final.csv
  data/raw/leish_LCT.csv
  data/raw/phlebotomus_sergenti_par_province.csv
  outputs/processed/communes_climate.csv   (optionnel : si extract_climate.py a tourne)

Sorties :
  outputs/processed/province_table.csv
  outputs/processed/adjacency_edges.npy

Usage :
  python src/data_prep/build_province_table.py
"""

from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
import networkx as nx

import config

MAX_EDGE_KM = 300

# alias : reconciliation partagee par tout le pipeline (voir config.norm_key)
norm = config.norm_key


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def main() -> None:
    config.ensure_dirs()

    communes = pd.read_csv(config.COMMUNES_CSV)
    # lct_clean.csv (dedupliqu par clean_lct.py) plutot que le brut LCT_CSV --
    # trouve par audit (workflow) : ce script lisait encore data/raw/leish_LCT.csv
    # brut, donc ~1003/1013 doublons nationaux (quasi tous classes "Autochtone")
    # etaient recomptes dans lct_cases, jusqu'a +9.3% localement (Chichaoua).
    lct_path = config.PROCESSED / "lct_clean.csv"
    if not lct_path.exists():
        lct_path = config.LCT_CSV
    lct = pd.read_csv(lct_path)
    sergenti = pd.read_csv(config.SERGENTI_CSV)


    communes["prov_key"] = communes["province"].map(norm)

    # ---------- centroides + region ----------
    prov = (
        communes.groupby("prov_key", as_index=False)
        .agg(
            province=("province", "first"),
            region=("region", "first"),
            n_communes=("commune", "count"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
    )

    # ---------- signal epidemiologique LCT ----------
    lct["prov_key"] = lct["Province"].map(norm)
    # Classification=NaN (pas juste != "Autochtone") exclue jusqu'ici, ce qui
    # sous-comptait silencieusement des provinces entieres : Chichaoua 2020
    # (533/534 lignes sans Classification, trouve par audit) tombait a 0 cas
    # "Autochtone" alors que 529 cas reels existent cette annee-la (le trou
    # touche 92.6% de son total 2020). Classification==NaN est presumee
    # Autochtone (base rate 96% sur les lignes renseignees) plutot qu'exclue ;
    # seules les classifications explicites "Importee"/"Indeterminee" restent
    # exclues. Alerte si l'imputation depasse 50 cas pour une province.
    n_imputed_by_prov = lct.loc[lct["Classification"].isna()].groupby("prov_key").size()
    if (n_imputed_by_prov > 50).any():
        flagged = n_imputed_by_prov[n_imputed_by_prov > 50].to_dict()
        print(f"[WARN] Classification manquante (imputee Autochtone) pour >50 cas dans : {flagged}")
    lct_autoch = lct[lct["Classification"].isin(["Autochtone"]) | lct["Classification"].isna()]
    lct_agg = lct_autoch.groupby("prov_key").size().rename("lct_cases").reset_index()
    prov = prov.merge(lct_agg, on="prov_key", how="left")
    prov["lct_cases"] = prov["lct_cases"].fillna(0).astype(int)
    prov["y_epi"] = (prov["lct_cases"] > 0).astype(int)
    prov["lct_rate"] = prov["lct_cases"] / prov["n_communes"]

    # ---------- signal entomologique P. sergenti ----------
    sergenti["prov_key"] = sergenti["Province"].map(norm)
    statut = sergenti["Statut_P_sergenti"].astype(str).str.lower()
    is_hard = statut.str.startswith("oui")
    is_soft = statut.str.contains("mention|verifier|incertain", regex=True) & ~is_hard
    hard_ento = set(sergenti.loc[is_hard, "prov_key"])
    soft_ento = set(sergenti.loc[is_soft, "prov_key"]) - hard_ento
    prov["y_ento_hard"] = prov["prov_key"].isin(hard_ento).astype(int)
    prov["y_ento_soft"] = prov["prov_key"].isin(soft_ento).astype(int)

    print("Provinces capture confirmee (hard):",
          sorted(prov.loc[prov.y_ento_hard == 1, "province"]))
    print("Provinces signal a verifier (soft):",
          sorted(prov.loc[prov.y_ento_soft == 1, "province"]))
    print("Provinces avec cas LCT autochtones:", int(prov["y_epi"].sum()), "/", len(prov))
    print("Provinces SANS aucun signal (gap):",
          int(((prov.y_epi == 0) & (prov.y_ento_hard == 0) & (prov.y_ento_soft == 0)).sum()))

    # ---------- covariables climatiques (si disponibles) ----------
    if config.COMMUNES_CLIMATE.exists():
        clim = pd.read_csv(config.COMMUNES_CLIMATE)
        # communes_climate.csv contient une ligne par commune (moyenne long-terme)
        # agregee par province en prenant la moyenne des centroides
        clim["prov_key"] = clim["province"].map(norm)
        clim_agg = (
            clim.groupby("prov_key")
            .agg(
                temp_mean=("temp_mean", "mean"),
                precip_annual=("precip_annual", "mean"),
            )
            .reset_index()
        )
        prov = prov.merge(clim_agg, on="prov_key", how="left")
        prov["arid_index"] = prov["precip_annual"] / (prov["temp_mean"] + 33)
        prov["temp_z"] = zscore(prov["temp_mean"])
        prov["precip_z"] = zscore(prov["precip_annual"])
        prov["arid_z"] = zscore(prov["arid_index"])
        print("[INFO] covariables climatiques fusionnees")
    else:
        print("[INFO] pas de communes_climate.csv -> covariables climat = 0 "
              "(le modele utilisera seulement la latitude)")
        for c in ["temp_z", "precip_z", "arid_z"]:
            prov[c] = 0.0

    # ---------- voisinage spatial (Delaunay) ----------
    pts = prov[["lon", "lat"]].values
    tri = Delaunay(pts)
    edges_raw = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges_raw.add((min(a, b), max(a, b)))

    edges = []
    for a, b in edges_raw:
        d = haversine(prov.lat[a], prov.lon[a], prov.lat[b], prov.lon[b])
        if d <= MAX_EDGE_KM:
            edges.append((a, b))
    edges = np.array(sorted(edges))
    print(f"Pruning : {len(edges_raw) - len(edges)} aretes > {MAX_EDGE_KM} km supprimees "
          f"sur {len(edges_raw)} ; {len(edges)} restantes")

    G = nx.Graph()
    G.add_nodes_from(range(len(prov)))
    G.add_edges_from([tuple(e) for e in edges])
    isolated = [i for i in G.nodes if G.degree[i] == 0]
    if isolated:
        tree = cKDTree(pts)
        extra = []
        for i in isolated:
            _, idxs = tree.query(pts[i], k=2)
            nearest = idxs[1]
            extra.append((min(i, nearest), max(i, nearest)))
            print(f"  reconnexion noeud isole {prov.province[i]} -> {prov.province[nearest]}")
        edges = np.array(sorted(set(map(tuple, edges.tolist())) | set(extra)))
        G.add_edges_from(extra)

    print(f"Graphe adjacence : {len(edges)} aretes, "
          f"{nx.number_connected_components(G)} composante(s) connexe(s)")

    # ---------- covariable proxy latitude ----------
    prov["lat_z"] = zscore(prov["lat"])

    prov.to_csv(config.PROVINCE_TABLE, index=False, encoding="utf-8")
    np.save(config.ADJ_EDGES, edges)
    print(f"\nEcrit : {config.PROVINCE_TABLE}")
    print(f"Ecrit : {config.ADJ_EDGES}")


if __name__ == "__main__":
    main()
