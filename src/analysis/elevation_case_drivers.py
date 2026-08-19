"""
elevation_case_drivers.py
==========================
Pour chacune des 4 classes d'altitude (elevation_classification.py), quels
parametres (climat, environnement, presence du vecteur, population) sont
associes au nombre de cas LCT par commune -- calcule SEPAREMENT dans chaque
classe (pas sur l'ensemble du pays), pour voir si les facteurs qui comptent
different selon qu'on est en plaine ou en haute montagne.

Methode : correlation de Spearman (robuste aux nombreux zeros du comptage de
cas -- 90%+ de communes-mois a 0 cas -- et ne suppose pas de relation
lineaire ni de normalite, contrairement a un GLM/Pearson). Un ZINB/NB GLM a
deja ete tente sur ces donnees (model_benchmark.py) et diverge numeriquement
-- Spearman est le choix robuste ici, pas un raccourci.

Deux cibles par commune :
  - total_cas       : somme des cas 2009-2021 (compte brut)
  - incidence_100k  : total_cas / pop_total * 100000 (normalise par population,
                       pour ne pas juste retrouver "plus de gens = plus de cas")

Entree :
  outputs/processed/commune_panel.csv   (build_commune_panel.py)
  outputs/processed/province_elevation_classes.csv (elevation_classification.py)

Sortie :
  outputs/processed/elevation_case_drivers.csv

Usage :
  python src/analysis/elevation_case_drivers.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

STATIC_FEATURES = ["elevation_m", "lai", "aridity_index", "psi_mean", "psi_sd", "latitude"]
DYNAMIC_FEATURES = ["temp_moy", "precip_mm", "humidite_pct"]
FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES


def build_commune_level() -> pd.DataFrame:
    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv")

    agg = {"n_cas": "sum", "pop_total": "first", "province": "first"}
    agg.update({c: "first" for c in STATIC_FEATURES})
    agg.update({c: "mean" for c in DYNAMIC_FEATURES})

    commune = panel.groupby("commune_id").agg(agg).reset_index()
    commune = commune.rename(columns={"n_cas": "total_cas"})
    commune["incidence_100k"] = np.where(
        commune["pop_total"] > 0,
        commune["total_cas"] / commune["pop_total"] * 100_000,
        np.nan,
    )
    return commune


def load_elevation_classes() -> pd.DataFrame:
    path = config.PROCESSED / "province_elevation_classes.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.\nLance d'abord : python src/analysis/elevation_classification.py")
    return pd.read_csv(path)[["province", "classe_altitude", "classe_altitude_rank"]]


def correlate_within_class(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        sub = df[[feat, target]].dropna()
        if len(sub) < 8 or sub[feat].nunique() < 3:
            continue
        rho, pval = spearmanr(sub[feat], sub[target])
        rows.append({"feature": feat, "n": len(sub), "spearman_rho": rho, "p_value": pval})
    out = pd.DataFrame(rows)
    if len(out):
        out["significatif_5pct"] = out["p_value"] < 0.05
        out = out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index)
    return out


def main() -> None:
    config.ensure_dirs()
    commune = build_commune_level()
    classes = load_elevation_classes()
    commune = commune.merge(classes, on="province", how="left")

    n_missing = commune["classe_altitude"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} communes sans classe d'altitude (province absente de la classification) -> exclues")
        commune = commune.dropna(subset=["classe_altitude"])

    all_results = []
    print(f"\n{'='*90}")
    for rank, g in commune.groupby("classe_altitude_rank"):
        label = g["classe_altitude"].iloc[0]
        n_communes = len(g)
        n_with_cases = (g["total_cas"] > 0).sum()
        print(f"\n[{rank}] {label}  (n={n_communes} communes, {n_with_cases} avec >=1 cas, "
              f"{g['total_cas'].sum():.0f} cas cumules)")

        for target in ["total_cas", "incidence_100k"]:
            res = correlate_within_class(g, target)
            if not len(res):
                continue
            res["classe_altitude"] = label
            res["classe_altitude_rank"] = rank
            res["target"] = target
            all_results.append(res)

            print(f"  -- correlations vs {target} --")
            for _, r in res.head(6).iterrows():
                star = "*" if r["significatif_5pct"] else " "
                print(f"    {r['feature']:<16} rho={r['spearman_rho']:+.3f}{star}  (p={r['p_value']:.4f}, n={r['n']:.0f})")

    print(f"\n{'='*90}")

    result = pd.concat(all_results, ignore_index=True)
    out_cols = ["classe_altitude_rank", "classe_altitude", "target", "feature",
                "spearman_rho", "p_value", "significatif_5pct", "n"]
    result = result[out_cols].sort_values(["classe_altitude_rank", "target"], key=lambda s: s if s.name != "target" else s)
    out_path = config.PROCESSED / "elevation_case_drivers.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
