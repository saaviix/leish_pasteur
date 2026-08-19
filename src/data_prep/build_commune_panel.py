"""
build_commune_panel.py
======================
Construit le panel spatio-temporel complet : Commune x Annee x Mois.

Reecrit en Phase 3 de la refonte -- la version precedente avait 3 bugs
serieux, tous corriges ici :
  1. `load_communes()` faisait `drop_duplicates(subset=["commune"])`, ce qui
     supprimait silencieusement les communes homonymes de provinces
     differentes (ex. "Ait Ouallal" existe a la fois a Zagora et a Meknes).
     -> on utilise desormais `commune_id` (l'id du referentiel) comme cle,
     jamais le nom seul.
  2. Le choix de la colonne commune pour les cas LCT prenait la premiere
     colonne "commune*" trouvee dans lct_clean.csv, qui est "Commune" (le
     texte BRUT, non reconcilie) et non "Commune_std"/"commune_id" (la
     reconciliation de la Phase 2, cf. clean_lct.py) -- la quasi-totalite des
     cas ne se joignaient donc jamais correctement au panel.
     -> on joint desormais explicitement sur `commune_id`.
  3. L'extraction climatique reimplementait sa propre extraction ERA5 (boucle
     xarray par commune par fichier), dupliquant extract_climate.py en moins
     robuste et surtout SANS jamais utiliser son resultat deja calcule.
     -> on lit directement climate_morocco.db (source unique, deja verifiee).

Ajoute aussi : l'environnement (altitude, LAI, aridite) et la population
(pour un futur taux d'incidence), absents du panel precedent.

Entrees :
  data/raw/communes_maroc_final.csv
  outputs/processed/lct_clean.csv           (clean_lct.py, Phase 2)
  outputs/processed/climate_morocco.db      (extract_climate.py)
  outputs/processed/environment_morocco.db  (build_environment.py)
  outputs/processed/province_table.csv      (build_province_table.py)
  data/raw/mun_pop.csv                      (fix_mun_pop.py, Phase 2)

Sortie :
  outputs/processed/commune_panel.csv (1503 communes x 12 mois x n annees ERA5)

Usage :
  python src/data_prep/build_commune_panel.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_communes() -> pd.DataFrame:
    df = pd.read_csv(config.COMMUNES_CSV)
    df = df.rename(columns={"id": "commune_id"})
    return df[["commune_id", "commune", "province", "region", "latitude", "longitude"]]


def load_climate() -> pd.DataFrame:
    if not config.CLIMATE_DB.exists():
        logger.error(f"{config.CLIMATE_DB} introuvable. Lance : python src/data_prep/extract_climate.py")
        return None
    conn = sqlite3.connect(config.CLIMATE_DB)
    df = pd.read_sql(
        "SELECT commune_id, annee, mois, temp_mean AS temp_moy, "
        "precipitation AS precip_mm, humidity AS humidite_pct FROM climate",
        conn,
    )
    conn.close()
    return df


def load_environment() -> pd.DataFrame:
    if not config.ENV_DB.exists():
        logger.warning(f"{config.ENV_DB} introuvable -> pas de covariables environnement "
                        f"(lance build_environment.py)")
        return None
    conn = sqlite3.connect(config.ENV_DB)
    df = pd.read_sql(
        "SELECT commune_id, altitude_m AS elevation_m, vegetation_lai AS lai, "
        "aridity_index FROM environment",
        conn,
    )
    conn.close()
    return df


def load_population() -> pd.DataFrame:
    if not config.MUN_POP_CSV.exists():
        logger.warning(f"{config.MUN_POP_CSV} introuvable -> pas de taux d'incidence "
                        f"(lance fix_mun_pop.py)")
        return None
    df = pd.read_csv(config.MUN_POP_CSV)[["commune_id", "pop_total"]].dropna(subset=["commune_id"])
    # quelques noms de commune ambigus (homonymes) font pointer plusieurs
    # lignes population vers le meme commune_id (cf. fix_mun_pop.py) -- les
    # garder ferait exploser le nombre de lignes du panel par jointure
    # many-to-many. On ne devine pas laquelle est correcte : on les exclut,
    # incidence_100k restera NaN pour ces quelques communes plutot que fausse.
    dup = df["commune_id"].duplicated(keep=False)
    if dup.any():
        n_ids = df.loc[dup, "commune_id"].nunique()
        logger.warning(f"{n_ids} commune_id avec plusieurs lignes population ambigues -> exclues "
                        f"(incidence_100k = NaN pour ces communes)")
        df = df[~dup]
    return df


def load_occupancy_posterior() -> pd.DataFrame:
    """psi_mean (probabilite de presence de P. sergenti) issue du modele
    bayesien ICAR -- ajoute comme covariable : c'est un signal reel
    (entomologique + epidemiologique + spatial), pas juste une feature de
    plus, et relie enfin les deux modeles du projet l'un a l'autre."""
    if not config.POSTERIOR_CSV.exists():
        logger.warning(f"{config.POSTERIOR_CSV} introuvable -> pas de covariable psi_mean "
                        f"(lance bayesian_occupancy.py)")
        return None
    df = pd.read_csv(config.POSTERIOR_CSV)
    return df[["province", "psi_mean", "psi_sd"]]


def find_years_without_month(df: pd.DataFrame) -> set:
    """Detecte dynamiquement les annees ou AUCUN cas (parmi ceux avec
    commune reconciliee) n'a de Mois_Diagnostic renseigne -- typiquement un
    format de rapport annuel different a la source, pas une vraie absence de
    cas. Trouve empiriquement en creusant pourquoi le signal temperature/
    saisonnalite etait faible : 2016/2018/2019 ont 0% de mois renseigne alors
    qu'ils contiennent des milliers de cas (5623 au total) -- ces annees
    etaient jusqu'ici silencieusement remplies a n_cas=0 (voir build_panel),
    ce qui injectait 3 annees de faux "zero cas" dans l'entrainement mensuel
    des modeles (GBM/PINN), corrompant tout signal saisonnier appris sur ces
    annees. Detection dynamique (pas une liste codee en dur) pour rester
    correct si le fichier source LCT est mis a jour plus tard."""
    matched = df[df["commune_matched"]].copy()
    matched["annee_num"] = pd.to_numeric(matched["Annee_Source"], errors="coerce")
    matched["mois_num"] = pd.to_numeric(matched["Mois_Diagnostic"], errors="coerce")
    coverage = matched.groupby("annee_num")["mois_num"].apply(lambda s: s.notna().mean())
    bad_years = set(coverage[coverage == 0.0].index.astype(int))
    if bad_years:
        n_cas_affectes = int((matched["annee_num"].isin(bad_years)).sum())
        logger.warning(
            f"Annees SANS AUCUN mois de diagnostic renseigne : {sorted(bad_years)} "
            f"({n_cas_affectes} cas avec commune reconciliee mais illocalisables dans le mois) -- "
            f"marquees n_cas=NaN dans le panel (pas 0) pour ces annees, colonne "
            f"'annee_sans_mois'=True. A EXCLURE du train/test des modeles mensuels."
        )
    return bad_years


def load_lct_cases() -> tuple:
    """Agrege les cas LCT par (commune_id, annee, mois), en utilisant la
    reconciliation commune de la Phase 2 (clean_lct.py) -- pas le texte brut.
    Retourne (cases, annees_sans_mois)."""
    lct_path = config.PROCESSED / "lct_clean.csv"
    if not lct_path.exists():
        raise FileNotFoundError(f"{lct_path} introuvable. Lance d'abord : python src/data_prep/clean_lct.py")
    df = pd.read_csv(lct_path)

    n_total = len(df)
    n_commune_matched = int(df["commune_matched"].sum())
    df["annee_num"] = pd.to_numeric(df["Annee_Source"], errors="coerce")
    df["mois_num"] = pd.to_numeric(df["Mois_Diagnostic"], errors="coerce")

    annees_sans_mois = find_years_without_month(df)

    placeable = df[
        df["commune_matched"]
        & df["annee_num"].between(2009, 2020)
        & df["mois_num"].between(1, 12)
    ].copy()

    logger.info(
        f"Cas LCT : {n_total} au total -> {n_commune_matched} avec commune reconciliee "
        f"({100 * n_commune_matched / n_total:.1f}%) -> {len(placeable)} placables dans le panel "
        f"commune x annee x mois ({100 * len(placeable) / n_total:.1f}% du total ; le reste manque "
        f"soit de commune reconciliee, soit de mois de diagnostic -- cas non fabriques, exclus)"
    )

    cases = (
        placeable.groupby(["commune_id", "annee_num", "mois_num"])
        .size()
        .rename("n_cas")
        .reset_index()
        .rename(columns={"annee_num": "annee", "mois_num": "mois"})
    )
    cases["annee"] = cases["annee"].astype(int)
    cases["mois"] = cases["mois"].astype(int)
    cases["commune_id"] = cases["commune_id"].astype(int)
    return cases, annees_sans_mois


def build_panel() -> None:
    config.ensure_dirs()

    communes = load_communes()
    logger.info(f"{len(communes)} communes (referentiel, id unique).")

    climate = load_climate()
    if climate is None:
        logger.error("Panel non construit : climat indisponible.")
        return

    years = sorted(climate["annee"].unique())
    months = list(range(1, 13))
    logger.info(f"Grille : {len(communes)} communes x {len(years)} annees ({years[0]}-{years[-1]}) x 12 mois")

    grid = communes.merge(pd.DataFrame({"annee": years}), how="cross").merge(
        pd.DataFrame({"mois": months}), how="cross"
    )

    panel = grid.merge(climate, on=["commune_id", "annee", "mois"], how="left")

    env = load_environment()
    if env is not None:
        panel = panel.merge(env, on="commune_id", how="left")

    pop = load_population()
    if pop is not None:
        panel = panel.merge(pop, on="commune_id", how="left")

    cases, annees_sans_mois = load_lct_cases()
    panel = panel.merge(cases, on=["commune_id", "annee", "mois"], how="left")
    panel["annee_sans_mois"] = panel["annee"].isin(annees_sans_mois)
    # n_cas=0 seulement pour les annees ou l'absence de cas places est reelle
    # (mois connu, aucun cas ce mois-la) ; pour les annees sans AUCUN mois
    # renseigne, n_cas reste NaN (inconnu, pas zero) -- voir load_lct_cases.
    fillable = (panel["annee"] <= 2020) & (~panel["annee_sans_mois"])
    panel.loc[fillable, "n_cas"] = panel.loc[fillable, "n_cas"].fillna(0)

    if pop is not None:
        panel["incidence_100k"] = np.where(
            panel["pop_total"] > 0, panel["n_cas"] / panel["pop_total"] * 100_000, np.nan
        )

    pt_path = config.PROVINCE_TABLE
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        pt_cols = [c for c in pt.columns if c not in ("province", "region", "lat", "lon")]
        panel = panel.merge(pt[["province"] + pt_cols], on="province", how="left")

    psi = load_occupancy_posterior()
    if psi is not None:
        panel = panel.merge(psi, on="province", how="left")

    logger.info("Calcul des lags climatiques (1 a 6 mois) par commune...")
    panel = panel.sort_values(["commune_id", "annee", "mois"])
    for lag in range(1, 7):
        for col in ["temp_moy", "precip_mm", "humidite_pct"]:
            panel[f"{col}_lag{lag}"] = panel.groupby("commune_id")[col].shift(lag)

    panel["sin_month"] = np.sin(2 * np.pi * panel["mois"] / 12.0)
    panel["cos_month"] = np.cos(2 * np.pi * panel["mois"] / 12.0)

    out_path = config.PROCESSED / "commune_panel.csv"
    panel.to_csv(out_path, index=False)
    n_cas_total = int(panel["n_cas"].sum())
    n_nan_rows = int(panel["n_cas"].isna().sum())
    logger.info(f"Panel ecrit : {out_path} ({len(panel)} lignes, {n_cas_total} cas places au total, "
                f"{n_nan_rows} lignes n_cas=NaN [annees sans mois de diagnostic, a exclure du train/test])")


if __name__ == "__main__":
    build_panel()
