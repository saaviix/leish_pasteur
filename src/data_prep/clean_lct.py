"""
clean_lct.py
============
Nettoyage et standardisation des donnees de cas de leishmaniose cutanee (LCT).

Objectifs :
  1. Charger data/raw/leish_LCT.csv (cas 2009-2020).
  2. Normaliser les libelles texte (region, province, commune) : trim, casse,
     harmonisation des accents/tirets (ex. "DrâaTafilalet" vs "Drâa-Tafilalet",
     "OURIKA" vs "Ourika").
  3. Reconcilier les region/province avec le referentiel unique
     communes_maroc_final.csv (source de verite geographique).
  4. Produire un RAPPORT de completude (donnees manquantes) : c'est la base de
     l'inference bayesienne pour les communes/provinces sans donnee.

Sorties :
  - outputs/processed/lct_clean.csv            (cas nettoyes)
  - outputs/processed/lct_missing_report.csv   (rapport de donnees manquantes)

Usage :
  python src/data_prep/clean_lct.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

import config

ROOT = config.ROOT
IN_LCT = config.LCT_CSV
IN_COMMUNES = config.COMMUNES_CSV
OUT_DIR = config.PROCESSED
OUT_CLEAN = OUT_DIR / "lct_clean.csv"
OUT_MISSING = OUT_DIR / "lct_missing_report.csv"


def norm_key(s):
    """Cle de rapprochement : sans accents, minuscule, espaces normalises."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s).strip()).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().split())
    return s


def load_and_standardize(lct_path: Path, communes_path: Path):
    lct = pd.read_csv(lct_path)
    communes = pd.read_csv(communes_path)

    # --- referentiel geographique (province -> region canonique) ---
    communes["prov_key"] = communes["province"].map(norm_key)
    ref = (
        communes.groupby("prov_key")
        .agg(province_ref=("province", "first"), region_ref=("region", "first"))
        .reset_index()
    )

    # --- standardisation des colonnes LCT ---
    lct["prov_key"] = lct["Province"].map(norm_key)
    # trim / casse title-case des libelles commune & localite
    for col in ["Commune", "Secteur", "Localite"]:
        if col in lct.columns:
            lct[col] = lct[col].astype(str).str.strip()
            # remet en Title Case seulement si tout en majuscules (ex "OURIKA")
            mask_upper = lct[col].str.isupper() & (lct[col].str.len() > 1)
            lct.loc[mask_upper, col] = lct.loc[mask_upper, col].str.title()

    # --- reconciliation region/province via le referentiel ---
    lct = lct.merge(ref, on="prov_key", how="left")
    # region/province canoniques quand on a trouve une correspondance
    lct["Region_std"] = lct["region_ref"].fillna(lct["Region"])
    lct["Province_std"] = lct["province_ref"].fillna(lct["Province"])
    lct["province_matched"] = lct["region_ref"].notna()

    return lct, communes


def completeness_report(lct: pd.DataFrame, communes: pd.DataFrame) -> pd.DataFrame:
    """Rapport : pour chaque province du referentiel, a-t-on des cas ? quelles annees ?"""
    communes["prov_key"] = communes["province"].map(norm_key)
    ref = (
        communes.groupby("prov_key")
        .agg(
            province=("province", "first"),
            region=("region", "first"),
            n_communes=("commune", "count"),
        )
        .reset_index()
    )

    cases = (
        lct.groupby("prov_key")
        .agg(
            n_cases=("Province", "size"),
            years=("Annee_Source", lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))),
        )
        .reset_index()
    )
    rep = ref.merge(cases, on="prov_key", how="left")
    rep["n_cases"] = rep["n_cases"].fillna(0).astype(int)
    rep["has_data"] = rep["n_cases"] > 0
    rep["years"] = rep["years"].apply(lambda v: v if isinstance(v, list) else [])
    rep["n_years_covered"] = rep["years"].apply(len)
    rep["status"] = rep["has_data"].map({True: "avec_donnees", False: "MANQUANT_gap"})
    rep = rep.sort_values(["has_data", "region", "province"]).reset_index(drop=True)
    return rep[["region", "province", "n_communes", "n_cases",
                "n_years_covered", "years", "status"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_LCT.exists():
        raise FileNotFoundError(f"Introuvable : {IN_LCT}")
    if not IN_COMMUNES.exists():
        raise FileNotFoundError(f"Introuvable : {IN_COMMUNES}")

    lct, communes = load_and_standardize(IN_LCT, IN_COMMUNES)
    lct.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    rep = completeness_report(lct, communes)
    rep.to_csv(OUT_MISSING, index=False, encoding="utf-8")

    # resume console
    n_prov_total = len(rep)
    if "has_data" in rep.columns:
        n_prov_gap = int((~rep["has_data"]).sum())
    else:
        n_prov_gap = int(rep["status"].eq("MANQUANT_gap").sum())
    n_unmatched = int((~lct["province_matched"]).sum())
    print("=" * 60)
    print("NETTOYAGE LCT termine")
    print("=" * 60)
    print(f"Lignes de cas          : {len(lct)}")
    print(f"Provinces (referentiel): {n_prov_total}")
    print(f"Provinces SANS donnee  : {n_prov_gap}  (-> a inferer par le modele bayesien)")
    print(f"Lignes province non reconnues au referentiel : {n_unmatched}")
    print("-" * 60)
    print(f"Ecrit : {OUT_CLEAN}")
    print(f"Ecrit : {OUT_MISSING}")


if __name__ == "__main__":
    main()
