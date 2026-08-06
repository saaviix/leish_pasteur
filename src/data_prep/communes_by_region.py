"""
communes_by_region.py
======================
Requete demandee : extraire TOUTES les communes du Maroc organisees par
Region -> Province -> Communes (avec lat/lon).

Entree  : data/raw/communes_maroc_final.csv
          (colonnes : id,commune,province,region,latitude,longitude,osm_id)

Sorties :
  - outputs/processed/communes_by_region.csv   (a plat, trie region/province/commune)
  - outputs/processed/communes_by_region.json  (hierarchie region > province > communes)
  - affichage console d'un resume (nb regions / provinces / communes)

Usage :
  python src/data_prep/communes_by_region.py
"""

import json
from pathlib import Path

import pandas as pd

import config

ROOT = config.ROOT
IN_CSV = config.COMMUNES_CSV
OUT_DIR = config.PROCESSED
OUT_CSV = OUT_DIR / "communes_by_region.csv"
OUT_JSON = OUT_DIR / "communes_by_region.json"


def load_communes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Verifie que communes_maroc_final.csv est bien dans data/raw/."
        )
    df = pd.read_csv(path)
    # colonnes attendues
    expected = {"commune", "province", "region", "latitude", "longitude"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {missing}")
    # nettoyage leger : on enleve les espaces autour des libelles texte
    for col in ["commune", "province", "region"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Table a plat triee, une ligne par commune."""
    cols = ["region", "province", "commune", "latitude", "longitude"]
    flat = (
        df[cols]
        .drop_duplicates(subset=["region", "province", "commune"])
        .sort_values(["region", "province", "commune"], key=lambda s: s.str.lower()
                     if s.dtype == object else s)
        .reset_index(drop=True)
    )
    return flat


def build_hierarchy(df: pd.DataFrame) -> dict:
    """Dict hierarchique region > province > liste de communes."""
    hierarchy: dict = {}
    for region, region_df in df.groupby("region"):
        provinces = {}
        for province, prov_df in region_df.groupby("province"):
            communes = [
                {
                    "commune": row["commune"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
                for _, row in prov_df.sort_values("commune").iterrows()
            ]
            provinces[province] = {
                "n_communes": len(communes),
                "communes": communes,
            }
        hierarchy[region] = {
            "n_provinces": len(provinces),
            "n_communes": int(region_df["commune"].nunique()),
            "provinces": provinces,
        }
    return hierarchy


def print_summary(df: pd.DataFrame, hierarchy: dict) -> None:
    n_regions = df["region"].nunique()
    n_provinces = df["province"].nunique()
    n_communes = df["commune"].nunique()
    print("=" * 60)
    print("RESUME : communes du Maroc par region / province")
    print("=" * 60)
    print(f"Regions   : {n_regions}")
    print(f"Provinces : {n_provinces}")
    print(f"Communes  : {n_communes}")
    print("-" * 60)
    for region in sorted(hierarchy):
        info = hierarchy[region]
        print(f"{region:35s} {info['n_provinces']:>3} prov  {info['n_communes']:>4} communes")
    print("=" * 60)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_communes(IN_CSV)

    flat = build_flat(df)
    flat.to_csv(OUT_CSV, index=False, encoding="utf-8")

    hierarchy = build_hierarchy(df)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    print_summary(df, hierarchy)
    print(f"\nEcrit : {OUT_CSV}")
    print(f"Ecrit : {OUT_JSON}")


if __name__ == "__main__":
    main()
