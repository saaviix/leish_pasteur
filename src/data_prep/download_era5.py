"""
download_era5.py
================
Telecharge les donnees ERA5 pour le Maroc depuis le Climate Data Store (CDS)
Copernicus via cdsapi.

Deux modes :
  --daily   (defaut) : donnees HORAIRES ERA5, annees 2009-2021, decoupees par
                annee. extract_climate.py agregera ensuite en "par jour" puis
                en "par mois" pour le modele et le dashboard.
  --monthly : moyennes mensuelles (plus rapide, moins de donnees).

Zone Maroc : [Nord, Ouest, Sud, Est] = [36, -13.5, 21, -1]

PREREQUIS (une seule fois) :
  1. Compte gratuit https://cds.climate.copernicus.eu/
  2. Accepter la licence ERA5 (onglet Download de la page du dataset).
  3. Recuperer ta cle API : https://cds.climate.copernicus.eu/how-to-api
  4. Creer ~/.cdsapirc (Windows : C:\\Users\\<toi>\\.cdsapirc) :
       url: https://cds.climate.copernicus.eu/api
       key: <UID>:<API-KEY>
  5. pip install cdsapi xarray netcdf4

Usage :
  python src/data_prep/download_era5.py            # daily 2009-2021 (defaut)
  python src/data_prep/download_era5.py --monthly  # mensuel 2009-2021
  python src/data_prep/download_era5.py --daily --start 2009 --end 2021
  python src/data_prep/download_era5.py --env      # couches env statiques ERA5-Land
"""

import argparse
import sys
from pathlib import Path

import config

MOROCCO_AREA = [36.0, -13.5, 21.0, -1.0]

# climatiques : temperature 2m, point de rosee (->humidite), precip
# Note: le vent/rayonnement/evap sont references dans le dashboard mais optionnels;
#       on ne telecharge que l'essentiel pour rester dans les limites du CDS gratuit.
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
]
VARIABLES_LAND = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

# environnementales statiques (sol, vegetation, altitude)
VARIABLES_ENV = [
    "soil_type",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "high_vegetation_cover",
    "low_vegetation_cover",
    "geopotential",
]


def _daily_request(start, end, variables, land):
    """Requete HORAIRE : une annee a la fois pour respecter la limite de cout CDS."""
    dataset = "reanalysis-era5-land" if land else "reanalysis-era5-single-levels"
    return start, end, dataset, variables  # on boucle annee par annee dans main()


def _monthly_request(start, end, variables, land):
    """Requete MENSUELLE (plus legere)."""
    years = [str(y) for y in range(start, end + 1)]
    dataset = "reanalysis-era5-land-monthly-means" if land else "reanalysis-era5-single-levels-monthly-means"
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variables,
        "year": years,
        "month": ALL_MONTHS,
        "time": ["00:00"],
        "area": MOROCCO_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, dataset


def _env_request(end, land):
    """Couches environnementales statiques (un seul pas de temps)."""
    dataset = "reanalysis-era5-land" if land else "reanalysis-era5-single-levels"
    return {
        "variable": VARIABLES_ENV,
        "year": [str(end)],
        "month": ["06"],
        "day": ["01"],
        "time": ["12:00"],
        "area": MOROCCO_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Telecharge ERA5 pour le Maroc")
    ap.add_argument("--daily", action="store_true", default=True,
                    help="donnees horaires (defaut) -> agregees en daily par extract_climate.py")
    ap.add_argument("--monthly", action="store_true",
                    help="moyennes mensuelles (plus rapide)")
    ap.add_argument("--start", type=int, default=2009)
    ap.add_argument("--end", type=int, default=2021)
    ap.add_argument("--land", action="store_true", help="ERA5-Land (0.1 deg)")
    ap.add_argument("--env", action="store_true",
                    help="couches ENV statiques -> data/raw/era5_env_static.nc")
    args = ap.parse_args()

    config.ensure_dirs()
    config.RAW.mkdir(parents=True, exist_ok=True)

    try:
        import cdsapi
    except ImportError:
        print("ERREUR : cdsapi non installe.  ->  pip install cdsapi")
        sys.exit(1)

    try:
        client = cdsapi.Client()
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter au CDS : {e}")
        print("  Verifie que le fichier ~/.cdsapirc existe et contient ta cle API.")
        print("  Guide : https://cds.climate.copernicus.eu/how-to-api")
        sys.exit(1)

    # ---- mode environnement statique ----
    if args.env:
        request, dataset = _env_request(args.end, args.land)
        target = str(config.RAW / "era5_env_static.nc")
        print(f"[CDS] couches env : {dataset}")
        print(f"  variables : {VARIABLES_ENV}")
        print(f"  sortie    : {target}")
        client.retrieve(dataset, request, target)
        print(f"[OK] ecrit {target}")
        return

    # ---- mode climatique ----
    # Always download year-by-year to stay within CDS free-tier cost limits.
    start, end = args.start, args.end
    years = list(range(start, end + 1))
    if args.monthly:
        dataset = "reanalysis-era5-land-monthly-means" if args.land else "reanalysis-era5-single-levels-monthly-means"
        request_base = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": VARIABLES,
            "month": ALL_MONTHS,
            "time": ["00:00"],
            "area": MOROCCO_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        out_dir = config.RAW
        print("=" * 70)
        print(f"Telechargement ERA5 [mensuel annee par annee]  {start}-{end}")
        print(f"  dataset   : {dataset}")
        print(f"  variables : {len(VARIABLES)}")
        print(f"  sortie    : {out_dir}")
        print("=" * 70)
        for idx, year in enumerate(years, 1):
            target = str(out_dir / f"era5_morocco_{year}_monthly.nc")
            req = dict(request_base, year=[str(year)])
            print(f"\n[{idx}/{len(years)}] Annee {year} -> {Path(target).name}")
            print("  (mis en queue cote CDS ; patiente...)")
            client.retrieve(dataset, req, target)
            print(f"  [OK] {target}")
        print(f"\n[OK] {len(years)} fichiers mensuels telecharges dans {out_dir}")
        print("Etape suivante :  python src/data_prep/extract_climate.py")
    else:
        dataset = "reanalysis-era5-land" if args.land else "reanalysis-era5-single-levels"
        request_base = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "area": MOROCCO_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        out_dir = config.RAW
        print("=" * 70)
        print(f"Telechargement ERA5 [horaire annee par annee]  {start}-{end}")
        print(f"  dataset   : {dataset}")
        print(f"  variables : {len(VARIABLES)}")
        print(f"  sortie    : {out_dir}")
        print("=" * 70)
        for idx, year in enumerate(years, 1):
            target = str(out_dir / f"era5_morocco_{year}_hourly.nc")
            req = dict(request_base, year=[str(year)])
            print(f"\n[{idx}/{len(years)}] Annee {year} -> {Path(target).name}")
            print("  (mis en queue cote CDS ; patiente...)")
            client.retrieve(dataset, req, target)
            print(f"  [OK] {target}")
        print(f"\n[OK] {len(years)} fichiers horaires telecharges dans {out_dir}")
        print("Etape suivante :  python src/data_prep/extract_climate.py")


if __name__ == "__main__":
    main()
