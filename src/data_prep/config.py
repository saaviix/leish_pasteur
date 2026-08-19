"""
config.py
=========
Chemins centralises du projet. Tous les scripts importent d'ici pour eviter
les chemins codes en dur.

Fournit aussi les utilitaires de reconciliation de libelles partages par tout
le pipeline (norm_key, CANONICAL_REGIONS) : avant la Phase 2 de la refonte,
chaque script (clean_lct.py, build_province_table.py, ...) redefinissait sa
propre version legerement differente de la meme fonction de normalisation,
ce qui causait des pertes de jointure silencieuses (ex. "Eddakhla-Oued
Eddahab" vs "Dakhla-Oued Ed-Dahab" ne matchaient nulle part).
"""

import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- entrees ---
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
# les PDF d'articles sont directement dans data/external/ (glob recursif gere
# aussi un sous-dossier articles/ si tu en crees un plus tard)
ARTICLES = EXTERNAL

COMMUNES_CSV = RAW / "communes_maroc_final.csv"
LCT_CSV = RAW / "leish_LCT.csv"
SERGENTI_CSV = RAW / "phlebotomus_sergenti_par_province.csv"
MUN_POP_SOURCE_XLSX = RAW / "mun_pop_source.xlsx"  # source brute (etait mun_pop.csv, en realite un xlsx)
MUN_POP_CSV = RAW / "mun_pop.csv"  # genere par fix_mun_pop.py, vrai CSV reconcilie au referentiel
REGIONAL_VERIF_CSV = RAW / "regional_verification_2021_2024.csv"

# NetCDF ERA5 horaire 2009-2021 (defaut apres download_era5.py sans --monthly)
ERA5_NC = RAW / "era5_morocco_2009_2021_hourly.nc"
# NetCDF ERA5 mensuel (download_era5.py --monthly)
ERA5_NC_MONTHLY = RAW / "era5_morocco.nc"
# couches environnementales statiques ERA5-Land (sol, vegetation, altitude)
ENV_STATIC_NC = RAW / "era5_env_static.nc"

# --- sorties ---
OUTPUTS = ROOT / "outputs"
PROCESSED = OUTPUTS / "processed"
POSTERIOR = OUTPUTS / "posterior"
FIGURES = OUTPUTS / "figures"

PROVINCE_TABLE = PROCESSED / "province_table.csv"
ADJ_EDGES = PROCESSED / "adjacency_edges.npy"
COMMUNES_CLIMATE = PROCESSED / "communes_climate.csv"
POSTERIOR_CSV = POSTERIOR / "psergenti_posterior_presence.csv"
ZONE_BIOCLIM_CSV = PROCESSED / "zone_bioclim_province.csv"
ELEVATION_CLASSES_CSV = PROCESSED / "province_elevation_classes.csv"
ENSEMBLE_PRED_CSV = PROCESSED / "ensemble_recalibrated_predictions.csv"
FORECAST_COMMUNES_CSV = PROCESSED / "forecast_2025_2045_communes.csv"
FORECAST_PROVINCES_CSV = PROCESSED / "forecast_2025_2045_provinces.csv"
FORECAST_REGIONS_CSV = PROCESSED / "forecast_2025_2045_regions.csv"

# --- bases pour le dashboard ---
CLIMATE_DB = PROCESSED / "climate_morocco.db"
ENV_DB = PROCESSED / "environment_morocco.db"
GEOJSON = PROCESSED / "communes_morocco.geojson"


def ensure_dirs():
    for d in (PROCESSED, POSTERIOR, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reconciliation de libelles (partagee par tout le pipeline)
# ---------------------------------------------------------------------------

def norm_key(s) -> str:
    """Cle de rapprochement robuste pour region/province/commune : sans
    accents, minuscule, tirets/apostrophes/slashs traites comme des espaces,
    espaces normalises. Utiliser cette fonction partout plutot que d'en
    redefinir une variante locale (source des pertes de jointure silencieuses
    identifiees dans l'audit Phase 2 : ex. "Aghbalou-N'Kerdous" vs
    "Aghbalou N'Kerdous" vs "Aghbalou-N/Kerdous")."""
    import pandas as pd  # import local pour eviter la dependance a l'import du module

    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s).strip()).encode("ascii", "ignore").decode()
    for ch in "-'/":
        s = s.replace(ch, " ")
    return " ".join(s.lower().split())


# Variantes orthographiques de regions connues -> libelle canonique du
# referentiel (communes_maroc_final.csv). A completer si de nouvelles
# variantes apparaissent dans d'autres sources.
CANONICAL_REGIONS = {
    norm_key("Eddakhla-Oued Eddahab"): "Dakhla-Oued Ed-Dahab",
}


def canonical_region(region: str) -> str:
    """Renvoie le libelle de region canonique si une variante connue est
    detectee, sinon renvoie la valeur d'origine inchangee."""
    return CANONICAL_REGIONS.get(norm_key(region), region)
