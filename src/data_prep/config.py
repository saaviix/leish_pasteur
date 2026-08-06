"""
config.py
=========
Chemins centralises du projet. Tous les scripts importent d'ici pour eviter
les chemins codes en dur.
"""

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

# --- bases pour le dashboard ---
CLIMATE_DB = PROCESSED / "climate_morocco.db"
ENV_DB = PROCESSED / "environment_morocco.db"
GEOJSON = PROCESSED / "communes_morocco.geojson"


def ensure_dirs():
    for d in (PROCESSED, POSTERIOR, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
