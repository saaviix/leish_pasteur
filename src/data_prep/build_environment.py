"""
build_environment.py
====================
Construit une base d'attributs environnementaux par commune.

Deux sources possibles pour l'altitude / vegetation / sol :
  A) le NetCDF ERA5-Land statique data/raw/era5_env_static.nc
     (telecharge par : python src/data_prep/download_era5.py --env)
     -> altitude (via geopotential), soil_type, couverture vegetale (LAI, cover)
  B) sinon, fallback : altitude via l'API d'elevation Open-Meteo.

Plus : aridity_index (indice de Martonne) derive de communes_climate.csv.

Sortie :
  outputs/processed/environment_morocco.db  (table 'environment')

Usage :
  python src/data_prep/build_environment.py
"""

import sqlite3
import time

import numpy as np
import pandas as pd

import config

try:
    import requests
except ImportError:
    requests = None

try:
    import xarray as xr
except ImportError:
    xr = None

ELEV_URL = "https://api.open-meteo.com/v1/elevation"
BATCH = 100
G0 = 9.80665  # gravite standard (geopotential -> altitude)


def pick(names, available):
    for n in names:
        if n in available:
            return n
    return None


def sample_era5_env(env_df):
    """Ajoute altitude/sol/vegetation depuis era5_env_static.nc si present."""
    if xr is None or not config.ENV_STATIC_NC.exists():
        return env_df, False
    print(f"[INFO] lecture couches env ERA5 : {config.ENV_STATIC_NC.name}")
    ds = xr.open_dataset(config.ENV_STATIC_NC)
    if "valid_time" in ds.dims or "time" in ds.dims:
        ds = ds.isel({[d for d in ("valid_time", "time") if d in ds.dims][0]: 0})
    avail = set(ds.variables)
    latn = pick(["latitude", "lat"], avail)
    lonn = pick(["longitude", "lon"], avail)
    v_geo = pick(["z", "geopotential"], avail)
    v_soil = pick(["slt", "soil_type"], avail)
    v_laih = pick(["lai_hv", "leaf_area_index_high_vegetation"], avail)
    v_lail = pick(["lai_lv", "leaf_area_index_low_vegetation"], avail)

    lats = ds[latn].values
    lons = ds[lonn].values
    alt, soil, veg = [], [], []
    for _, r in env_df.iterrows():
        iy = int(np.abs(lats - r["latitude"]).argmin())
        ix = int(np.abs(lons - r["longitude"]).argmin())
        alt.append(float(ds[v_geo].isel({latn: iy, lonn: ix}).values / G0) if v_geo else np.nan)
        soil.append(float(ds[v_soil].isel({latn: iy, lonn: ix}).values) if v_soil else np.nan)
        lh = float(ds[v_laih].isel({latn: iy, lonn: ix}).values) if v_laih else 0.0
        ll = float(ds[v_lail].isel({latn: iy, lonn: ix}).values) if v_lail else 0.0
        veg.append(lh + ll)
    env_df["altitude_m"] = np.round(alt, 1)
    env_df["soil_type"] = soil
    env_df["vegetation_lai"] = np.round(veg, 3)
    return env_df, True


def fetch_elevations(lats, lons):
    """Altitude via Open-Meteo (batch de 100). Retourne liste (NaN si echec).
    Avec retry/backoff exponentiel en cas de 429 Too Many Requests."""
    if requests is None:
        print("[WARN] requests non installe -> altitude = NaN")
        return [np.nan] * len(lats)
    out = []
    for i in range(0, len(lats), BATCH):
        blat = lats[i:i + BATCH]
        blon = lons[i:i + BATCH]
        for attempt in range(5):  # jusqu'a 5 tentatives par batch
            try:
                r = requests.get(
                    ELEV_URL,
                    params={"latitude": ",".join(map(str, blat)),
                            "longitude": ",".join(map(str, blon))},
                    timeout=30,
                )
                if r.status_code == 429:
                    wait = 2 ** attempt  # 1, 2, 4, 8, 16 secondes
                    print(f"[WARN] 429 Too Many Requests (batch {i}) -> attente {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                out.extend(r.json().get("elevation", [np.nan] * len(blat)))
                break  # succes, passer au batch suivant
            except Exception as e:
                wait = 2 ** attempt
                print(f"[WARN] elevation batch {i} tentative {attempt+1}: {e} -> attente {wait}s")
                time.sleep(wait)
        else:
            # toutes les tentatives ont echoue
            print(f"[ERREUR] elevation batch {i} : echec apres 5 tentatives -> NaN")
            out.extend([np.nan] * len(blat))
        time.sleep(1)  # delai minimum entre batches
    return out


def main() -> None:
    config.ensure_dirs()
    communes = pd.read_csv(config.COMMUNES_CSV)

    env = communes[["id", "commune", "province", "region", "latitude", "longitude"]].copy()
    env = env.rename(columns={"id": "commune_id"})

    # --- source A : couches env ERA5-Land statiques (altitude/sol/vegetation) ---
    env, used_era5 = sample_era5_env(env)

    # --- source B : fallback altitude via API pour les valeurs manquantes ---
    if "altitude_m" not in env.columns:
        # ERA5 n'a pas pu fournir l'altitude du tout
        print(f"[INFO] altitude via API Open-Meteo pour {len(env)} communes...")
        env["altitude_m"] = fetch_elevations(
            env["latitude"].round(4).tolist(), env["longitude"].round(4).tolist()
        )
    elif env["altitude_m"].isna().any():
        # ERA5 a des trous : on remplit seulement les NaN
        n_missing = int(env["altitude_m"].isna().sum())
        print(f"[INFO] altitude ERA5 avec {n_missing} trous -> remplissage via API...")
        missing_mask = env["altitude_m"].isna()
        env.loc[missing_mask, "altitude_m"] = fetch_elevations(
            env.loc[missing_mask, "latitude"].round(4).tolist(),
            env.loc[missing_mask, "longitude"].round(4).tolist(),
        )
    elif used_era5:
        print("[INFO] altitude/sol/vegetation issues d'ERA5-Land")

    # aridite (indice de Martonne) depuis le climat si dispo
    if config.COMMUNES_CLIMATE.exists():
        clim = pd.read_csv(config.COMMUNES_CLIMATE)[
            ["commune_id", "temp_mean", "precip_annual"]
        ]
        env = env.merge(clim, on="commune_id", how="left")
        env["aridity_index"] = env["precip_annual"] / (env["temp_mean"] + 33)
        env = env.drop(columns=["temp_mean", "precip_annual"])
    else:
        print("[WARN] communes_climate.csv absent -> aridity_index = NaN "
              "(lance extract_climate.py avant)")
        env["aridity_index"] = np.nan

    # placeholders a enrichir plus tard
    env["dist_water_km"] = np.nan
    env["pop_density"] = np.nan
    env["urban_class"] = None

    conn = sqlite3.connect(config.ENV_DB)
    env.to_sql("environment", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_env ON environment(region, province, commune)")
    conn.commit()
    conn.close()
    print(f"[OK] ecrit {config.ENV_DB} ({len(env)} communes)")


if __name__ == "__main__":
    main()
