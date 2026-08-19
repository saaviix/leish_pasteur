"""
extract_climate.py
==================
Extrait les variables climatiques ERA5 pour chaque commune du Maroc.

Accepte deux formats d'entree :
  A) DONNEES HORAIRES (defaut apres download_era5.py sans --monthly) :
     agrege en "par jour" (moyenne des 24h) pour la base SQLite, puis en
     moyennes mensuelles long-terme pour les covariables du modele.
  B) DONNEES MENSUELLES (download_era5.py --monthly) :
     utilise directement.

Entrees :
  data/raw/era5_morocco_2009_2021_hourly.nc   (A - defaut)
  data/raw/era5_morocco_1990_2025.nc           (B - mensuel,ancien nom)
  data/raw/communes_maroc_final.csv

Sorties :
  outputs/processed/climate_morocco.db   (table 'climate' : commune x jour)
  outputs/processed/communes_climate.csv (moyennes mensuelles long-terme : covariables)

Usage :
  python src/data_prep/extract_climate.py

Robustesse : detecte automatiquement les noms de variables ERA5
(t2m/2m_temperature, tp/total_precipitation, etc.)
"""

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config

try:
    import xarray as xr
except ImportError:
    xr = None

# candidats de noms de variables ERA5
VAR_CANDIDATES = {
    "t2m":  ["t2m", "2m_temperature", "temperature_2m"],
    "d2m":  ["d2m", "2m_dewpoint_temperature", "dewpoint_temperature_2m"],
    "tp":   ["tp", "total_precipitation", "precipitation"],
    "si10": ["si10", "10m_wind_speed", "wind_speed_10m"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "pev":  ["pev", "potential_evaporation"],
}
LAT_CANDIDATES = ["latitude", "lat"]
LON_CANDIDATES = ["longitude", "lon"]
TIME_CANDIDATES = ["time", "valid_time"]


def pick(names, available):
    for n in names:
        if n in available:
            return n
    return None


def _find_nc_file() -> Path | list[Path]:
    """Retourne le fichier NetCDF (ou la liste de fichiers annee par annee)."""
    # 1) fichier unique (ancien comportement)
    for p in [config.ERA5_NC, config.ERA5_NC_MONTHLY, config.RAW / "era5_morocco.nc"]:
        if p.exists():
            return p
    # 2) fichiers annee par annee (horaires ou mensuels)
    per_year_hourly = sorted(config.RAW.glob("era5_morocco_*_hourly.nc"))
    per_year_monthly = sorted(config.RAW.glob("era5_morocco_*_monthly.nc"))
    if per_year_hourly:
        print(f"[INFO] {len(per_year_hourly)} fichiers horaires annee par annee detectes")
        return per_year_hourly
    if per_year_monthly:
        print(f"[INFO] {len(per_year_monthly)} fichiers mensuels annee par annee detectes")
        return per_year_monthly
    raise FileNotFoundError(
        f"NetCDF ERA5 introuvable dans {config.RAW}.\n"
        "Lance d'abord : python src/data_prep/download_era5.py\n"
        "  --env        (couches statiques)\n"
        "  --monthly    (donnees mensuelles, plus rapide)\n"
        "  --daily      (donnees horaires, annee par annee)"
    )


def _load_ds(path: Path):
    if xr is None:
        raise ImportError("xarray requis : pip install xarray netcdf4")
    print(f"[INFO] ouverture {path.name}")
    return xr.open_dataset(path)


def _resolve_names(ds):
    avail = set(ds.variables)
    latn = pick(LAT_CANDIDATES, avail)
    lonn = pick(LON_CANDIDATES, avail)
    timen = pick(TIME_CANDIDATES, avail)
    if not (latn and lonn and timen):
        raise ValueError(f"Coordonnees lat/lon/time introuvables. Variables : {sorted(avail)}")
    resolved = {k: pick(v, avail) for k, v in VAR_CANDIDATES.items()}
    resolved = {k: v for k, v in resolved.items() if v}
    print(f"[INFO] temps detecte : {timen}")
    print(f"[INFO] variables     : {resolved}")
    return latn, lonn, timen, resolved


def _aggregate(ds, latn, lonn, timen, resolved, communes):
    """
    Agrege le dataset (horaire ou mensuel) en :
      1) daily : moyenne de chaque jour (si donnees horaires)
      2) monthly : moyenne de chaque mois (pour covariables modele)

    Retourne :
      daily_df  : une ligne par commune x jour
      monthly_df : une ligne par commune x mois (moyennes long-terme)
    """
    lats = ds[latn].values
    lons = ds[lonn].values
    times = pd.to_datetime(ds[timen].values)

    # detecter si c'est horaire (< 1 ecart) ou mensuel (~30 jours)
    if len(times) > 1:
        td = (times[1] - times[0]).total_seconds()
        is_hourly = td < 86400
    else:
        is_hourly = False

    if is_hourly:
        print(f"[INFO] donnees HORAIRES detectees ({len(times)} pas de temps)")
        ds_daily = ds.resample({timen: "1D"}).mean()
        times_d = pd.to_datetime(ds_daily[timen].values)
    else:
        print(f"[INFO] donnees MENSUELLES detectees ({len(times)} pas de temps)")
        ds_daily = ds
        times_d = times

    daily_rows = []
    monthly_rows = []

    # Fix (trouve par audit) : le plus-proche-voisin brut tombe parfois sur
    # un pixel ERA5-Land masque (ocean) pres des cotes -> serie NaN sur toute
    # la duree pour la commune entiere. Confirme sur 24 communes cotieres,
    # dont Agadir (501797 hab.), Casablanca, Nador, Tanger, Fnideq -- 146 cas
    # LCT reels invisibles a tout modele climat-dependant (dropna). Precalcule
    # un masque de validite (sur t2m, la variable toujours presente) et,
    # quand le pixel le plus proche est masque, cherche le pixel valide le
    # plus proche par anneaux concentriques croissants plutot que d'accepter
    # un NaN silencieux.
    mask_invalid = None
    if "t2m" in resolved:
        mask_invalid = np.isnan(ds_daily[resolved["t2m"]].isel({timen: 0}).transpose(latn, lonn).values)

    def _nearest_valid(iy, ix, max_r=10):
        if mask_invalid is None or not mask_invalid[iy, ix]:
            return iy, ix, 0
        ny, nx = mask_invalid.shape
        for r in range(1, max_r + 1):
            candidates = [
                (iy + dy, ix + dx)
                for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if max(abs(dy), abs(dx)) == r and 0 <= iy + dy < ny and 0 <= ix + dx < nx
                and not mask_invalid[iy + dy, ix + dx]
            ]
            if candidates:
                best = min(candidates, key=lambda c: (lats[c[0]] - lats[iy]) ** 2 + (lons[c[1]] - lons[ix]) ** 2)
                return best[0], best[1], r
        return iy, ix, -1  # aucun voisin valide trouve dans le rayon -- reste NaN

    n_relocated = 0
    for _, com in communes.iterrows():
        clat, clon = com["latitude"], com["longitude"]
        iy = int(np.abs(lats - clat).argmin())
        ix = int(np.abs(lons - clon).argmin())
        iy, ix, moved_r = _nearest_valid(iy, ix)
        if moved_r > 0:
            n_relocated += 1

        # extraction des series pour cette commune
        series = {}
        for key, vname in resolved.items():
            series[key] = ds_daily[vname].isel({latn: iy, lonn: ix}).values

        months_temp = {}
        months_precip = {}

        for t_idx, ts in enumerate(times_d):
            t2m = series.get("t2m")
            d2m = series.get("d2m")
            tp = series.get("tp")

            temp_c = float(t2m[t_idx] - 273.15) if t2m is not None and not np.isnan(t2m[t_idx]) else None
            dew_c = float(d2m[t_idx] - 273.15) if d2m is not None and not np.isnan(d2m[t_idx]) else None
            # ERA5 "monthly averaged" tp est un TAUX QUOTIDIEN MOYEN sur le mois
            # (m/jour), pas un total mensuel deja accumule -- confirme empiriquement
            # (une region humide comme Chefchaouen donnait ~45mm/an avec la formule
            # precedente, ~10x trop bas). Multiplier par le nb de jours du mois
            # donne le vrai total mensuel en mm. Pour les donnees HORAIRES
            # resamplees en 1D (ds_daily), tp est deja un total/moyenne journaliere
            # au sens propre -- ne pas multiplier par days_in_month dans ce cas.
            if tp is not None and not np.isnan(tp[t_idx]):
                precip_rate_mm = float(tp[t_idx] * 1000.0)  # m -> mm
                precip = precip_rate_mm if is_hourly else precip_rate_mm * ts.days_in_month
            else:
                precip = None

            hum = None
            if temp_c is not None and dew_c is not None:
                hum = 100 * (math.exp((17.625 * dew_c) / (243.04 + dew_c)) /
                             math.exp((17.625 * temp_c) / (243.04 + temp_c)))

            daily_rows.append({
                "commune_id": int(com["id"]),
                "commune": com["commune"],
                "province": com["province"],
                "region": com["region"],
                "latitude": clat,
                "longitude": clon,
                "date": ts.strftime("%Y-%m-%d"),
                "annee": int(ts.year),
                "mois": int(ts.month),
                "temp_mean": round(temp_c, 2) if temp_c is not None else None,
                "humidity": round(hum, 1) if hum is not None else None,
                "precipitation": round(precip, 2) if precip is not None else None,
            })

            # accumuler pour le monthly long-terme -- DOIT etre dans la boucle
            # temporelle : c'etait par erreur au niveau de la boucle commune
            # (indentation), donc n'accumulait jamais que le DERNIER mois de
            # la serie au lieu des 156 mois -> precip_annual/temp_mean
            # long-terme totalement faux (calcules sur un seul mois de decembre).
            if temp_c is not None:
                months_temp.setdefault((ts.year, ts.month), []).append(temp_c)
            if precip is not None:
                months_precip.setdefault((ts.year, ts.month), []).append(precip)

        # moyennes mensuelles pour ce commune (une ligne par mois)
        all_months = set(months_temp) | set(months_precip)
        for yr, mo in sorted(all_months):
            monthly_rows.append({
                "commune_id": int(com["id"]),
                "commune": com["commune"],
                "province": com["province"],
                "region": com["region"],
                "latitude": clat,
                "longitude": clon,
                "annee": yr,
                "mois": mo,
                "temp_mean": round(float(np.nanmean(months_temp.get((yr, mo), [np.nan]))), 2),
                "precip_monthly": round(float(np.nanmean(months_precip.get((yr, mo), [np.nan]))), 2),
            })

    if n_relocated:
        print(f"[INFO] {n_relocated} communes relocalisees vers le pixel ERA5 valide le plus proche "
              f"(pixel initial masque -- typiquement cotier, cf. fix audit)")

    daily_df = pd.DataFrame(daily_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    return daily_df, monthly_df


def main() -> None:
    config.ensure_dirs()

    if xr is None:
        raise ImportError("xarray requis : pip install xarray netcdf4")

    nc_path = _find_nc_file()

    # support fichiers annee par annee : on concatene avec xarray
    if isinstance(nc_path, list):
        print(f"[INFO] concatenation de {len(nc_path)} fichiers annuels...")
        datasets = [xr.open_dataset(p) for p in nc_path]
        try:
            # bug corrige : concatener sur "time" alors que la dimension
            # temporelle reelle des fichiers ERA5 est "valid_time" creait une
            # dimension "time" (longueur = nb de fichiers) EN PLUS de
            # "valid_time" (longueur 12) inchangee dans chaque fichier, au
            # lieu de les fusionner -> tableaux 4D au lieu de 3D et crash
            # "truth value of an array with more than one element" plus loin.
            concat_dim = pick(TIME_CANDIDATES, set(datasets[0].variables))
            if concat_dim is None:
                raise ValueError(
                    f"Dimension temporelle introuvable dans {nc_path[0].name} : "
                    f"{sorted(datasets[0].variables)}"
                )
            ds = xr.concat(datasets, dim=concat_dim, coords="minimal",
                            compat="override", join="override")
            ds = ds.sortby(concat_dim)
        finally:
            for d in datasets:
                d.close()
        print(f"[INFO] concatene : {len(ds[concat_dim])} pas de temps (dimension '{concat_dim}')")
    else:
        ds = _load_ds(nc_path)

    latn, lonn, timen, resolved = _resolve_names(ds)
    communes = pd.read_csv(config.COMMUNES_CSV)

    daily_df, monthly_df = _aggregate(ds, latn, lonn, timen, resolved, communes)
    ds.close()

    print(f"[INFO] daily   : {len(daily_df)} lignes  (commune x jour)")
    print(f"[INFO] monthly : {len(monthly_df)} lignes (commune x mois, covariables modele)")

    # --- SQLite daily pour le dashboard ---
    conn = sqlite3.connect(config.CLIMATE_DB)
    daily_df.to_sql("climate", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geo ON climate(region, province, commune)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON climate(annee, mois)")
    conn.commit()
    conn.close()
    print(f"[OK] SQLite daily : {config.CLIMATE_DB}")

    # --- CSV long-terme par commune (covariables pour le modele) ---
    # moyenne de toutes les annees : precip_monthly * 12 = precip_annuelle
    clim_agg = (
        monthly_df.groupby(["commune_id", "commune", "province", "region", "latitude", "longitude"])
        .agg(
            temp_mean=("temp_mean", "mean"),
            precip_annual=("precip_monthly", lambda s: s.mean() * 12 if s.notna().any() else np.nan),
        )
        .reset_index()
    )
    clim_agg.to_csv(config.COMMUNES_CLIMATE, index=False, encoding="utf-8")
    print(f"[OK] covariables : {config.COMMUNES_CLIMATE}")


if __name__ == "__main__":
    main()
