"""
build_commune_panel.py
======================
Construit le panel spatio-temporel complet : Commune x Année x Mois.

Données d'entrée :
  - data/raw/communes_maroc_final.csv (1503 communes)
  - outputs/processed/lct_clean.csv (Cas de LCT 2009-2020)
  - data/raw/era5_morocco_*_monthly.nc (Données climatiques ERA5 2009-2021)
  - outputs/processed/province_table.csv (Caractéristiques environnementales statiques)

Sortie :
  - outputs/processed/commune_panel.csv (~1503 communes x 12 mois x 12 ans)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_communes():
    comm_path = config.RAW / "communes_maroc_final.csv"
    if not comm_path.exists():
        comm_path = config.PROCESSED / "communes_by_region.csv"
    df = pd.read_csv(comm_path)
    # Normalisation des colonnes
    col_map = {}
    for c in df.columns:
        clow = c.lower().strip()
        if clow in ["commune", "nom_commune", "municipality"]: col_map[c] = "commune"
        elif clow in ["province", "nom_province"]: col_map[c] = "province"
        elif clow in ["region", "nom_region"]: col_map[c] = "region"
        elif clow in ["lat", "latitude"]: col_map[c] = "latitude"
        elif clow in ["lon", "longitude"]: col_map[c] = "longitude"
    df = df.rename(columns=col_map)
    # Nettoyage
    df["commune"] = df["commune"].astype(str).str.strip()
    df["province"] = df["province"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    return df[["commune", "province", "region", "latitude", "longitude"]].drop_duplicates(subset=["commune"])

def load_lct_cases():
    lct_path = config.PROCESSED / "lct_clean.csv"
    if not lct_path.exists():
        lct_path = config.RAW / "leish_LCT.csv"
    df = pd.read_csv(lct_path)
    
    # Identifier colonnes
    comm_col = None
    for c in df.columns:
        if c.lower() in ["commune", "nom_commune", "commune_std"]:
            comm_col = c
            break
            
    year_col = None
    for c in df.columns:
        if c.lower() in ["annee", "annee_source", "year"]:
            year_col = c
            break
            
    month_col = None
    for c in df.columns:
        if c.lower() in ["mois", "mois_source", "mois_diagnostic", "month"]:
            month_col = c
            break
            
    cases_col = "n_cases" if "n_cases" in df.columns else ("Nbr_cas" if "Nbr_cas" in df.columns else "cases")
    
    if comm_col is None or year_col is None or month_col is None:
        logger.warning(f"Colonnes manquantes: comm_col={comm_col}, year_col={year_col}, month_col={month_col}")
        # Si cases_col n'existe pas, chaque ligne de lct_clean.csv représente 1 cas
        if cases_col not in df.columns:
            df["n_cas_count"] = 1
            cases_col = "n_cas_count"
    elif cases_col not in df.columns:
        df["n_cas_count"] = 1
        cases_col = "n_cas_count"
        
    df[comm_col] = df[comm_col].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
    
    # Aggréger par commune, année, mois
    cases = df.groupby([comm_col, year_col, month_col])[cases_col].sum().reset_index()
    cases.columns = ["commune", "annee", "mois", "n_cas"]
    cases = cases[(cases["annee"] >= 2009) & (cases["annee"] <= 2020) & (cases["mois"] >= 1) & (cases["mois"] <= 12)]
    return cases

def extract_climate_panel(communes_df):
    """Extrait le climat ERA5 mensuel pour chaque commune de 2009 à 2024."""
    logger.info("Extraction des séries climatiques ERA5 mensuelles...")
    
    monthly_files = sorted(config.RAW.glob("era5_morocco_*_monthly.nc"))
    if not monthly_files:
        # Fallback sur 1990_2025.nc s'il existe
        single_nc = config.RAW / "era5_morocco_1990_2025.nc"
        if single_nc.exists():
            monthly_files = [single_nc]
            
    records = []
    
    for nc_file in monthly_files:
        try:
            ds = xr.open_dataset(nc_file)
            
            # Nommer les dims
            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            time_name = "time" if "time" in ds.coords else "valid_time"
            
            t2m_name = "t2m" if "t2m" in ds.data_vars else ("2m_temperature" if "2m_temperature" in ds.data_vars else None)
            tp_name = "tp" if "tp" in ds.data_vars else ("total_precipitation" if "total_precipitation" in ds.data_vars else None)
            d2m_name = "d2m" if "d2m" in ds.data_vars else ("2m_dewpoint_temperature" if "2m_dewpoint_temperature" in ds.data_vars else None)
            
            times = pd.to_datetime(ds[time_name].values)
            
            for _, comm in communes_df.iterrows():
                c_lat, c_lon = comm["latitude"], comm["longitude"]
                
                # Extraction du point le plus proche
                sub_ds = ds.sel({lat_name: c_lat, lon_name: c_lon}, method="nearest")
                
                t2m = sub_ds[t2m_name].values - 273.15 if t2m_name else np.nan # K -> °C
                tp = sub_ds[tp_name].values * 1000.0 if tp_name else np.nan # m -> mm
                
                if d2m_name:
                    d2m = sub_ds[d2m_name].values - 273.15
                    # Humidité relative approximative Magnus
                    rh = 100.0 * np.exp((17.625 * d2m) / (243.04 + d2m)) / np.exp((17.625 * t2m) / (243.04 + t2m))
                    rh = np.clip(rh, 0.0, 100.0)
                else:
                    rh = np.nan
                    
                for idx, t in enumerate(times):
                    records.append({
                        "commune": comm["commune"],
                        "annee": t.year,
                        "mois": t.month,
                        "temp_moy": float(t2m[idx]) if hasattr(t2m, '__len__') else float(t2m),
                        "precip_mm": float(tp[idx]) if hasattr(tp, '__len__') else float(tp),
                        "humidite_pct": float(rh[idx]) if hasattr(rh, '__len__') else float(rh)
                    })
            ds.close()
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {nc_file.name}: {e}")
            
    if not records:
        logger.error("Aucune donnée climatique extraite des fichiers NC.")
        return None
        
    df_clim = pd.DataFrame(records).drop_duplicates(subset=["commune", "annee", "mois"])
    return df_clim

def build_panel():
    communes = load_communes()
    logger.info(f"Chargé {len(communes)} communes distinctes.")
    
    cases = load_lct_cases()
    climate = extract_climate_panel(communes)
    
    if climate is None:
        logger.error("Impossible de créer le panel sans climat.")
        return
        
    # Créer la grille complète commune x année (2009-2024) x mois (1-12)
    years = list(range(2009, 2025))
    months = list(range(1, 13))
    
    grid = []
    for _, comm in communes.iterrows():
        for y in years:
            for m in months:
                grid.append({
                    "commune": comm["commune"],
                    "province": comm["province"],
                    "region": comm["region"],
                    "latitude": comm["latitude"],
                    "longitude": comm["longitude"],
                    "annee": y,
                    "mois": m
                })
    panel = pd.DataFrame(grid)
    
    # Merger avec le climat
    panel = panel.merge(climate, on=["commune", "annee", "mois"], how="left")
    
    # Merger avec les cas (remplir 0 pour les mois sans cas si annee <= 2020)
    if cases is not None:
        panel = panel.merge(cases, on=["commune", "annee", "mois"], how="left")
        # Si année <= 2020, NaN de cas = 0 cas
        panel.loc[panel["annee"] <= 2020, "n_cas"] = panel.loc[panel["annee"] <= 2020, "n_cas"].fillna(0)
    else:
        panel["n_cas"] = np.nan
        
    # Charger les infos statiques des provinces (altitude, etc.)
    pt_path = config.PROCESSED / "province_table.csv"
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        # Harmoniser les noms
        pt_cols = [c for c in pt.columns if c not in ["province", "lat", "lon"]]
        panel = panel.merge(pt[["province"] + pt_cols], on="province", how="left")
        
    # Harmoniser le nom de la colonne région si suffixe _x ou _y
    for col in panel.columns:
        if col.startswith("region"):
            panel["region"] = panel[col]
            break
    
    # Calculer les lags climatiques (1 à 6 mois) par commune
    logger.info("Calcul des lags climatiques (1 à 6 mois)...")
    for lag in range(1, 7):
        for col in ["temp_moy", "precip_mm", "humidite_pct"]:
            panel[f"{col}_lag{lag}"] = panel.groupby("commune")[col].shift(lag)
            
    # Encoder la saisonnalité (sinus / cosinus du mois)
    panel["sin_month"] = np.sin(2 * np.pi * panel["mois"] / 12.0)
    panel["cos_month"] = np.cos(2 * np.pi * panel["mois"] / 12.0)
    
    # Sauvegarder
    out_path = config.PROCESSED / "commune_panel.csv"
    panel.to_csv(out_path, index=False)
    logger.info(f"Panel spatio-temporel sauvegardé dans : {out_path} ({len(panel)} lignes)")

if __name__ == "__main__":
    build_panel()
