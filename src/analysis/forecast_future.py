"""
forecast_future.py
==================
Génère les projections épidémiologiques de LCT à 20 ans (2025-2045).

Sorties :
  - outputs/processed/forecast_2025_2045_communes.csv
  - outputs/processed/forecast_2025_2045_provinces.csv
  - outputs/processed/forecast_2025_2045_regions.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_future_forecasts(gbm_model=None):
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        logger.error(f"Fichier panel introuvable: {panel_path}")
        return
        
    panel = pd.read_csv(panel_path)
    
    # Identification robuste des colonnes de localisation
    cols = [c for c in panel.columns if c in ["commune", "province", "region", "latitude", "longitude"]]
    communes = panel[cols].drop_duplicates(subset=["commune"])
    
    logger.info(f"Génération de la grille de projection 2025-2045 pour {len(communes)} communes...")
    future_years = list(range(2025, 2046))
    future_months = list(range(1, 13))
    
    rows = []
    # Calcul des moyennes climatiques historiques par commune
    clim_hist = panel[panel["annee"] <= 2021].groupby(["commune", "mois"])[["temp_moy", "precip_mm", "humidite_pct"]].mean().reset_index()
    
    for _, comm in communes.iterrows():
        c_name = comm["commune"]
        c_hist = clim_hist[clim_hist["commune"] == c_name]
        
        for y in future_years:
            # Réchauffement progressif
            delta_t = (y - 2021) * 0.03
            
            for m in future_months:
                h_m = c_hist[c_hist["mois"] == m]
                if not h_m.empty:
                    t_val = h_m.iloc[0]["temp_moy"] + delta_t
                    p_val = h_m.iloc[0]["precip_mm"]
                    h_val = h_m.iloc[0]["humidite_pct"]
                else:
                    t_val, p_val, h_val = 20.0, 30.0, 50.0
                    
                rec = {
                    "commune": c_name,
                    "province": comm.get("province", "Inconnue"),
                    "region": comm.get("region", "Inconnue"),
                    "latitude": comm.get("latitude", 31.0),
                    "longitude": comm.get("longitude", -7.0),
                    "annee": y,
                    "mois": m,
                    "temp_moy": t_val,
                    "precip_mm": p_val,
                    "humidite_pct": h_val,
                    "sin_month": np.sin(2 * np.pi * m / 12.0),
                    "cos_month": np.cos(2 * np.pi * m / 12.0)
                }
                rows.append(rec)
                
    future_df = pd.DataFrame(rows)
    
    # Lags synthétiques
    for lag in range(1, 7):
        future_df[f"temp_moy_lag{lag}"] = future_df["temp_moy"]
        future_df[f"precip_mm_lag{lag}"] = future_df["precip_mm"]
        future_df[f"humidite_pct_lag{lag}"] = future_df["humidite_pct"]
        
    feature_cols = [c for c in future_df.columns if ("lag" in c or c in ["latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct", "sin_month", "cos_month"])]
    
    if gbm_model is not None:
        X_fut = future_df[feature_cols].fillna(0.0)
        future_df["cas_predits"] = np.clip(gbm_model.predict(X_fut), 0, None)
    else:
        # Estimation épidémiologique baseline basée sur la température et l'humidité
        t_factor = np.clip(future_df["temp_moy"] - 15.0, 0, None) / 10.0
        h_factor = future_df["humidite_pct"] / 60.0
        future_df["cas_predits"] = np.clip(t_factor * h_factor * (1.0 + 0.5 * future_df["sin_month"]), 0, 50)
        
    # Bornes d'incertitude 95%
    future_df["ci_lower_95"] = np.clip(future_df["cas_predits"] - 1.96 * np.sqrt(future_df["cas_predits"] + 1e-6), 0, None)
    future_df["ci_upper_95"] = future_df["cas_predits"] + 1.96 * np.sqrt(future_df["cas_predits"] + 1e-6)
    
    # 1. Export Communes
    comm_out = config.PROCESSED / "forecast_2025_2045_communes.csv"
    future_df[["commune", "province", "region", "annee", "mois", "cas_predits", "ci_lower_95", "ci_upper_95"]].to_csv(comm_out, index=False)
    logger.info(f"Projections par commune exportées : {comm_out}")
    
    # 2. Export Provinces
    prov_df = future_df.groupby(["province", "region", "annee", "mois"])[["cas_predits", "ci_lower_95", "ci_upper_95"]].sum().reset_index()
    prov_out = config.PROCESSED / "forecast_2025_2045_provinces.csv"
    prov_df.to_csv(prov_out, index=False)
    logger.info(f"Projections par province exportées : {prov_out}")
    
    # 3. Export Régions
    reg_df = future_df.groupby(["region", "annee", "mois"])[["cas_predits", "ci_lower_95", "ci_upper_95"]].sum().reset_index()
    reg_out = config.PROCESSED / "forecast_2025_2045_regions.csv"
    reg_df.to_csv(reg_out, index=False)
    logger.info(f"Projections par région exportées : {reg_out}")

if __name__ == "__main__":
    generate_future_forecasts()
