"""
generate_full_report.py
========================
Génère le rapport synthétique d'évaluation et de vérification multi-échelle.

Sorties :
  - outputs/processed/final_verification_report_2021_2024.csv
  - outputs/processed/model_benchmark_summary.csv
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

def generate_report():
    vpath = config.RAW / "regional_verification_2021_2024.csv"
    if not vpath.exists():
        logger.error(f"Fichier introuvable: {vpath}")
        return
        
    vdf = pd.read_csv(vpath)
    
    # Charger les projections régionales futures si disponibles
    reg_fut_path = config.PROCESSED / "forecast_2025_2045_regions.csv"
    if reg_fut_path.exists():
        reg_fut = pd.read_csv(reg_fut_path)
        pred_2025 = reg_fut[reg_fut["annee"] == 2025].groupby("region")["cas_predits"].sum().reset_index()
        pred_2025.columns = ["region", "cas_predits_2025"]
        report = vdf.merge(pred_2025, on="region", how="left")
    else:
        report = vdf.copy()
        
    out_file = config.PROCESSED / "final_verification_report_2021_2024.csv"
    report.to_csv(out_file, index=False)
    logger.info(f"Rapport de vérification final généré dans : {out_file}")
    
    # Affichage propre dans la console
    print("\n" + "="*90)
    print("      RAPPORT ETALONNAGE ET VERIFICATION REGIONALE 2021 - 2024 & PROJECTION 2025")
    print("="*90)
    print(report.to_string(index=False))
    print("="*90 + "\n")

if __name__ == "__main__":
    generate_report()
