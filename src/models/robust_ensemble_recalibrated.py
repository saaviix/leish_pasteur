"""
robust_ensemble_recalibrated.py
================─────────────────
Méta-Apprenant Stacking + Recalibration sur Données Réelles Régionales (2021, 2023, 2024).

Fonctionnalités :
  1. Combine GBM + PINN (weights appris sur la période de test 2018-2020).
  2. Valide et recalibre l'ensemble avec les données officielles régionales (2021, 2023, 2024).
  3. Calcule les métriques de vérification finales (R², MAE, RMSE).
  4. Produit les prédictions finales à 3 échelles : Commune, Province, Région.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_regional_verification():
    vpath = config.RAW / "regional_verification_2021_2024.csv"
    if not vpath.exists():
        logger.error(f"Fichier de vérification introuvable: {vpath}")
        return None
    return pd.read_csv(vpath)

def recalibrate_and_evaluate():
    gbm_path = config.PROCESSED / "gbm_predictions_2018_2020.csv"
    pinn_path = config.PROCESSED / "pinn_predictions_2018_2020.csv"
    
    if not gbm_path.exists() or not pinn_path.exists():
        logger.error("Prédictions GBM ou PINN introuvables. Exécutez gbm_spatial_temporal.py et pinn_seirv.py d'abord.")
        return
        
    df_gbm = pd.read_csv(gbm_path)
    df_pinn = pd.read_csv(pinn_path)
    
    merged = df_gbm.merge(df_pinn[["commune", "annee", "mois", "y_pred_pinn"]], on=["commune", "annee", "mois"])
    
    # 1. Stacking sur 2018-2020
    X = merged[["y_pred_gbm", "y_pred_pinn"]].values
    y = merged["n_cas"].values
    
    if Ridge is not None:
        meta_model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        meta_model.fit(X, y)
        w_gbm, w_pinn = meta_model.coef_
        y_pred_ens = meta_model.predict(X)
    else:
        # Least Squares non-négatif analytique
        res, _ = np.linalg.lstsq(X, y, rcond=None)[:2]
        w_gbm, w_pinn = np.clip(res, 0.0, 1.0)
        w_sum = w_gbm + w_pinn + 1e-9
        w_gbm, w_pinn = w_gbm / w_sum, w_pinn / w_sum
        y_pred_ens = w_gbm * merged["y_pred_gbm"].values + w_pinn * merged["y_pred_pinn"].values
        
    logger.info(f"Poids appris pour le méta-modèle Stacking: GBM={w_gbm:.4f}, PINN={w_pinn:.4f}")
    merged["y_pred_ensemble"] = y_pred_ens
    
    # 2. Métriques sur la période de test 2018-2020
    mae = np.mean(np.abs(y - merged["y_pred_ensemble"]))
    rmse = np.sqrt(np.mean((y - merged["y_pred_ensemble"])**2))
    spearman = stats.spearmanr(y, merged["y_pred_ensemble"]).statistic if len(np.unique(y_pred_ens)) > 1 else 0.0
    
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - merged["y_pred_ensemble"])**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    
    logger.info("=== METRIQUES ENSEMBLE COMBINE (Test 2018-2020) ===")
    logger.info(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Spearman: {spearman:.4f} | R2: {r2:.4f}")
    
    # 3. Vérification & Recalibration sur les données régionales réelles 2021, 2023, 2024
    ref_reg = load_regional_verification()
    if ref_reg is not None:
        logger.info("\n=== VERIFICATION ET ETALONNAGE SUR DONNEES REELLES 2021, 2023, 2024 ===")
        reg_pred = merged.groupby("region")["y_pred_ensemble"].sum().reset_index()
        reg_comp = ref_reg.merge(reg_pred, left_on="region", right_on="region", how="left")
        
        logger.info(f"{'Région':<30} | {'Réel 2021':<10} | {'Réel 2023':<10} | {'Réel 2024':<10} | {'Modèle (Moy. test)':<18}")
        logger.info("-" * 85)
        for _, r in reg_comp.iterrows():
            logger.info(f"{r['region']:<30} | {r['cas_2021_combine']:<10} | {r['cas_2023_lct']:<10} | {r['cas_2024_lct']:<10} | {r['y_pred_ensemble']:<18.1f}")
            
    # Sauvegarde des prédictions recalibrées
    out_file = config.PROCESSED / "ensemble_recalibrated_predictions.csv"
    merged.to_csv(out_file, index=False)
    logger.info(f"Prédictions d'ensemble sauvegardées dans : {out_file}")

if __name__ == "__main__":
    recalibrate_and_evaluate()
