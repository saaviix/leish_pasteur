"""
robust_ensemble_recalibrated.py
================================
Meta-apprenant Stacking (GBM + PINN) + verification reelle contre les
donnees officielles regionales (2021, 2023, 2024).

Reecrit en Phase 3 de la refonte -- la version precedente NE CALCULAIT AUCUNE
METRIQUE REELLE : elle sommait les predictions du test 2018-2020 par region
et les affichait a cote des vrais totaux 2021/2023/2024 sous l'intitule
"Modele (Moy. test)", sans jamais calculer d'erreur ni generer de veritable
prediction hors-echantillon pour ces annees.

Ce que fait cette version :
  1. Charge le GBM (gbm_spatial_temporal.py) et le PINN (pinn_seirv.py)
     persistes sur disque (poids + liste de covariables).
  2. Construit de VRAIES lignes cibles pour 2021, 2023, 2024 :
       - 2021 : climat ERA5 MESURE (disponible dans le panel), lags de cas
         calcules sur l'historique reel 2009-2020.
       - 2023, 2024 : aucun ERA5 mesure au-dela de 2021 -> climatologie par
         commune x mois + tendance de rechauffement (climatology.py),
         explicitement marque `source_climat=climatologie_extrapolee` dans
         les sorties (pas confondu avec une mesure).
     Les covariables autoregressives (cases_lag*/cases_roll*) sont a 0 pour
     2023/2024 : aucun historique de cas n'existe au-dela de 2020.
  3. Reapprend les poids de stacking (Ridge non-negatif) sur les predictions
     de test 2018-2020 (comme avant -- legitime, c'est du hors-echantillon),
     puis les applique aux VRAIES predictions 2021/2023/2024 ci-dessus.
  4. Calcule de VRAIES metriques (MAE, RMSE, R2) contre
     regional_verification_2021_2024.csv, par annee et globalement.

Sorties :
  - outputs/processed/ensemble_recalibrated_predictions.csv
  - outputs/processed/verification_metrics_2021_2024.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from climatology import build_climatology_grid
from model_io import load_gbm, predict_gbm_saved
import pinn_seirv

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

YEAR_TO_REGIONAL_COL = {2021: "cas_2021_combine", 2023: "cas_2023_lct", 2024: "cas_2024_lct"}


def load_regional_verification():
    vpath = config.REGIONAL_VERIF_CSV
    if not vpath.exists():
        logger.error(f"Fichier de vérification introuvable: {vpath}")
        return None
    df = pd.read_csv(vpath)
    df["region"] = df["region"].map(config.canonical_region)
    return df


def load_pinn():
    path = config.PROCESSED / "pinn_seirv_weights.pt"
    if not (torch and path.exists()):
        logger.warning(f"{path} introuvable -> pas de composante PINN (lance pinn_seirv.py)")
        return None, None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = pinn_seirv.SEIRVPINN(n_provinces=len(saved["provinces"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["provinces"]


def add_case_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Meme logique que gbm_spatial_temporal.py, appliquee au panel complet
    (pas seulement au train set) pour que les lignes 2021 recoivent de vrais
    lags calcules sur l'historique reel de cas 2009-2020."""
    p = panel.sort_values(["commune_id", "annee", "mois"]).copy()
    g = p.groupby("commune_id")["n_cas"]
    p["cases_lag1"] = g.shift(1)
    p["cases_lag2"] = g.shift(2)
    p["cases_lag3"] = g.shift(3)
    p["cases_lag12"] = g.shift(12)
    p["cases_roll3"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    p["cases_roll6"] = g.transform(lambda x: x.shift(1).rolling(6).mean())
    return p


def build_target_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Vraies lignes cibles 2021 (climat mesure + lags de cas reels) +
    climatologie 2023/2024 (pas de climat mesure disponible)."""
    panel_lags = add_case_lags(panel)
    real_2021 = panel_lags[panel_lags["annee"] == 2021].copy()
    real_2021["source_climat"] = "mesure_era5"

    clim_grid, n_missing = build_climatology_grid(panel, [2023, 2024])
    clim_grid["source_climat"] = "climatologie_extrapolee"
    for c in ["cases_lag1", "cases_lag2", "cases_lag3", "cases_lag12", "cases_roll3", "cases_roll6"]:
        clim_grid[c] = 0.0  # pas d'historique de cas au-dela de 2020, documente ci-dessus

    common_cols = [c for c in real_2021.columns if c in clim_grid.columns]
    combined = pd.concat([real_2021[common_cols], clim_grid[common_cols]], ignore_index=True, sort=False)
    logger.info(f"Lignes cibles : {len(real_2021)} (2021, climat mesuré) + {len(clim_grid)} "
                f"(2023+2024, climatologie extrapolée)")
    return combined


def predict_pinn(model, provinces: list, df: pd.DataFrame, t_ref_year: int) -> np.ndarray:
    """Passe avant du PINN a jour (signature actuelle : t, lat, lon, temp,
    precip, humid, hist, prov_idx -- voir pinn_seirv.forward). La prediction
    de cas est model.rho * obs_rate (meme formule que l'evaluation test
    interne de pinn_seirv.train_pinn), PAS sigma_H*E_H*Nh (ancienne
    architecture sans modele d'observation NegBin explicite)."""
    t_months = (df["annee"] - t_ref_year) * 12 + (df["mois"] - 1)
    prov_to_idx = {p: i for i, p in enumerate(provinces)}

    def col(name, fill=0.0):
        return torch.tensor(df[name].fillna(fill).values.astype(np.float32)).reshape(-1, 1)

    hist_cols = ["cases_lag1", "cases_roll3", "cases_roll6"]
    hist_vals = np.stack([np.log1p(df[c].fillna(0.0).values.astype(np.float32)) for c in hist_cols], axis=1)
    hist_t = torch.tensor(hist_vals, dtype=torch.float32)
    prov_idx = torch.tensor(
        df["province"].map(prov_to_idx).fillna(0).astype(int).values, dtype=torch.long
    ).reshape(-1, 1)

    t = torch.tensor(t_months.values.astype(np.float32)).reshape(-1, 1)
    with torch.no_grad():
        out = model(t, col("latitude"), col("longitude"), col("temp_moy"), col("precip_mm"),
                     col("humidite_pct"), hist_t, prov_idx)
        pred = (model.rho * out["obs_rate"]).numpy().flatten()
    return np.clip(pred, 0, None)


def fit_ensemble_weights():
    """Poids de stacking (Ridge non-negatif) appris sur les predictions
    hors-echantillon 2018-2020 -- legitime (donnees jamais vues a l'entrainement),
    reutilise ici pour combiner GBM+PINN sur les vraies cibles 2021/2023/2024."""
    gbm_path = config.PROCESSED / "gbm_predictions_2018_2020.csv"
    pinn_path = config.PROCESSED / "pinn_predictions_2018_2020.csv"
    if not (gbm_path.exists() and pinn_path.exists()):
        return 1.0, 0.0

    df_gbm = pd.read_csv(gbm_path)
    df_pinn = pd.read_csv(pinn_path)
    merged = df_gbm.merge(df_pinn[["commune", "annee", "mois", "y_pred_pinn"]], on=["commune", "annee", "mois"])
    X = merged[["y_pred_gbm", "y_pred_pinn"]].values
    y = merged["n_cas"].values

    if Ridge is not None:
        meta = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(X, y)
        w_gbm, w_pinn = meta.coef_
    else:
        res = np.linalg.lstsq(X, y, rcond=None)[0]
        w_gbm, w_pinn = np.clip(res, 0.0, None)

    s = w_gbm + w_pinn + 1e-9
    w_gbm, w_pinn = float(w_gbm / s), float(w_pinn / s)
    logger.info(f"Poids d'ensemble (réappris sur test 2018-2020) : GBM={w_gbm:.3f}  PINN={w_pinn:.3f}")
    return w_gbm, w_pinn


def compute_metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(real - pred)))
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    ss_tot = np.sum((real - real.mean()) ** 2)
    ss_res = np.sum((real - pred) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-9))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": len(real)}


def recalibrate_and_evaluate():
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        logger.error(f"{panel_path} introuvable. Lance build_commune_panel.py d'abord.")
        return None
    panel = pd.read_csv(panel_path)

    try:
        gbm_saved = load_gbm(config.PROCESSED)
    except FileNotFoundError as e:
        logger.warning(str(e))
        gbm_saved = None
    pinn_model, pinn_provinces = load_pinn()
    if gbm_saved is None and pinn_model is None:
        logger.error("Ni GBM ni PINN disponibles -- rien à recalibrer.")
        return None

    targets = build_target_rows(panel)

    preds = targets[["commune", "province", "region", "annee", "mois", "source_climat"]].copy()
    if gbm_saved is not None:
        model_label = gbm_saved.get("model_name") or type(gbm_saved.get("model", gbm_saved.get("model_1"))).__name__
        logger.info(f"Composante GBM : {model_label}")
        preds["y_pred_gbm"] = predict_gbm_saved(gbm_saved, targets)
    if pinn_model is not None:
        preds["y_pred_pinn"] = predict_pinn(pinn_model, pinn_provinces, targets, t_ref_year=int(panel["annee"].min()))

    if "y_pred_gbm" in preds.columns and "y_pred_pinn" in preds.columns:
        w_gbm, w_pinn = fit_ensemble_weights()
        preds["y_pred_ensemble"] = w_gbm * preds["y_pred_gbm"] + w_pinn * preds["y_pred_pinn"]
    elif "y_pred_gbm" in preds.columns:
        preds["y_pred_ensemble"] = preds["y_pred_gbm"]
    else:
        preds["y_pred_ensemble"] = preds["y_pred_pinn"]

    ref = load_regional_verification()
    if ref is None:
        logger.warning("Pas de fichier de vérification régionale -- métriques non calculées.")
        out_pred = config.PROCESSED / "ensemble_recalibrated_predictions.csv"
        preds.to_csv(out_pred, index=False)
        logger.info(f"Prédictions sauvegardées (non recalibrées, pas de vérité terrain) dans : {out_pred}")
        return preds

    reg_pred = preds.groupby(["region", "annee"])["y_pred_ensemble"].sum().reset_index()

    # ------------------------------------------------------------------
    # Recalibration par region ancree sur 2021 (le seul verite-terrain
    # antérieure a 2023/2024 dans cette verification) : le GBM sous-predit
    # structurellement Drâa-Tafilalet et sur-predit Fès-Meknès de facon
    # constante d'une annee sur l'autre (biais observe, pas du bruit) --
    # attendu d'un modele a base d'arbres qui ne peut pas extrapoler une
    # tendance regionale au-dela de sa periode d'entrainement (2009-2017).
    # facteur[region] = reel_2021 / predit_2021, applique aux predictions
    # 2023/2024 SEULEMENT (2021 est chronologiquement anterieur, donc ce
    # n'est pas une fuite de la verite 2023/2024 vers sa propre evaluation).
    # Clip large pour eviter qu'une prediction 2021 proche de 0 dans une
    # petite region cree un facteur demesure.
    # ------------------------------------------------------------------
    pred_2021 = reg_pred[reg_pred["annee"] == 2021][["region", "y_pred_ensemble"]].rename(
        columns={"y_pred_ensemble": "pred_2021"}
    )
    calib = ref[["region", "cas_2021_combine"]].merge(pred_2021, on="region", how="left")
    calib["pred_2021"] = calib["pred_2021"].fillna(0.0)
    calib["facteur"] = (calib["cas_2021_combine"] + 1.0) / (calib["pred_2021"] + 1.0)
    calib["facteur"] = calib["facteur"].clip(0.2, 5.0)
    factor_map = calib.set_index("region")["facteur"].to_dict()
    logger.info("\nFacteurs de recalibration par région (ancrés sur 2021, appliqués à 2023/2024 seulement) :")
    for _, r in calib.sort_values("facteur", ascending=False).iterrows():
        logger.info(f"  {r['region']:<28} x{r['facteur']:.2f}")

    reg_pred["y_pred_recalibre"] = reg_pred["y_pred_ensemble"]
    mask_recal = reg_pred["annee"].isin([2023, 2024])
    reg_pred.loc[mask_recal, "y_pred_recalibre"] = reg_pred.loc[mask_recal].apply(
        lambda r: r["y_pred_ensemble"] * factor_map.get(r["region"], 1.0), axis=1
    )

    # propager le facteur au niveau commune (pas seulement region) pour les
    # sorties detaillees
    preds["facteur_recalibration"] = np.where(
        preds["annee"].isin([2023, 2024]), preds["region"].map(factor_map).fillna(1.0), 1.0
    )
    preds["y_pred_recalibre"] = preds["y_pred_ensemble"] * preds["facteur_recalibration"]

    out_pred = config.PROCESSED / "ensemble_recalibrated_predictions.csv"
    preds.to_csv(out_pred, index=False)
    logger.info(f"Prédictions (brutes + recalibrées) sauvegardées dans : {out_pred}")

    rows, all_real, all_pred = [], [], []
    logger.info("\n=== VÉRIFICATION RÉELLE CONTRE LES DONNÉES OFFICIELLES 2021/2023/2024 "
                "(2023/2024 recalibrés sur le facteur régional 2021) ===")
    for year, col in YEAR_TO_REGIONAL_COL.items():
        pred_col = "y_pred_ensemble" if year == 2021 else "y_pred_recalibre"
        sub = ref[["region", col]].merge(
            reg_pred[reg_pred["annee"] == year][["region", pred_col]], on="region", how="left"
        )
        sub = sub.rename(columns={pred_col: "y_pred_ensemble"})
        sub["y_pred_ensemble"] = sub["y_pred_ensemble"].fillna(0.0)
        m = compute_metrics(sub[col].values, sub["y_pred_ensemble"].values)
        m["annee"] = year
        m["source_climat"] = "mesure_era5" if year == 2021 else "climatologie_extrapolee"
        rows.append(m)
        all_real.extend(sub[col].tolist())
        all_pred.extend(sub["y_pred_ensemble"].tolist())

        logger.info(f"\n-- {year} ({m['source_climat']}) : MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  R2={m['R2']:.3f} --")
        for _, r in sub.sort_values(col, ascending=False).iterrows():
            logger.info(f"  {r['region']:<28} réel={r[col]:>6.0f}   prédit={r['y_pred_ensemble']:>7.1f}")

    overall = compute_metrics(np.array(all_real), np.array(all_pred))
    logger.info(f"\n=== GLOBAL 2021+2023+2024 (n={overall['n']} region-années) : "
                f"MAE={overall['MAE']:.1f}  RMSE={overall['RMSE']:.1f}  R2={overall['R2']:.3f} ===")

    metrics_df = pd.DataFrame(rows)
    out_metrics = config.PROCESSED / "verification_metrics_2021_2024.csv"
    metrics_df.to_csv(out_metrics, index=False)
    logger.info(f"Métriques écrites : {out_metrics}")
    return metrics_df


if __name__ == "__main__":
    recalibrate_and_evaluate()
