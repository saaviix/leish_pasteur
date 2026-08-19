"""
ensemble_final.py
===================
Combine GBM (LightGBM Tweedie) + PINN SEIR-V (tete d'observation) par
stacking (Ridge non-negatif), ENTIEREMENT dans la fenetre Pasteur 2009-2020
-- pas de 2021-2024.

Les deux modeles sont deja entraines sur 2009-2017 et evalues sur le meme
holdout 2018-2020 (gbm_predictions_2018_2020.csv, pinn_predictions_2018_
2020.csv). Pour apprendre les poids du melange SANS evaluer sur les memes
lignes qui servent a les apprendre (ce qui gonflerait artificiellement le
R2 rapporte), le holdout est lui-meme split chronologiquement :
  - 2018       -> apprentissage des poids de l'ensemble
  - 2019-2020  -> evaluation finale, honnete

Sortie :
  outputs/processed/ensemble_final_predictions.csv
  outputs/processed/ensemble_final_metrics.csv

Usage :
  python src/models/ensemble_final.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    sp = stats.spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Spearman": sp}


def main() -> None:
    config.ensure_dirs()
    gbm = pd.read_csv(config.PROCESSED / "gbm_predictions_2018_2020.csv")
    pinn = pd.read_csv(config.PROCESSED / "pinn_predictions_2018_2020.csv")

    keys = ["commune", "province", "region", "annee", "mois"]
    merged = gbm.merge(pinn[keys + ["y_pred_pinn"]], on=keys, how="inner")
    assert len(merged) == len(gbm), "jointure incomplete entre GBM et PINN -- verifier les cles"
    merged["n_cas"] = merged["n_cas_x"] if "n_cas_x" in merged.columns else merged["n_cas"]

    fit_mask = merged["annee"] == 2018
    eval_mask = merged["annee"].isin([2019, 2020])
    print(f"Apprentissage des poids : {fit_mask.sum()} lignes (2018)")
    print(f"Evaluation finale       : {eval_mask.sum()} lignes (2019-2020)")

    X_fit = merged.loc[fit_mask, ["y_pred_gbm", "y_pred_pinn"]].values
    y_fit = merged.loc[fit_mask, "n_cas"].values
    X_eval = merged.loc[eval_mask, ["y_pred_gbm", "y_pred_pinn"]].values
    y_eval = merged.loc[eval_mask, "n_cas"].values

    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    ridge.fit(X_fit, y_fit)
    w_gbm, w_pinn = ridge.coef_
    print(f"\nPoids appris (2018) : GBM={w_gbm:.4f}  PINN={w_pinn:.4f}  intercept={ridge.intercept_:.4f}")

    y_pred_ens_eval = np.clip(ridge.predict(X_eval), 0, None)
    y_pred_gbm_eval = merged.loc[eval_mask, "y_pred_gbm"].values
    y_pred_pinn_eval = merged.loc[eval_mask, "y_pred_pinn"].values

    print(f"\n{'='*80}\nEVALUATION HONNETE SUR 2019-2020 (pas les lignes utilisees pour apprendre les poids)\n{'='*80}")
    results = {
        "GBM seul": evaluate(y_eval, y_pred_gbm_eval),
        "PINN seul": evaluate(y_eval, y_pred_pinn_eval),
        "Ensemble (Ridge non-negatif)": evaluate(y_eval, y_pred_ens_eval),
    }
    for name, m in results.items():
        print(f"{name:<30} R2={m['R2']:+.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  Spearman={m['Spearman']:.4f}")

    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "modele"})
    metrics_df.to_csv(config.PROCESSED / "ensemble_final_metrics.csv", index=False, encoding="utf-8")

    out = merged.loc[eval_mask, keys + ["n_cas", "y_pred_gbm", "y_pred_pinn"]].copy()
    out["y_pred_ensemble"] = y_pred_ens_eval
    out.to_csv(config.PROCESSED / "ensemble_final_predictions.csv", index=False, encoding="utf-8")
    print(f"\n[OK] outputs/processed/ensemble_final_metrics.csv")
    print(f"[OK] outputs/processed/ensemble_final_predictions.csv")


if __name__ == "__main__":
    main()
