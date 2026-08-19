"""
baseline_vs_model.py
======================
Ce que le modele GBM apporte VRAIMENT par rapport a des references naives,
sur plusieurs decoupages temporels (walk-forward), ENTIEREMENT a l'interieur
des donnees Pasteur 2009-2020 -- pas de comparaison contre 2021-2024.

3 references naives, aucune ne regarde le climat :
  - moyenne_globale   : predit la moyenne d'entrainement pour toutes les lignes
  - moyenne_commune   : predit la moyenne historique (train) de CETTE commune
  - persistance_m12   : predit n_cas du meme mois, annee precedente, meme commune

Compare a un LightGBM Tweedie (memes features que gbm_spatial_temporal.py),
sur 3 folds walk-forward qui sautent les annees 2016/2018/2019 (aucun mois de
diagnostic dans la source -> n_cas=NaN, voir build_commune_panel.py) :
  Fold 1 : train 2009-2013 -> test 2014-2015
  Fold 2 : train 2009-2015 -> test 2017
  Fold 3 : train 2009-2017 -> test 2020

Sortie :
  outputs/processed/baseline_vs_model_metrics.csv

Usage :
  python src/models/baseline_vs_model.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config  # noqa: E402

import lightgbm as lgb

FOLDS = [
    {"train": (2009, 2013), "test": (2014, 2015)},
    {"train": (2009, 2015), "test": (2017, 2017)},
    {"train": (2009, 2017), "test": (2020, 2020)},
]

FEATURE_COLS_BASE = [
    "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct",
    "sin_month", "cos_month", "elevation_m", "lai", "aridity_index",
    "psi_mean", "psi_sd", "y_epi", "y_ento_hard", "y_ento_soft",
]


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    spearman = stats.spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Spearman": spearman}


def build_features(panel: pd.DataFrame):
    data = panel[panel["annee"] <= 2020].copy()
    data = data.sort_values(["commune", "annee", "mois"])
    data["cases_lag1"] = data.groupby("commune")["n_cas"].shift(1)
    data["cases_lag12"] = data.groupby("commune")["n_cas"].shift(12)
    data["cases_roll3"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    data["cases_roll6"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())
    data = data.dropna(subset=["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"])
    feature_cols = [c for c in data.columns if "lag" in c or "roll" in c]
    feature_cols += [c for c in FEATURE_COLS_BASE if c in data.columns]
    return data, feature_cols


def run_fold(data: pd.DataFrame, feature_cols: list, fold: dict) -> dict:
    tr_lo, tr_hi = fold["train"]
    te_lo, te_hi = fold["test"]

    has_target = data["n_cas"].notna()
    train_mask = (data["annee"] >= tr_lo) & (data["annee"] <= tr_hi) & has_target
    test_mask = (data["annee"] >= te_lo) & (data["annee"] <= te_hi) & has_target

    train = data.loc[train_mask].copy()
    test = data.loc[test_mask].copy()
    if len(train) < 500 or len(test) < 100:
        return {}

    results = {}

    # ---- baseline 1 : moyenne globale d'entrainement ----
    mean_global = train["n_cas"].mean()
    results["moyenne_globale"] = evaluate(test["n_cas"], np.full(len(test), mean_global))

    # ---- baseline 2 : moyenne historique par commune (fallback = moyenne globale) ----
    mean_by_commune = train.groupby("commune")["n_cas"].mean()
    pred_commune = test["commune"].map(mean_by_commune).fillna(mean_global)
    results["moyenne_commune"] = evaluate(test["n_cas"], pred_commune)

    # ---- baseline 3 : persistance (meme mois, annee-1, meme commune) ----
    hist = data.set_index(["commune", "annee", "mois"])["n_cas"]
    key = list(zip(test["commune"], test["annee"] - 1, test["mois"]))
    pred_persist = pd.Series(key, index=test.index).map(
        lambda k: hist.get(k, np.nan)
    )
    pred_persist = pred_persist.fillna(mean_global)
    results["persistance_m12"] = evaluate(test["n_cas"], pred_persist)

    # ---- modele : LightGBM Tweedie (memes features que gbm_spatial_temporal.py) ----
    non_cat_cols = [c for c in feature_cols if c != "province"]
    fillzero_cols = [c for c in non_cat_cols if not (c.startswith("cases_lag") or c.startswith("cases_roll"))]
    cols = feature_cols + (["province"] if "province" in data.columns else [])

    X_train = train[cols].copy()
    X_test = test[cols].copy()
    if "province" in cols:
        X_train["province"] = X_train["province"].astype("category")
        X_test["province"] = pd.Categorical(X_test["province"], categories=X_train["province"].cat.categories)
    X_train[fillzero_cols] = X_train[fillzero_cols].fillna(0.0)
    X_test[fillzero_cols] = X_test[fillzero_cols].fillna(0.0)
    y_train = train["n_cas"].values

    sample_weight = np.where(y_train > 0, 20.0, 1.0)
    model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=300,
                               learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = np.clip(model.predict(X_test), 0, None)
    results["lightgbm_tweedie"] = evaluate(test["n_cas"], y_pred)

    for k in results:
        results[k]["n_train"] = len(train)
        results[k]["n_test"] = len(test)
    return results


def main():
    config.ensure_dirs()
    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv")
    data, feature_cols = build_features(panel)

    rows = []
    print(f"\n{'='*95}\nBASELINE NAIVE vs LIGHTGBM -- walk-forward, 100% dans 2009-2020\n{'='*95}")
    for i, fold in enumerate(FOLDS, 1):
        print(f"\n[Fold {i}] train {fold['train']} -> test {fold['test']}")
        res = run_fold(data, feature_cols, fold)
        if not res:
            print("  [SKIP] pas assez de donnees")
            continue
        for method, m in res.items():
            print(f"  {method:<20} R2={m['R2']:+.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
                  f"Spearman={m['Spearman']:.4f}")
            rows.append({"fold": i, "train": str(fold["train"]), "test": str(fold["test"]), "method": method, **m})

    out = pd.DataFrame(rows)
    print(f"\n{'='*95}\nMOYENNE SUR LES {len(FOLDS)} FOLDS, PAR METHODE\n{'='*95}")
    summary = out.groupby("method")[["R2", "MAE", "RMSE", "Spearman"]].mean().sort_values("R2", ascending=False)
    print(summary.to_string())

    out_path = config.PROCESSED / "baseline_vs_model_metrics.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
