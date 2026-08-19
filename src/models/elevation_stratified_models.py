"""
elevation_stratified_models.py
================================
Au lieu d'UN modele global (gbm_spatial_temporal.py), entraine un modele
LightGBM (Tweedie) SEPARE par classe d'altitude (elevation_classification.py)
pour voir si les facteurs climat/environnement qui pesent le plus sur le
nombre de cas different selon qu'on est en plaine, plateau, moyenne ou haute
montagne -- complement multivarie (controle simultanement pour toutes les
covariables) aux correlations marginales de elevation_case_drivers.py.

Meme feature set, meme split temporel (train 2009-2017 / test 2018-2020),
meme upweighting des cas positifs que gbm_spatial_temporal.py -- pour rester
directement comparable au modele global. La seule difference : un modele par
classe au lieu d'un modele unique + "province" (feature) devient optionnelle
vu que chaque classe ne contient deja plus qu'un sous-ensemble de provinces.

Entrees :
  outputs/processed/commune_panel.csv
  outputs/processed/province_elevation_classes.csv

Sorties :
  outputs/processed/elevation_stratified_metrics.csv
  outputs/processed/elevation_stratified_importance.csv

Usage :
  python src/models/elevation_stratified_models.py
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

FEATURE_COLS_BASE = [
    "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct",
    "sin_month", "cos_month", "elevation_m", "lai", "aridity_index",
    "psi_mean", "psi_sd", "y_epi", "y_ento_hard", "y_ento_soft",
]


def evaluate_metrics(y_true, y_pred) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        spearman = stats.spearmanr(y_true, y_pred).statistic
    else:
        spearman = 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Spearman": spearman}


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = panel[panel["annee"] <= 2020].copy()
    data = data.sort_values(["commune", "annee", "mois"])

    data["cases_lag1"] = data.groupby("commune")["n_cas"].shift(1)
    data["cases_lag2"] = data.groupby("commune")["n_cas"].shift(2)
    data["cases_lag3"] = data.groupby("commune")["n_cas"].shift(3)
    data["cases_lag12"] = data.groupby("commune")["n_cas"].shift(12)
    data["cases_roll3"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    data["cases_roll6"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())

    data = data.dropna(subset=["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"])

    feature_cols = [c for c in data.columns if "lag" in c or "roll" in c]
    feature_cols += [c for c in FEATURE_COLS_BASE if c in data.columns]
    return data, feature_cols


def train_one_class(data: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[dict, pd.DataFrame, dict]:
    train_mask = (data["annee"] >= 2009) & (data["annee"] <= 2017)
    test_mask = (data["annee"] >= 2018) & (data["annee"] <= 2020)

    use_province = data["province"].nunique() > 1
    cols = feature_cols + (["province"] if use_province else [])

    X_train = data.loc[train_mask, cols].copy()
    X_test = data.loc[test_mask, cols].copy()
    if use_province:
        X_train["province"] = X_train["province"].astype("category")
        X_test["province"] = pd.Categorical(X_test["province"], categories=X_train["province"].cat.categories)
    X_train[feature_cols] = X_train[feature_cols].fillna(0.0)
    X_test[feature_cols] = X_test[feature_cols].fillna(0.0)
    y_train = data.loc[train_mask, "n_cas"].values
    y_test = data.loc[test_mask, "n_cas"].values

    if len(X_train) < 200 or len(X_test) < 50 or y_train.sum() == 0:
        print(f"  [SKIP] {label} : pas assez de donnees (train={len(X_train)}, test={len(X_test)})")
        return {}, pd.DataFrame(), {}

    sample_weight = np.where(y_train > 0, 20.0, 1.0)

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.3, n_estimators=300,
        learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=-1,
        importance_type="gain", verbosity=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = np.clip(model.predict(X_test), 0, None)

    metrics = evaluate_metrics(y_test, y_pred)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["cas_cumules"] = int(data["n_cas"].sum())

    importance = pd.DataFrame({"feature": cols, "importance_gain": model.feature_importances_})
    importance = importance[importance["importance_gain"] > 0].copy()
    importance["importance_pct"] = 100 * importance["importance_gain"] / importance["importance_gain"].sum()
    importance = importance.sort_values("importance_pct", ascending=False)

    return metrics, importance, {"model": model, "feature_cols": cols, "uses_onehot_province": False}


def main() -> None:
    config.ensure_dirs()

    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv")
    classes_path = config.PROCESSED / "province_elevation_classes.csv"
    if not classes_path.exists():
        raise FileNotFoundError(f"{classes_path} introuvable.\nLance d'abord : python src/analysis/elevation_classification.py")
    classes = pd.read_csv(classes_path)[["province", "classe_altitude", "classe_altitude_rank"]]

    panel = panel.merge(classes, on="province", how="left")
    n_missing = panel["classe_altitude"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} lignes sans classe d'altitude (province hors classification) -> exclues")
        panel = panel.dropna(subset=["classe_altitude"])

    data, feature_cols = build_features(panel)

    all_metrics = []
    all_importance = []
    models_by_class = {}
    print(f"\n{'='*90}\nMODELES SEPARES PAR CLASSE D'ALTITUDE (LightGBM Tweedie, meme feature set/split que le modele global)\n{'='*90}")

    for rank, g in sorted(data.groupby("classe_altitude_rank"), key=lambda t: t[0]):
        label = g["classe_altitude"].iloc[0]
        n_prov = g["province"].nunique()
        print(f"\n[{rank}] {label} (n_provinces={n_prov}, n_communes={g['commune'].nunique()})")

        metrics, importance, model_bundle = train_one_class(g, feature_cols, label)
        if not metrics:
            continue
        models_by_class[label] = model_bundle

        print(f"  R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"Spearman={metrics['Spearman']:.4f}  (train={metrics['n_train']}, test={metrics['n_test']})")
        print("  Top facteurs (importance gain, %) :")
        for _, r in importance.head(10).iterrows():
            print(f"    {r['feature']:<18} {r['importance_pct']:5.1f}%")

        metrics_row = {"classe_altitude_rank": rank, "classe_altitude": label, **metrics}
        all_metrics.append(metrics_row)
        importance["classe_altitude_rank"] = rank
        importance["classe_altitude"] = label
        all_importance.append(importance)

    print(f"\n{'='*90}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = config.PROCESSED / "elevation_stratified_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"\n[OK] {metrics_path}")

    importance_df = pd.concat(all_importance, ignore_index=True)
    importance_path = config.PROCESSED / "elevation_stratified_importance.csv"
    importance_df.to_csv(importance_path, index=False, encoding="utf-8")
    print(f"[OK] {importance_path}")

    import joblib
    models_path = config.PROCESSED / "elevation_stratified_models.joblib"
    joblib.dump(models_by_class, models_path)
    print(f"[OK] {models_path} ({len(models_by_class)} modeles, cle = classe_altitude)")


if __name__ == "__main__":
    main()
