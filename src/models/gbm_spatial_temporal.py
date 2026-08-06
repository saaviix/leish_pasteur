

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
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    HistGradientBoostingRegressor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_metrics(y_true, y_pred):
    """Calcule l'ensemble complet des métriques épidémiologiques."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    medae = np.median(np.abs(y_true - y_pred))
    
    eps = 1e-9
    y_true_c = np.clip(y_true, eps, None)
    y_pred_c = np.clip(y_pred, eps, None)
    dev = 2 * np.mean(y_true * np.log(y_true_c / y_pred_c) - (y_true - y_pred))
    
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        spearman = stats.spearmanr(y_true, y_pred).statistic
    else:
        spearman = 0.0
        
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1.0 - (ss_res / (ss_tot + eps))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MedAE": medae,
        "Deviance_Poisson": dev,
        "Spearman": spearman,
        "R2": r2
    }

def train_lgb_model():
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        logger.error(f"Fichier panel introuvable: {panel_path}")
        return None, None
        
    df = pd.read_csv(panel_path)
    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["n_cas"].describe())
    print("\nMost frequent values:")
    print(df["n_cas"].value_counts().head(20))
    print("\nFraction of zeros:")
    zeros = (df["n_cas"] == 0).sum()
    print("Zeros:", zeros)
    print("Total:", len(df))
    print("Fraction:", zeros / len(df))
    print("\nMaximum number of cases:")
    print(df["n_cas"].max())
    print("=========================================\n")
    
    # Filtrer données avec cas connus (2009-2020)
    data = df[df["annee"] <= 2020].copy()
    # -------------------------------------------------
    # Sort chronologically
    # -------------------------------------------------
    data = data.sort_values(["commune", "annee", "mois"])

    # -------------------------------------------------
    # Lag features
    # -------------------------------------------------
    data["cases_lag1"] = (
        data.groupby("commune")["n_cas"].shift(1)
    )

    data["cases_lag2"] = (
        data.groupby("commune")["n_cas"].shift(2)
    )

    data["cases_lag3"] = (
       data.groupby("commune")["n_cas"].shift(3)
   )

    data["cases_lag12"] = (
        data.groupby("commune")["n_cas"].shift(12)
    )

   # -------------------------------------------------
   # Rolling averages
   # -------------------------------------------------
    data["cases_roll3"] = (
       data.groupby("commune")["n_cas"]
         .transform(lambda x: x.shift(1).rolling(3).mean())
    )

    data["cases_roll6"] = (
         data.groupby("commune")["n_cas"]
            .transform(lambda x: x.shift(1).rolling(6).mean())
    )
    print(data.columns.tolist())

    data = data.dropna(subset=["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"])
    
    feature_cols = [
     c
     for c in data.columns
     if (
        "lag" in c
        or "roll" in c
        or c in [
            "latitude",
            "longitude",
            "temp_moy",
            "precip_mm",
            "humidite_pct",
            "sin_month",
            "cos_month",
            "elevation_m",
            "lai",
            "sand_pct",
        ]
        )
    ]
    print(feature_cols)
    print(len(feature_cols))

    logger.info(f"Nombre de covariables utilisées: {len(feature_cols)}")
    
    # Split temporel strict demandé par l'utilisateur (80% train / 20% test)
    train_mask = (data["annee"] >= 2009) & (data["annee"] <= 2017)
    test_mask = (data["annee"] >= 2018) & (data["annee"] <= 2020)
    
    X_train = data.loc[train_mask, feature_cols].fillna(0.0)
    y_train = data.loc[train_mask, "n_cas"].values
    X_test = data.loc[test_mask, feature_cols].fillna(0.0)
    y_test = data.loc[test_mask, "n_cas"].values
    
    logger.info(f"Train samples (2009-2017): {len(X_train)} | Test samples (2018-2020): {len(X_test)}")
    
    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.3,
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        sample_weight = np.where(y_train > 0, 20.0, 1.0)
        model.fit(
                  X_train,
                  y_train,
        sample_weight=sample_weight
        )

        # ==========================================
        # Feature importance
        # ==========================================
        importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\n================ FEATURE IMPORTANCE ================")
        print(importance)
        print("====================================================\n")

        # Predictions
        y_pred = np.clip(model.predict(X_test), 0, None)
        y_pred = np.clip(model.predict(X_test), 0, None)
        print("\n================ PREDICTIONS =================")

        print(pd.Series(y_pred).describe())

        print("\nLargest 20 predictions:")
        print(np.sort(y_pred)[-20:])

        print("===============================================\n")
    elif xgb is not None:
        model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)
    elif HistGradientBoostingRegressor is not None:
        model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)
    else:
        # Poisson Ridge Regression fallback via Scipy
        logger.info("Utilisation du Poisson GLM via scipy.optimize...")
        X_tr = np.hstack([np.ones((len(X_train), 1)), X_train.values])
        X_te = np.hstack([np.ones((len(X_test), 1)), X_test.values])
        
        # Fit ridge linear baseline
        beta = np.linalg.pinv(X_tr.T @ X_tr + 1e-3 * np.eye(X_tr.shape[1])) @ (X_tr.T @ y_train)
        y_pred = np.clip(X_te @ beta, 0, None)
        model = beta
        
    metrics = evaluate_metrics(y_test, y_pred)
    logger.info("=== METRIQUES DU MODELE GBM (Validation 2018-2020) ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    # Sauvegarder les prédictions test
    reg_col = "region" if "region" in data.columns else [c for c in data.columns if "region" in c][0]
    test_df = data.loc[test_mask, ["commune", "province", reg_col, "annee", "mois", "n_cas"]].copy()
    test_df["region"] = test_df[reg_col]
    test_df["y_pred_gbm"] = y_pred
    out_file = config.PROCESSED / "gbm_predictions_2018_2020.csv"
    test_df.to_csv(out_file, index=False)
    logger.info(f"Prédictions test sauvegardées dans : {out_file}")
    
    return model, metrics

if __name__ == "__main__":
    train_lgb_model()
