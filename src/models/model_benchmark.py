"""
model_benchmark.py
===================
Comparatif etendu de familles de modeles de prediction du nombre de cas LCT
(commune x mois), en plus du LightGBM/XGBoost deja couverts par
gbm_spatial_temporal.py :

  - GLM Poisson / Binomiale Negative (statsmodels)
  - Zero-Inflated Negative Binomial -- adapte aux comptages avec beaucoup
    de zeros (cas rares dans la plupart des communes/mois)
  - GAM Poisson a splines (pygam) sur les variables climatiques les plus
    pertinentes
  - Random Forest / Extra Trees / Gradient Boosting (sklearn)
  - CatBoost

Chaque famille depend d'un package optionnel (statsmodels/pygam/catboost) :
si absent, la famille est sautee proprement avec un message expliquant quoi
installer, plutot que de faire echouer tout le comparatif.

Ajoute aussi la metrique "erreur sur le mois de pic" (en mois), utile en
epidemiologie saisonniere et absente de gbm_spatial_temporal.py.

Migre et adapte depuis l'ancien script exploratoire
`main src/comparatif_10_modeles.py` (donnees ad hoc /home/claude/...
remplacees par le panel officiel du pipeline, commune_panel.csv).

Note : les features (lags/rolling) sont volontairement construites de la
meme facon que dans gbm_spatial_temporal.py pour une comparaison equitable ;
cette duplication est un candidat naturel a factoriser en un module
`features.py` partage (Phase 3 de la refonte).

Entrees :
  outputs/processed/commune_panel.csv          (build_commune_panel.py)
  outputs/processed/zone_bioclim_province.csv  (bioclimatic_zoning.py, optionnel)

Sortie :
  outputs/processed/metrics_comparatif_etendu.csv
  outputs/figures/comparatif_modeles.png

Usage :
  python src/models/model_benchmark.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
except ImportError:
    sm = smf = ZeroInflatedNegativeBinomialP = None

try:
    from pygam import PoissonGAM, s
except ImportError:
    PoissonGAM = s = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


LAG_FEATURES = ["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"]
GAM_FEATURES = ["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"]


def build_panel() -> pd.DataFrame:
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} introuvable.\nLance d'abord : python src/data_prep/build_commune_panel.py"
        )
    df = pd.read_csv(panel_path)
    data = df[df["annee"] <= 2020].copy().sort_values(["commune", "annee", "mois"])

    data["cases_lag1"] = data.groupby("commune")["n_cas"].shift(1)
    data["cases_lag2"] = data.groupby("commune")["n_cas"].shift(2)
    data["cases_lag3"] = data.groupby("commune")["n_cas"].shift(3)
    data["cases_lag12"] = data.groupby("commune")["n_cas"].shift(12)
    data["cases_roll3"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    data["cases_roll6"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())

    data = data.dropna(subset=LAG_FEATURES)

    zone_path = config.PROCESSED / "zone_bioclim_province.csv"
    if zone_path.exists() and "province" in data.columns:
        zone = pd.read_csv(zone_path)[["province", "zone_bioclim"]]
        data = data.merge(zone, on="province", how="left")
        data["zone_bioclim"] = data["zone_bioclim"].fillna(-1).astype(int)
    else:
        print(f"[INFO] {zone_path} absent -> pas de covariable zone_bioclim "
              f"(lance bioclimatic_zoning.py pour l'ajouter)")

    return data


def peak_month_error(sub_test_df: pd.DataFrame, pred_array: np.ndarray) -> float:
    """Erreur moyenne (en mois) sur le mois de pic, par commune ayant >=1 cas en test."""
    sub = sub_test_df.copy()
    sub["pred"] = pred_array
    errors = []
    for _, g in sub.groupby("commune"):
        if g["n_cas"].sum() == 0:
            continue
        obs_peak = g.loc[g["n_cas"].idxmax(), "mois"]
        pred_peak = g.loc[g["pred"].idxmax(), "mois"]
        errors.append(abs(obs_peak - pred_peak))
    return float(np.mean(errors)) if errors else np.nan


def run_benchmark() -> pd.DataFrame:
    data = build_panel()

    feature_cols = [
        c for c in data.columns
        if ("lag" in c or "roll" in c) or c in [
            "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct",
            "sin_month", "cos_month", "elevation_m", "lai", "sand_pct",
        ]
    ]
    has_zone = "zone_bioclim" in data.columns

    train_mask = (data["annee"] >= 2009) & (data["annee"] <= 2017)
    test_mask = (data["annee"] >= 2018) & (data["annee"] <= 2020)
    train, test = data.loc[train_mask].copy(), data.loc[test_mask].copy()
    print(f"Train={len(train)}  Test={len(test)}  ({len(feature_cols)} covariables"
          f"{' + zone_bioclim' if has_zone else ''})")

    if has_zone:
        X_train = pd.get_dummies(train[feature_cols + ["zone_bioclim"]], columns=["zone_bioclim"])
        X_test = pd.get_dummies(test[feature_cols + ["zone_bioclim"]], columns=["zone_bioclim"]) \
            .reindex(columns=X_train.columns, fill_value=0)
    else:
        X_train, X_test = train[feature_cols].fillna(0.0), test[feature_cols].fillna(0.0)
    X_train, X_test = X_train.fillna(0.0), X_test.fillna(0.0)
    y_train, y_test = train["n_cas"].values, test["n_cas"].values

    predictions = {}

    # ---- famille 1 : GLM (statsmodels) ----
    if smf is not None:
        formula = "n_cas ~ " + " + ".join(feature_cols) + (" + C(zone_bioclim)" if has_zone else "")
        try:
            m_poisson = smf.glm(formula=formula, data=train, family=sm.families.Poisson()).fit()
            predictions["Poisson GLM"] = m_poisson.predict(test).values
        except Exception as e:
            print(f"[WARN] Poisson GLM a echoue : {e}")
        try:
            m_nb = smf.glm(formula=formula, data=train, family=sm.families.NegativeBinomial()).fit()
            predictions["Negative Binomial GLM"] = m_nb.predict(test).values
        except Exception as e:
            print(f"[WARN] Negative Binomial GLM a echoue : {e}")

        try:
            exog_train = sm.add_constant(X_train.astype(float))
            exog_test = sm.add_constant(X_test.astype(float))
            exog_infl_train = np.ones((len(train), 1))
            exog_infl_test = np.ones((len(test), 1))
            m_zinb = ZeroInflatedNegativeBinomialP(
                y_train, exog_train, exog_infl=exog_infl_train, inflation="logit"
            ).fit(maxiter=200, disp=0)
            predictions["Zero-Inflated NB"] = m_zinb.predict(exog_test, exog_infl=exog_infl_test)
        except Exception as e:
            print(f"[WARN] Zero-Inflated NB a echoue : {e}")
    else:
        print("[INFO] statsmodels non installe -> GLM/ZINB sautes (pip install statsmodels)")

    # ---- famille 2 : GAM (pygam) ----
    if PoissonGAM is not None:
        gam_cols = [c for c in GAM_FEATURES if c in data.columns]
        try:
            Xg_train, Xg_test = train[gam_cols].values, test[gam_cols].values
            terms = s(0)
            for i in range(1, len(gam_cols)):
                terms = terms + s(i)
            gam = PoissonGAM(terms).fit(Xg_train, y_train)
            predictions["GAM (Poisson, splines)"] = gam.predict(Xg_test)
        except Exception as e:
            print(f"[WARN] GAM a echoue : {e}")
    else:
        print("[INFO] pygam non installe -> GAM saute (pip install pygam)")

    # ---- famille 3 : bagging / arbres (sklearn) ----
    rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=0).fit(X_train, y_train)
    predictions["Random Forest"] = np.clip(rf.predict(X_test), 0, None)

    et = ExtraTreesRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=0).fit(X_train, y_train)
    predictions["Extra Trees"] = np.clip(et.predict(X_test), 0, None)

    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X_train, y_train)
    predictions["Gradient Boosting"] = np.clip(gbm.predict(X_test), 0, None)

    # ---- famille 4 : boosting (packages optionnels) ----
    if xgb is not None:
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      objective="count:poisson", random_state=0)
        xgb_model.fit(X_train, y_train)
        predictions["XGBoost"] = np.clip(xgb_model.predict(X_test), 0, None)
    else:
        print("[INFO] xgboost non installe -> saute (pip install xgboost)")

    if lgb is not None:
        lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       objective="poisson", verbosity=-1, random_state=0)
        lgb_model.fit(X_train, y_train)
        predictions["LightGBM"] = np.clip(lgb_model.predict(X_test), 0, None)
    else:
        print("[INFO] lightgbm non installe -> saute (pip install lightgbm)")

    if cb is not None:
        cat_model = cb.CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05,
                                          loss_function="Poisson", verbose=False, random_state=0)
        cat_model.fit(X_train, y_train)
        predictions["CatBoost"] = np.clip(cat_model.predict(X_test), 0, None)
    else:
        print("[INFO] catboost non installe -> saute (pip install catboost)")

    # ---- metriques ----
    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        pred_safe = np.clip(pred, 1e-6, None)
        rows.append(dict(
            modele=name,
            MAE=mean_absolute_error(y_test, pred),
            RMSE=np.sqrt(mean_squared_error(y_test, pred)),
            MedAE=median_absolute_error(y_test, pred),
            Deviance_Poisson=mean_poisson_deviance(y_test, pred_safe),
            Spearman=spearmanr(y_test, pred).statistic,
            R2=r2_score(y_test, pred),
            Erreur_pic_mois=peak_month_error(test, pred),
        ))

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, test, predictions


def main() -> None:
    config.ensure_dirs()
    results, test, predictions = run_benchmark()

    pd.set_option("display.width", 140)
    print("\n" + results.round(3).to_string(index=False))

    out_csv = config.PROCESSED / "metrics_comparatif_etendu.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n[OK] {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        metric_specs = [
            ("MAE", "MAE (cas/mois, plus bas = mieux)", False),
            ("RMSE", "RMSE (cas/mois, plus bas = mieux)", False),
            ("MedAE", "Erreur mediane absolue (plus bas = mieux)", False),
            ("Deviance_Poisson", "Deviance de Poisson (plus bas = mieux)", False),
            ("Spearman", "Correlation de rang Spearman (plus haut = mieux)", True),
            ("Erreur_pic_mois", "Erreur sur le mois de pic (plus bas = mieux)", False),
        ]
        for ax, (col, title, higher_better) in zip(axes.flat, metric_specs):
            sorted_r = results.sort_values(col, ascending=not higher_better)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_r)))
            ax.barh(sorted_r["modele"], sorted_r[col], color=colors)
            ax.set_title(title, fontsize=10)
            ax.invert_yaxis()
        plt.suptitle("Comparatif de modeles -- test 2018-2020, climat ERA5", fontsize=13)
        plt.tight_layout()
        fig_path = config.FIGURES / "comparatif_modeles.png"
        plt.savefig(fig_path, dpi=130)
        print(f"[OK] {fig_path}")
    except ImportError:
        print("[INFO] matplotlib non installe -> figure non generee")


if __name__ == "__main__":
    main()
