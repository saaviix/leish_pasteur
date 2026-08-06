
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                                ExtraTreesRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_poisson_deviance, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from pygam import PoissonGAM, s, f

df = pd.read_csv("/home/claude/features_finales.csv")
feature_cols = [c for c in df.columns if "lag" in c] + ["mois_sin", "mois_cos"]

train = df[df["annee"] <= 2014].copy()
val   = df[df["annee"].isin([2015, 2017])].copy()
test  = df[df["annee"] == 2020].copy()
print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

X_cols = feature_cols + ["zone_bioclim"]
X_train = pd.get_dummies(train[X_cols], columns=["zone_bioclim"])
X_test = pd.get_dummies(test[X_cols], columns=["zone_bioclim"]).reindex(columns=X_train.columns, fill_value=0)
y_train, y_test = train["cas"].values, test["cas"].values

predictions = {}

# ---------------------------------------------------------------------------
# Famille 1 : GLM
# ---------------------------------------------------------------------------
formula = "cas ~ " + " + ".join(feature_cols) + " + C(zone_bioclim)"

m_poisson = smf.glm(formula=formula, data=train, family=sm.families.Poisson()).fit()
predictions["Poisson GLM"] = m_poisson.predict(test).values

m_nb = smf.glm(formula=formula, data=train, family=sm.families.NegativeBinomial()).fit()
predictions["Negative Binomial GLM"] = m_nb.predict(test).values

# ZINB : partie comptage avec les features climat, partie inflation (zéro structurel) intercept seul
exog_train = sm.add_constant(X_train.astype(float))
exog_test = sm.add_constant(X_test.astype(float))
exog_infl_train = np.ones((len(train), 1))
exog_infl_test = np.ones((len(test), 1))
try:
    m_zinb = ZeroInflatedNegativeBinomialP(y_train, exog_train, exog_infl=exog_infl_train, inflation="logit").fit(maxiter=200, disp=0)
    predictions["Zero-Inflated NB"] = m_zinb.predict(exog_test, exog_infl=exog_infl_test)
except Exception as e:
    print("ZINB a échoué:", e)

# ---------------------------------------------------------------------------
# Famille 2 : GAM (splines sur les variables climat les plus importantes)
# ---------------------------------------------------------------------------
gam_features = ["temp_lag1", "temp_lag2", "temp_lag4", "precip_lag2", "humid_lag2"]
Xg_train = train[gam_features].values
Xg_test = test[gam_features].values
try:
    gam = PoissonGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(Xg_train, y_train)
    predictions["GAM (Poisson, splines)"] = gam.predict(Xg_test)
except Exception as e:
    print("GAM a échoué:", e)

# ---------------------------------------------------------------------------
# Famille 3 : Bagging / arbres
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=0).fit(X_train, y_train)
predictions["Random Forest"] = np.clip(rf.predict(X_test), 0, None)

et = ExtraTreesRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=0).fit(X_train, y_train)
predictions["Extra Trees"] = np.clip(et.predict(X_test), 0, None)

# ---------------------------------------------------------------------------
# Famille 4 : Boosting
# ---------------------------------------------------------------------------
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X_train, y_train)
predictions["Gradient Boosting"] = np.clip(gbm.predict(X_test), 0, None)

xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               objective="count:poisson", random_state=0)
xgb_model.fit(X_train, y_train)
predictions["XGBoost"] = np.clip(xgb_model.predict(X_test), 0, None)

lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                objective="poisson", verbosity=-1, random_state=0)
lgb_model.fit(X_train, y_train)
predictions["LightGBM"] = np.clip(lgb_model.predict(X_test), 0, None)

cat_model = cb.CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05,
                                   loss_function="Poisson", verbose=False, random_state=0)
cat_model.fit(X_train, y_train)
predictions["CatBoost"] = np.clip(cat_model.predict(X_test), 0, None)

# ---------------------------------------------------------------------------
# Métriques enrichies
# ---------------------------------------------------------------------------
def peak_month_error(sub_test_df, pred_array):
    """Erreur moyenne (en mois) sur le mois de pic, par province ayant >=1 cas en test."""
    sub = sub_test_df.copy()
    sub["pred"] = pred_array
    errors = []
    for prov, g in sub.groupby("province"):
        if g["cas"].sum() == 0:
            continue
        obs_peak = g.loc[g["cas"].idxmax(), "mois"]
        pred_peak = g.loc[g["pred"].idxmax(), "mois"]
        errors.append(abs(obs_peak - pred_peak))
    return np.mean(errors) if errors else np.nan

rows = []
for name, pred in predictions.items():
    pred_safe = np.clip(pred, 1e-6, None)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    medae = median_absolute_error(y_test, pred)
    dev = mean_poisson_deviance(y_test, pred_safe)
    rho, _ = spearmanr(y_test, pred)
    r2 = r2_score(y_test, pred)
    peak_err = peak_month_error(test, pred)
    rows.append(dict(modele=name, MAE=mae, RMSE=rmse, MedAE=medae,
                      Deviance_Poisson=dev, Spearman=rho, R2=r2, Erreur_pic_mois=peak_err))

results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
pd.set_option("display.width", 140)
print("\n" + results.round(3).to_string(index=False))
results.to_csv("/home/claude/metrics_comparatif_etendu.csv", index=False)

# ---------------------------------------------------------------------------
# Visualisation : classement multi-métriques
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
metric_specs = [
    ("MAE", "MAE (cas/mois, plus bas = mieux)", False),
    ("RMSE", "RMSE (cas/mois, plus bas = mieux)", False),
    ("MedAE", "Erreur médiane absolue (plus bas = mieux)", False),
    ("Deviance_Poisson", "Déviance de Poisson (plus bas = mieux)", False),
    ("Spearman", "Corrélation de rang Spearman (plus haut = mieux)", True),
    ("Erreur_pic_mois", "Erreur sur le mois de pic (mois, plus bas = mieux)", False),
]
for ax, (col, title, higher_better) in zip(axes.flat, metric_specs):
    sorted_r = results.sort_values(col, ascending=not higher_better)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_r)))
    ax.barh(sorted_r["modele"], sorted_r[col], color=colors)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
plt.suptitle("Comparatif de 10 modèles — test 2020, climat RÉEL ERA5", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/comparatif_10_modeles.png", dpi=130)
print("\nFigure sauvegardée : comparatif_10_modeles.png")
