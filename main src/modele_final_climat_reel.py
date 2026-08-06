"""
Table finale (cas réels x climat RÉEL ERA5, avec lags) + entraînement et comparaison
de 3 modèles de référence : GLM Binomiale Négative, Random Forest, Gradient Boosting.

Niveau : PROVINCE x mois (compromis entre le niveau commune trop sparse et le niveau
région trop grossier -- 50 provinces ont au moins un mois de cas connu).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# 1. Jointure cas x climat RÉEL
# ---------------------------------------------------------------------------
cases = pd.read_csv("/home/claude/panel_province_mois.csv")
cases = cases.rename(columns={"Region": "region", "Province": "province"})
climate = pd.read_csv("/home/claude/climat_reel_province_mois.csv")
zone_epi = pd.read_csv("/home/claude/zone_epi_final.csv")[["province", "zone_bioclim"]]

df = climate.merge(cases, on=["region", "province", "annee", "mois"], how="left")
df["cas"] = df["cas"].fillna(0).astype(int)
df = df.merge(zone_epi, on="province", how="left")
df = df.sort_values(["province", "annee", "mois"]).reset_index(drop=True)

print(f"Table jointe : {df.shape[0]} lignes, {df['province'].nunique()} provinces "
      f"(climat dispo pour toutes, cas réels quand connus)")

# ---------------------------------------------------------------------------
# 2. Lags climatiques (1-6 mois)
# ---------------------------------------------------------------------------
LAGS = [1, 2, 3, 4, 5, 6]

df = df.sort_values(["province", "annee", "mois"]).reset_index(drop=True)
for lag in LAGS:
    df[f"temp_lag{lag}"] = df.groupby("province")["temp_moy"].shift(lag)
    df[f"precip_lag{lag}"] = df.groupby("province")["precip_mm"].shift(lag)
    df[f"humid_lag{lag}"] = df.groupby("province")["humidite_pct"].shift(lag)
df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

feature_cols = [c for c in df.columns if "lag" in c] + ["mois_sin", "mois_cos"]

# On ne garde pour l'ENTRAÎNEMENT que les provinces qui ont EFFECTIVEMENT des cas
# rapportés avec mois connu à un moment donné (les autres n'ont aucun signal pour apprendre)
provinces_avec_cas = cases["province"].unique()
df_model = df[df["province"].isin(provinces_avec_cas)].dropna(subset=feature_cols).copy()

print(f"Après lags, provinces avec signal exploitable : {df_model.shape[0]} lignes, "
      f"{df_model['province'].nunique()} provinces")

df_model.to_csv("/home/claude/features_finales.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Split temporel (rappel : 2016/2018/2019 sans mois connu dans TOUT le fichier source)
# ---------------------------------------------------------------------------
train = df_model[df_model["annee"] <= 2014]
val   = df_model[df_model["annee"].isin([2015, 2017])]
test  = df_model[df_model["annee"] == 2020]
print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

# ---------------------------------------------------------------------------
# 4. Modèle 1 : GLM Binomiale Négative (par province, effet fixe)
# ---------------------------------------------------------------------------
formula = "cas ~ " + " + ".join(feature_cols) + " + C(zone_bioclim)"
nb_model = smf.glm(formula=formula, data=train, family=sm.families.NegativeBinomial()).fit()
pred_nb_val = nb_model.predict(val)
pred_nb_test = nb_model.predict(test)

# ---------------------------------------------------------------------------
# 5. Modèle 2 & 3 : Random Forest et Gradient Boosting
# ---------------------------------------------------------------------------
X_cols = feature_cols + ["zone_bioclim"]
X_train = pd.get_dummies(train[X_cols], columns=["zone_bioclim"])
X_val = pd.get_dummies(val[X_cols], columns=["zone_bioclim"]).reindex(columns=X_train.columns, fill_value=0)
X_test = pd.get_dummies(test[X_cols], columns=["zone_bioclim"]).reindex(columns=X_train.columns, fill_value=0)

rf = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=0)
rf.fit(X_train, train["cas"])
pred_rf_val = np.clip(rf.predict(X_val), 0, None)
pred_rf_test = np.clip(rf.predict(X_test), 0, None)

gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
gbm.fit(X_train, train["cas"])
pred_gbm_val = np.clip(gbm.predict(X_val), 0, None)
pred_gbm_test = np.clip(gbm.predict(X_test), 0, None)

# ---------------------------------------------------------------------------
# 6. Évaluation
# ---------------------------------------------------------------------------
def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(modele=name, MAE=round(mae, 2), RMSE=round(rmse, 2))

results = []
print("\n--- Validation (2015, 2017) ---")
for name, pred in [("GLM Binomiale Négative", pred_nb_val), ("Random Forest", pred_rf_val), ("Gradient Boosting", pred_gbm_val)]:
    r = report(name, val["cas"], pred)
    print(f"{r['modele']:28s} MAE={r['MAE']:6.2f}  RMSE={r['RMSE']:6.2f}")

print("\n--- Test (2020) ---")
for name, pred in [("GLM Binomiale Négative", pred_nb_test), ("Random Forest", pred_rf_test), ("Gradient Boosting", pred_gbm_test)]:
    r = report(name, test["cas"], pred)
    results.append(r)
    print(f"{r['modele']:28s} MAE={r['MAE']:6.2f}  RMSE={r['RMSE']:6.2f}")

pd.DataFrame(results).to_csv("/home/claude/metrics_comparatif.csv", index=False)

print("\nImportance des variables (Gradient Boosting, top 10) :")
importances = pd.Series(gbm.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importances.head(10))
importances.head(15).to_csv("/home/claude/feature_importance_gbm.csv")

# ---------------------------------------------------------------------------
# 7. Graphe comparatif des modèles (barres MAE/RMSE)
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(res_df))
width = 0.35
ax.bar(x - width/2, res_df["MAE"], width, label="MAE")
ax.bar(x + width/2, res_df["RMSE"], width, label="RMSE")
ax.set_xticks(x)
ax.set_xticklabels(res_df["modele"], rotation=10)
ax.set_ylabel("Erreur (cas/mois)")
ax.set_title("Comparaison des modèles — test 2020 (climat RÉEL ERA5)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/comparaison_modeles.png", dpi=130)
print("\nFigure comparaison sauvegardée : /home/claude/comparaison_modeles.png")

# ---------------------------------------------------------------------------
# 8. Observé vs prédit, meilleure province par volume
# ---------------------------------------------------------------------------
top_province = train.groupby("province")["cas"].sum().idxmax()
sub_test = test[test["province"] == top_province].sort_values(["annee", "mois"])
sub_train_full = df_model[df_model["province"] == top_province].sort_values(["annee", "mois"])

sub_X_full = pd.get_dummies(sub_train_full[X_cols], columns=["zone_bioclim"]).reindex(columns=X_train.columns, fill_value=0)
pred_nb_full = nb_model.predict(sub_train_full)
pred_gbm_full = np.clip(gbm.predict(sub_X_full), 0, None)

fig, ax = plt.subplots(figsize=(11, 4.5))
t_axis = sub_train_full["annee"] + (sub_train_full["mois"] - 1) / 12
ax.plot(t_axis, sub_train_full["cas"], "o-", label="Cas observés", color="black", markersize=3)
ax.plot(t_axis, pred_nb_full, "--", label="GLM Binomiale Négative", alpha=0.8)
ax.plot(t_axis, pred_gbm_full, "--", label="Gradient Boosting", alpha=0.8)
ax.axvline(2015, color="gray", ls=":", lw=1)
ax.axvline(2020, color="gray", ls=":", lw=1)
ax.set_title(f"Observé vs prédit — {top_province} (climat RÉEL, train<2015 | val 2015,2017 | test 2020)")
ax.set_xlabel("Année")
ax.set_ylabel("Cas / mois")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/observe_vs_predit_reel.png", dpi=130)
print("Figure observé/prédit sauvegardée : /home/claude/observe_vs_predit_reel.png")
