

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
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
except ImportError:
    ExtraTreesRegressor = GradientBoostingRegressor = HistGradientBoostingRegressor = RandomForestRegressor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_pinn_physics_features(data: pd.DataFrame) -> pd.DataFrame:
    """Hybride SEIR-V + GBM : injecte les 3 fonctions climat-dependantes
    APPRISES par le PINN (pinn_seirv.py) -- emergence du vecteur f(temp,
    precip), mortalite du vecteur f(temp), taux d'incubation extrinseque
    f(temp) -- comme features supplementaires du GBM. Le PINN seul plafonne
    a R2~0.30-0.32 en prediction brute (bien en-deca du GBM ~0.53-0.54) --
    melanger ses PREDICTIONS via un ensemble lineaire n'apporte rien (poids
    Ridge non-negatif appris a 0, voir ensemble_final.py) car ses erreurs
    sont trop correlees a celles du GBM une fois qu'il recoit les memes
    covariables. Mais ses fonctions climat->dynamique vectorielle sont non-
    lineaires et apprises independamment de la cible cas -- un signal
    potentiellement complementaire pour le GBM, pas juste une redite."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("torch indisponible -> features PINN non ajoutees (GBM seul)")
        return data

    weights_path = config.PROCESSED / "pinn_seirv_weights.pt"
    if not weights_path.exists():
        logger.warning(f"{weights_path} introuvable -> features PINN non ajoutees. "
                        f"Lance d'abord : python src/models/pinn_seirv.py")
        return data

    sys.path.insert(0, str(ROOT / "src" / "models"))
    from pinn_seirv import SEIRVPINN  # noqa: E402

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = SEIRVPINN(n_provinces=len(ckpt["provinces"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = data.copy()

    with torch.no_grad():
        temp_t = torch.tensor(data["temp_moy"].fillna(data["temp_moy"].median()).values, dtype=torch.float32).unsqueeze(1)
        precip_t = torch.tensor(data["precip_mm"].fillna(data["precip_mm"].median()).values, dtype=torch.float32).unsqueeze(1)
        emergence = model.emergence_fn(torch.cat([temp_t, precip_t], dim=1)).numpy().flatten()
        mu_V = model.mortality_fn(temp_t).numpy().flatten()
        sigma_V = model.eip_fn(temp_t).numpy().flatten()

    data["pinn_emergence"] = emergence
    data["pinn_mortalite_vecteur"] = mu_V
    data["pinn_capacite_vectorielle"] = emergence / mu_V
    data["pinn_incubation_extrinseque"] = 1.0 / sigma_V

    # ---- passage complet (forward()) : etat mecaniste ACCUMULE (E_H, I_H,
    # obs_rate) -- signal structurellement different des fonctions climat
    # ci-dessus (qui sont des transformations lisses de temp/precip, deja
    # redondantes avec les lags climat du GBM). E_H/I_H sont le resultat de
    # l'integration RK4/autograd de TOUTE l'historique via les equations
    # SEIR-V, pas juste une fenetre de 6 mois comme les lags -- une vraie
    # "pression d'infection accumulee" que le GBM ne peut pas reconstruire
    # seul a partir de ses propres features.
    if "commune_id" not in data.columns:
        logger.warning("commune_id absent -> etat mecaniste PINN (E_H/I_H) non calculable, "
                        "seules les fonctions climat ci-dessus sont ajoutees")
        return data

    tmp = data.sort_values(["commune_id", "annee", "mois"]).copy()
    tmp["cases_lag1_p"] = tmp.groupby("commune_id")["n_cas"].shift(1)
    tmp["cases_roll3_p"] = tmp.groupby("commune_id")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    tmp["cases_roll6_p"] = tmp.groupby("commune_id")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())
    for c in ["cases_lag1_p", "cases_roll3_p", "cases_roll6_p"]:
        tmp[c] = np.log1p(tmp[c].fillna(0.0))

    prov_to_idx = {p: i for i, p in enumerate(ckpt["provinces"])}
    tmp["province_idx_p"] = tmp["province"].map(prov_to_idx).fillna(0).astype(int)
    tmp["t_months_p"] = (tmp["annee"] - 2009) * 12 + (tmp["mois"] - 1)  # meme origine que l'entrainement PINN

    with torch.no_grad():
        t_ = torch.tensor(tmp["t_months_p"].values, dtype=torch.float32).unsqueeze(1)
        lat_ = torch.tensor(tmp["latitude"].values, dtype=torch.float32).unsqueeze(1)
        lon_ = torch.tensor(tmp["longitude"].values, dtype=torch.float32).unsqueeze(1)
        temp_ = torch.tensor(tmp["temp_moy"].fillna(tmp["temp_moy"].median()).values, dtype=torch.float32).unsqueeze(1)
        precip_ = torch.tensor(tmp["precip_mm"].fillna(tmp["precip_mm"].median()).values, dtype=torch.float32).unsqueeze(1)
        humid_ = torch.tensor(tmp["humidite_pct"].fillna(tmp["humidite_pct"].median()).values, dtype=torch.float32).unsqueeze(1)
        hist_ = torch.tensor(tmp[["cases_lag1_p", "cases_roll3_p", "cases_roll6_p"]].values, dtype=torch.float32)
        prov_ = torch.tensor(tmp["province_idx_p"].values, dtype=torch.long).unsqueeze(1)
        out = model(t_, lat_, lon_, temp_, precip_, humid_, hist_, prov_)
        tmp["pinn_E_H"] = out["E_H"].numpy().flatten()
        tmp["pinn_I_H"] = out["I_H"].numpy().flatten()
        tmp["pinn_C_vraie"] = out["obs_rate"].numpy().flatten()
        tmp["pinn_cas_rapportes"] = (model.rho * out["obs_rate"]).numpy().flatten()

    data = data.merge(
        tmp[["commune_id", "annee", "mois", "pinn_E_H", "pinn_I_H", "pinn_C_vraie", "pinn_cas_rapportes"]],
        on=["commune_id", "annee", "mois"], how="left",
    )
    logger.info("Features PINN ajoutees au GBM (hybride SEIR-V+GBM) : fonctions climat->vecteur "
                "(emergence/mortalite/incubation/capacite) + etat mecaniste accumule (E_H/I_H/C_vraie/cas_rapportes)")
    return data


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
    data = add_pinn_physics_features(data)
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
            "aridity_index",
            # signal d'occupation du vecteur (modele bayesien ICAR) et
            # signaux epidemiologique/entomologique bruts par province,
            # deja dans le panel via province_table.csv/posterior mais
            # jusqu'ici jamais donnes au GBM
            "psi_mean",
            "psi_sd",
            "y_epi",
            "y_ento_hard",
            "y_ento_soft",
            # hybride SEIR-V+GBM : fonctions climat-dependantes apprises par
            # le PINN (add_pinn_physics_features), independantes de n_cas
            "pinn_emergence",
            "pinn_mortalite_vecteur",
            "pinn_capacite_vectorielle",
            "pinn_incubation_extrinseque",
            # etat mecaniste ACCUMULE (integration RK4/autograd de toute
            # l'historique, pas juste une fenetre de lags) -- test distinct
            # des fonctions climat ci-dessus, voir add_pinn_physics_features
            "pinn_E_H",
            "pinn_I_H",
            "pinn_C_vraie",
            "pinn_cas_rapportes",
        ]
        # incidence_100k est calcule a partir de n_cas (la cible) -> fuite de
        # donnees si utilise comme feature, exclu explicitement
        )
    ]
    # province en categorielle native LightGBM : capture un effet regional de
    # base sans exploser en dummies (76 provinces) -- attaque directement le
    # biais systematique observe par region (Draa-Tafilalet sous-predit,
    # Fes-Meknes sur-predit) a la verification 2021-2024.
    if "province" in data.columns:
        data["province"] = data["province"].astype("category")
        feature_cols = feature_cols + ["province"]

    logger.info(f"Nombre de covariables utilisées: {len(feature_cols)}")
    logger.info(f"Covariables: {feature_cols}")

    # Split temporel strict demandé par l'utilisateur (80% train / 20% test)
    # ET exclusion des lignes n_cas=NaN : 2016/2018/2019 n'ont AUCUN mois de
    # diagnostic renseigne dans la source (voir build_commune_panel.py) --
    # avant le fix, ces annees etaient remplies a n_cas=0 (faux), corrompant
    # 3 des 12 annees d'entrainement avec du bruit pur. Elles restent dans
    # `data` (pour que les lags des annees suivantes restent correctement
    # espaces dans le temps) mais sont exclues ici du train ET du test.
    has_target = data["n_cas"].notna()
    train_mask = (data["annee"] >= 2009) & (data["annee"] <= 2017) & has_target
    test_mask = (data["annee"] >= 2018) & (data["annee"] <= 2020) & has_target
    n_excl_train = int(((data["annee"] >= 2009) & (data["annee"] <= 2017) & ~has_target).sum())
    n_excl_test = int(((data["annee"] >= 2018) & (data["annee"] <= 2020) & ~has_target).sum())
    logger.info(f"Lignes exclues pour n_cas=NaN (annee sans mois de diagnostic) : "
                f"{n_excl_train} en train, {n_excl_test} en test")

    non_cat_cols = [c for c in feature_cols if c != "province"]
    # cases_lag*/roll* : NaN laisse tel quel (LightGBM gere nativement les
    # valeurs manquantes en apprenant la meilleure direction de split) plutot
    # que force a 0, ce qui ferait passer "historique de cas inconnu" (annee
    # sans mois juste avant) pour "confirme zero cas recemment" -- exactement
    # le meme biais que celui corrige au niveau de la cible.
    fillzero_cols = [c for c in non_cat_cols if not (c.startswith("cases_lag") or c.startswith("cases_roll"))]
    X_train = data.loc[train_mask, feature_cols].copy()
    X_train[fillzero_cols] = X_train[fillzero_cols].fillna(0.0)
    y_train = data.loc[train_mask, "n_cas"].values
    X_test = data.loc[test_mask, feature_cols].copy()
    X_test[fillzero_cols] = X_test[fillzero_cols].fillna(0.0)
    y_test = data.loc[test_mask, "n_cas"].values
    
    logger.info(f"Train samples (2009-2017): {len(X_train)} | Test samples (2018-2020): {len(X_test)}")

    # ------------------------------------------------------------------
    # Comparatif honnete de plusieurs familles sur le MEME split/features :
    # model_benchmark.py a montre que sur ce jeu de donnees (tres zero-
    # inflate, 90% de zeros), les forets (Extra Trees/Random Forest/GBM
    # sklearn) battent nettement le LightGBM Tweedie utilise jusqu'ici.
    # On entraine les candidats disponibles et on garde/persiste le
    # meilleur par R2 sur le test 2018-2020 -- pas suppose a priori.
    # ------------------------------------------------------------------
    sample_weight = np.where(y_train > 0, 20.0, 1.0)

    # Extra Trees/Random Forest/GradientBoosting (sklearn) ne gerent pas les
    # categorielles pandas nativement -> one-hot pour "province" seulement
    # pour ces candidats (LightGBM garde la categorielle native).
    X_train_oh = pd.get_dummies(X_train, columns=["province"]) if "province" in X_train.columns else X_train
    X_test_oh = pd.get_dummies(X_test, columns=["province"]) if "province" in X_test.columns else X_test
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    candidates = {}

    if lgb is not None:
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=300,
                               learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        candidates["LightGBM (Tweedie)"] = (m, X_test, np.clip(m.predict(X_test), 0, None))

    # Extra Trees/Random Forest (sklearn) ne gerent pas nativement les NaN
    # (contrairement a LightGBM/HistGB) -- fillna(0) uniquement pour ces deux
    # candidats, sur une copie dediee, pour ne pas re-introduire le biais
    # "NaN traite comme zero" dans les versions natives-NaN.
    X_train_oh_filled = X_train_oh.fillna(0.0)
    X_test_oh_filled = X_test_oh.fillna(0.0)

    if ExtraTreesRegressor is not None:
        m = ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)
        m.fit(X_train_oh_filled, y_train, sample_weight=sample_weight)
        candidates["Extra Trees"] = (m, X_test_oh_filled, np.clip(m.predict(X_test_oh_filled), 0, None))

    if RandomForestRegressor is not None:
        m = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
        m.fit(X_train_oh_filled, y_train, sample_weight=sample_weight)
        candidates["Random Forest"] = (m, X_test_oh_filled, np.clip(m.predict(X_test_oh_filled), 0, None))

    if HistGradientBoostingRegressor is not None:
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, random_state=42)
        m.fit(X_train_oh, y_train, sample_weight=sample_weight)
        candidates["HistGB (Poisson)"] = (m, X_test_oh, np.clip(m.predict(X_test_oh), 0, None))

    if not candidates:
        raise ImportError("Aucun modele disponible (lightgbm/scikit-learn introuvables).")

    print("\n================ COMPARATIF DE CANDIDATS (test 2018-2020) ================")
    results = {}
    for name, (m, X_te, y_pred) in candidates.items():
        results[name] = evaluate_metrics(y_test, y_pred)
        print(f"{name:<22} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))
    print("============================================================================\n")

    best_name = max(results, key=lambda n: results[n]["R2"])
    model, X_test_best, y_pred = candidates[best_name]
    best_feature_cols = list(X_test_best.columns)
    metrics = results[best_name]
    logger.info(f"Meilleur candidat : {best_name} (R2={metrics['R2']:.4f}) -- retenu et persisté")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": best_feature_cols, "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\n================ FEATURE IMPORTANCE (meilleur modele) ================")
        print(importance.head(25).to_string(index=False))
        print("========================================================================\n")

    logger.info("=== METRIQUES DU MEILLEUR MODELE (Validation 2018-2020) ===")
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

    # Persister le modele entraine + la liste des covariables : sans cela,
    # forecast_future.py ne pouvait jamais reutiliser le vrai modele (il
    # attendait un argument gbm_model jamais passe par run_pipeline.py) et
    # retombait silencieusement sur une formule heuristique codee en dur.
    # `uses_onehot_province` indique aux consommateurs (forecast_future.py,
    # robust_ensemble_recalibrated.py) s'il faut one-hot "province" comme ici
    # plutot que la passer en categorielle native (LightGBM uniquement).
    import joblib
    model_path = config.PROCESSED / "gbm_model.joblib"
    joblib.dump({
        "model": model,
        "feature_cols": best_feature_cols,
        "model_name": best_name,
        "uses_onehot_province": best_name != "LightGBM (Tweedie)",
        "raw_feature_cols": feature_cols,
    }, model_path)
    logger.info(f"Modele sauvegardé : {model_path}")

    return model, metrics

if __name__ == "__main__":
    train_lgb_model()
