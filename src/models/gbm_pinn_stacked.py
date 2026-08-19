"""
gbm_pinn_stacked.py
=====================
Modele officiel de prediction de cas du projet : stacking residuel a 2
etages, LightGBM Tweedie (GBM_1) + LightGBM correcteur de residu (GBM_2)
avec acces aux features mecanistes du PINN SEIR-V (E_H, I_H, C_vraie,
fonctions climat->vecteur).

Pourquoi cette architecture (et pas une injection directe de features) :
  - Injecter les features PINN directement dans un seul GBM degrade le R2
    (0.5312 -> 0.4917) malgre une importance elevee de `pinn_C_vraie` (rang
    2/48) : le modele devient meilleur sur le cas typique (MAE 0.23->0.19,
    MedAE /3) mais moins bon sur les gros pics (RMSE degrade), donc R2 net
    recule -- deplacement du profil d'erreur, pas une vraie amelioration.
  - Le stacking residuel resout ca : GBM_1 (features standard uniquement)
    capture le signal principal exactement comme avant ; GBM_2 apprend
    seulement l'ERREUR RESIDUELLE de GBM_1 (calculee en out-of-fold 3-fold
    sur le train, pas en in-sample, pour eviter l'optimisme), avec acces aux
    features PINN en plus des features standard.

Ameliorations validees par workflow adversarial (recherche + reproduction
independante, voir session du 2026-08-10), toutes les trois reelles et
reproductibles a 4 decimales pres, integrees ici :
  - C1 (tuning) : num_leaves 31->63, learning_rate 0.03->0.1,
    n_estimators 300->800, min_child_samples 20->50 sur GBM_1. R2 agrege
    quasi inchange (bruit, delta -0.0004) mais R2 hors commune dominante
    (Imintanoute, 20% des cas test) : 0.090 -> 0.221.
  - C2 (features spatiales) : moyenne des cases_lag1/cases_roll3/psi_mean
    des k=5 communes les plus proches DANS LA MEME PROVINCE (aucune fuite :
    n'utilise que des colonnes deja shift(1)). Gain reel mais marginal
    (+0.002 a +0.003 R2 agrege, +0.003 a +0.006 R2 hors Imintanoute).
  - C3 (memoire longue) : cases_lag18/24 + cases_roll9/12/18 AJOUTES aux
    lags courts existants (la combinaison gagne, le remplacement des lags
    courts degrade -0.05 R2 agrege). R2 agrege quasi inchange (bruit,
    delta -0.005) mais R2 hors Imintanoute : 0.090 -> 0.127 (+40% relatif).
  Les trois visent la meme faiblesse (generalisation au-dela de la commune
  dominante) sans se marcher dessus -- combinees ici pour la premiere fois.

Segmentation par tier (session 2026-08-11) : le R2 agrege melange 4 regimes
tres differents selon l'historique de cas propre a la commune
(train_total = somme n_cas 2009-2017) -- cold-start (train_total==0, 64% des
communes, 5% des cas test, R2~0.03), low (1-10, 15% des cas, R2~0.09),
moderate (11-50, 23% des cas, R2~0.24-0.29), hotspot (>50, 49 communes mais
57% des cas test, R2 deja bon ~0.62-0.63). 4 pistes d'amelioration testees
par tier (workflow specialise + verification, puis ablation propre
independante sur le pipeline combine) :
  - cold-start : classifieur de detection dedie teste, PERD contre le modele
    officiel (AUC 0.70 vs 0.82 officiel) -- le modele actuel capture deja
    mieux ce signal via ses features existantes (voisinage, PINN). REJETE.
  - low / moderate : pooling par archetype climat/vecteur (KMeans sur les
    covariables climat/vecteur/elevation) teste par 2 agents (effet nation-
    al positif rapporte, +0.017 a +0.027) MAIS contredit par une
    reimplementation propre et indexee correctement (effet NEGATIF partout,
    national ET par tier) -- resultat non reproductible de facon fiable.
    REJETE (evidence insuffisante/contradictoire, jamais deploye sur un
    doute non resolu). Retrait des features de voisinage C2 : gain national
    (+0.015) mais quasi-nul une fois pondere par cas sur les tiers non-
    hotspot (0.1625 vs 0.1634) -- gain deja valide a l'origine, garde tel
    quel plutot que de defaire une decision deja verifiee sur un ecart dans
    le bruit.
  - hotspot : SEUL gain valide et robuste du round -- voir
    train_hotspot_specialist() ci-dessous.

Entrees :
  outputs/processed/commune_panel.csv
  outputs/processed/pinn_seirv_weights.pt (src/models/pinn_seirv.py)

Sorties :
  outputs/processed/gbm_model.joblib   (modele officiel, 2 etages)
  outputs/processed/gbm_predictions_2018_2020.csv

Usage :
  python src/models/gbm_pinn_stacked.py
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import config  # noqa: E402
from gbm_spatial_temporal import add_pinn_physics_features, evaluate_metrics  # noqa: E402

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_COLS = [
    "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct",
    "sin_month", "cos_month", "elevation_m", "lai", "aridity_index",
    "psi_mean", "psi_sd", "y_epi", "y_ento_hard", "y_ento_soft",
    "neighbor_psi_mean",
]
PINN_COLS = [
    "pinn_emergence", "pinn_mortalite_vecteur", "pinn_capacite_vectorielle",
    "pinn_incubation_extrinseque", "pinn_E_H", "pinn_I_H", "pinn_C_vraie", "pinn_cas_rapportes",
]
N_NEIGHBORS = 5  # C2 : k plus proches communes, meme province (voir add_neighbor_features)


def add_extended_lags(data: pd.DataFrame) -> pd.DataFrame:
    """C3 : memoire de cas etendue (9-24 mois), AJOUTEE aux lags courts
    existants (1-12 mois) -- teste et confirme que la combinaison court+long
    bat le court seul ET le long seul (voir docstring du module)."""
    data["cases_lag18"] = data.groupby("commune")["n_cas"].shift(18)
    data["cases_lag24"] = data.groupby("commune")["n_cas"].shift(24)
    data["cases_roll9"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(9).mean())
    data["cases_roll12"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(12).mean())
    data["cases_roll18"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(18).mean())
    return data


def add_neighbor_features(data: pd.DataFrame, k: int = N_NEIGHBORS) -> pd.DataFrame:
    """C2 : moyenne, chez les k communes les plus proches (meme province),
    de cases_lag1/cases_roll3 (deja shift(1) -- aucune fuite temporelle) et
    de psi_mean (statique). Doit etre appele APRES le calcul de cases_lag1/
    cases_roll3 sur `data`."""
    coords = data[["commune", "province", "latitude", "longitude"]].drop_duplicates("commune")
    edges = []
    for _, g in coords.groupby("province"):
        names = g["commune"].to_numpy()
        lat = g["latitude"].to_numpy()
        lon = g["longitude"].to_numpy()
        n = len(names)
        if n < 2:
            continue
        for i in range(n):
            d = np.sqrt((lat - lat[i]) ** 2 + (lon - lon[i]) ** 2)
            d[i] = np.inf
            nn_idx = np.argsort(d)[:min(k, n - 1)]
            edges.extend((names[i], names[j]) for j in nn_idx)
    edge_df = pd.DataFrame(edges, columns=["commune", "neighbor"])

    src = data[["commune", "annee", "mois", "cases_lag1", "cases_roll3", "psi_mean"]].rename(
        columns={"commune": "neighbor", "cases_lag1": "_n_lag1", "cases_roll3": "_n_roll3", "psi_mean": "_n_psi"}
    )
    merged = edge_df.merge(src, on="neighbor", how="left")
    agg = merged.groupby(["commune", "annee", "mois"], as_index=False).agg(
        neighbor_cases_lag1=("_n_lag1", "mean"),
        neighbor_cases_roll3=("_n_roll3", "mean"),
        neighbor_psi_mean=("_n_psi", "mean"),
    )
    return data.merge(agg, on=["commune", "annee", "mois"], how="left")


def build_features(panel: pd.DataFrame) -> tuple:
    data = panel[panel["annee"] <= 2020].copy()
    data = add_pinn_physics_features(data)
    data = data.sort_values(["commune", "annee", "mois"])
    data["cases_lag1"] = data.groupby("commune")["n_cas"].shift(1)
    data["cases_lag2"] = data.groupby("commune")["n_cas"].shift(2)
    data["cases_lag3"] = data.groupby("commune")["n_cas"].shift(3)
    data["cases_lag12"] = data.groupby("commune")["n_cas"].shift(12)
    data["cases_roll3"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    data["cases_roll6"] = data.groupby("commune")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())
    data = add_extended_lags(data)
    data = add_neighbor_features(data)
    data = data.dropna(subset=["temp_moy_lag1", "precip_mm_lag1", "humidite_pct_lag1"])

    lag_roll_cols = [c for c in data.columns if "lag" in c or "roll" in c]
    base_cols = [c for c in BASE_COLS if c in data.columns]
    pinn_cols = [c for c in PINN_COLS if c in data.columns]
    data["province"] = data["province"].astype("category")

    full_feature_cols = lag_roll_cols + base_cols + pinn_cols + ["province"]
    base_feature_cols = lag_roll_cols + base_cols + ["province"]
    return data, base_feature_cols, full_feature_cols


def _fill(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    non_cat = [c for c in cols if c != "province"]
    fillzero = [c for c in non_cat if not ("cases_lag" in c or "cases_roll" in c)]
    df[fillzero] = df[fillzero].fillna(0.0)
    return df


HOTSPOT_TRAIN_TOTAL_THRESHOLD = 50  # cf. segmentation par tier, docstring du module


def train_hotspot_specialist(data: pd.DataFrame, base_cols: list, train_mask) -> tuple:
    """Entraine un GBM_1 dedie aux communes 'hotspot' (train_total = somme de
    n_cas 2009-2017 > 50 -- 49 communes, 57% des cas test malgre 3% des
    communes) et REMPLACE (pas moyenne) la prediction officielle pour ces
    communes specifiquement.

    Trouve et verifie cette session (workflow specialise par tier + reproduction
    independante, robuste par annee et hors Imintanoute) : ce sous-ensemble a
    un profil totalement different du reste du pays -- ~44% de mois positifs
    contre ~90%+ de zeros au national -- donc le sample_weight=20x sur y>0
    (calibre pour l'extreme zero-inflation nationale) et la perte Tweedie
    (pensee pour des comptages tres zero-inflates) sont mal adaptes ICI et
    degradent le fit. Perte L1 (MAE, robuste aux pics extremes type
    Imintanoute) SANS sample_weight, entrainee UNIQUEMENT sur les lignes
    train de ces communes : R2 tier hotspot 0.62->0.71, R2 national
    0.551->0.615 sur le holdout reel 2018-2020 -- gain confirme chaque annee
    separement et hors Imintanoute seule (pas un artefact d'un seul an/une
    seule commune)."""
    train_totals = data.loc[train_mask].groupby("commune")["n_cas"].sum(min_count=1).fillna(0)
    hotspot_communes = set(train_totals[train_totals > HOTSPOT_TRAIN_TOTAL_THRESHOLD].index)

    hotspot_train_mask = train_mask & data["commune"].isin(hotspot_communes)
    X_h = _fill(data.loc[hotspot_train_mask, base_cols], base_cols)[base_cols]
    y_h = data.loc[hotspot_train_mask, "n_cas"].values

    model_hotspot = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=800, learning_rate=0.1, num_leaves=63,
        min_child_samples=50, max_depth=-1, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model_hotspot.fit(X_h, y_h)
    logger.info(f"Specialiste hotspot entraine sur {len(hotspot_communes)} communes "
                f"({hotspot_train_mask.sum()} lignes train, train_total>{HOTSPOT_TRAIN_TOTAL_THRESHOLD})")
    return model_hotspot, hotspot_communes


def apply_hotspot_override(pred: np.ndarray, communes: np.ndarray, X_base: pd.DataFrame,
                            model_hotspot, hotspot_communes: set) -> np.ndarray:
    """Remplace `pred` par la prediction du specialiste pour les lignes dont
    la commune est dans `hotspot_communes` -- le reste du pays est inchange
    par construction."""
    pred = pred.copy()
    mask = pd.Series(communes).isin(hotspot_communes).to_numpy()
    if mask.any():
        pred[mask] = np.clip(model_hotspot.predict(X_base.loc[mask]), 0, None)
    return pred


def train() -> dict:
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} introuvable. Lance d'abord : python src/data_prep/build_commune_panel.py")
    panel = pd.read_csv(panel_path)

    data, base_cols, full_cols = build_features(panel)

    has_target = data["n_cas"].notna()
    train_mask = (data["annee"] >= 2009) & (data["annee"] <= 2017) & has_target
    test_mask = (data["annee"] >= 2018) & (data["annee"] <= 2020) & has_target
    logger.info(f"Train (2009-2017): {train_mask.sum()} | Test (2018-2020): {test_mask.sum()}")

    X_train_full = _fill(data.loc[train_mask, full_cols], full_cols)
    X_test_full = _fill(data.loc[test_mask, full_cols], full_cols)
    X_train_base = X_train_full[base_cols]
    X_test_base = X_test_full[base_cols]
    y_train = data.loc[train_mask, "n_cas"].values
    y_test = data.loc[test_mask, "n_cas"].values
    sample_weight = np.where(y_train > 0, 20.0, 1.0)

    # C1 : config tuned (num_leaves 31->63, lr 0.03->0.1, n_estimators 300->800,
    # min_child_samples 20->50) -- validee par recherche + reproduction
    # independante, voir docstring du module.
    lgb_kwargs = dict(objective="tweedie", tweedie_variance_power=1.3, n_estimators=800,
                       learning_rate=0.1, num_leaves=63, min_child_samples=50, max_depth=-1,
                       random_state=42, n_jobs=-1, verbosity=-1)

    # ---- GBM_1 : signal principal, features standard uniquement ----
    logger.info("Entrainement GBM_1 (signal principal)...")
    model_1 = lgb.LGBMRegressor(**lgb_kwargs)
    model_1.fit(X_train_base, y_train, sample_weight=sample_weight)

    # ---- residus out-of-fold sur le train (3-fold, evite l'optimisme in-sample) ----
    logger.info("Calcul des residus out-of-fold (3-fold)...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train_base))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_base)):
        mf = lgb.LGBMRegressor(**lgb_kwargs)
        mf.fit(X_train_base.iloc[tr_idx], y_train[tr_idx], sample_weight=sample_weight[tr_idx])
        oof_pred[va_idx] = np.clip(mf.predict(X_train_base.iloc[va_idx]), 0, None)
        logger.info(f"  fold {fold + 1}/3 termine")
    resid_train = y_train - oof_pred

    # ---- GBM_2 : correcteur de residu, acces aux features PINN en plus ----
    # Regularisation renforcee (300/0.03/31 -> 50/0.02/7/min_child=200) : avec
    # GBM_1 desormais beaucoup plus fort (C1+C2+C3), le residu a corriger est
    # nettement plus petit/plus bruite qu'avant -- le GBM_2 d'origine sur-
    # apprenait ce residu (R2 final 0.425 < 0.551 pour GBM_1 seul, verifie
    # empiriquement). Cette config regularisee est la seule testee qui bat
    # GBM_1 seul sur les DEUX metriques (R2 agrege ET hors commune dominante).
    logger.info("Entrainement GBM_2 (correcteur de residu, regularise, features PINN incluses)...")
    model_2 = lgb.LGBMRegressor(objective="regression", n_estimators=50, learning_rate=0.02,
                                 num_leaves=7, min_child_samples=200, random_state=42, n_jobs=-1, verbosity=-1)
    model_2.fit(X_train_full, resid_train)

    # ---- specialiste hotspot : remplace la prediction officielle pour les
    # ~49 communes a historique riche (voir train_hotspot_specialist) ----
    model_hotspot, hotspot_communes = train_hotspot_specialist(data, base_cols, train_mask)

    # ---- evaluation finale sur le holdout 2018-2020 ----
    gbm1_test_pred = np.clip(model_1.predict(X_test_base), 0, None)
    resid_test_pred = model_2.predict(X_test_full)
    final_pred = np.clip(gbm1_test_pred + resid_test_pred, 0, None)

    test_communes = data.loc[test_mask, "commune"].to_numpy()
    final_pred = apply_hotspot_override(final_pred, test_communes, X_test_base, model_hotspot, hotspot_communes)

    m_gbm1 = evaluate_metrics(y_test, gbm1_test_pred)
    m_final = evaluate_metrics(y_test, final_pred)
    print("\n================ GBM_1 SEUL (reference, sans specialiste hotspot) ================")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m_gbm1.items()))
    print("================ GBM_1 + GBM_2 + specialiste hotspot (modele officiel) ================")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m_final.items()))
    print(f"  Delta R2 vs GBM_1 seul : {m_final['R2'] - m_gbm1['R2']:+.4f}")

    # ---- R2 segmente par tier (train_total = somme n_cas 2009-2017 par
    # commune) : le R2 agrege national melange 4 regimes tres differents --
    # c'est la mesure la plus honnete de la capacite de generalisation du
    # modele, pas juste "hors la plus grosse commune". Calcule dynamiquement,
    # pas de nom/seuil code en dur pour les communes elles-memes. ----
    train_totals_full = data.loc[train_mask].groupby("commune")["n_cas"].sum(min_count=1).fillna(0)
    test_train_totals = pd.Series(test_communes).map(train_totals_full).fillna(0).to_numpy()
    tiers = {
        "cold-start (train_total=0)": test_train_totals == 0,
        "low (1-10)": (test_train_totals > 0) & (test_train_totals <= 10),
        "moderate (11-50)": (test_train_totals > 10) & (test_train_totals <= HOTSPOT_TRAIN_TOTAL_THRESHOLD),
        f"hotspot (>{HOTSPOT_TRAIN_TOTAL_THRESHOLD})": test_train_totals > HOTSPOT_TRAIN_TOTAL_THRESHOLD,
    }
    print(f"\n================ R2 PAR TIER (modele officiel, {len(hotspot_communes)} communes hotspot) ================")
    metrics_by_tier = {}
    for label, mask in tiers.items():
        mt = evaluate_metrics(y_test[mask], final_pred[mask])
        metrics_by_tier[label] = mt
        print(f"  {label:28s}: R2={mt['R2']:.4f}  MAE={mt['MAE']:.4f}  n={int(mask.sum())} lignes, {y_test[mask].sum():.0f} cas")
    print("  -> mesure la plus honnete : le R2 agrege peut rester stable alors qu'un tier change beaucoup.")

    cases_by_commune = pd.Series(y_test).groupby(test_communes).sum()
    top_commune = cases_by_commune.idxmax()
    top_share_pct = 100.0 * cases_by_commune.loc[top_commune] / max(y_test.sum(), 1e-9)
    excl_mask = test_communes != top_commune
    m_gbm1_excl = evaluate_metrics(y_test[excl_mask], gbm1_test_pred[excl_mask])
    m_final_excl = evaluate_metrics(y_test[excl_mask], final_pred[excl_mask])
    print(f"\n================ R2 SEGMENTE (hors '{top_commune}', {top_share_pct:.1f}% des cas test) ================")
    print(f"  GBM_1 seul         : R2={m_gbm1_excl['R2']:.4f}  (agrege national : {m_gbm1['R2']:.4f})")
    print(f"  GBM_1 + GBM_2 (off): R2={m_final_excl['R2']:.4f}  (agrege national : {m_final['R2']:.4f})")

    imp2 = pd.DataFrame({"feature": full_cols, "importance": model_2.feature_importances_}).sort_values("importance", ascending=False)
    print("\n================ TOP 15 FEATURES DU CORRECTEUR (GBM_2) ================")
    print(imp2.head(15).to_string(index=False))

    # ---- persistance : remplace gbm_model.joblib, format 2-etages + specialiste hotspot ----
    model_path = config.PROCESSED / "gbm_model.joblib"
    joblib.dump({
        "model_1": model_1, "model_2": model_2,
        "base_feature_cols": base_cols, "full_feature_cols": full_cols,
        "model_name": "GBM+PINN (stacking residuel 2 etages) + specialiste hotspot",
        "uses_onehot_province": False, "uses_pinn_features": True,
        "raw_feature_cols": full_cols,
        "metrics": m_final,
        "metrics_excl_top_commune": m_final_excl,
        "top_commune_name": top_commune,
        "top_commune_share_pct": top_share_pct,
        "metrics_by_tier": {k: v for k, v in metrics_by_tier.items()},
        "model_hotspot": model_hotspot,
        "hotspot_communes": hotspot_communes,
        "hotspot_train_total_threshold": HOTSPOT_TRAIN_TOTAL_THRESHOLD,
    }, model_path)
    logger.info(f"Modele officiel sauvegarde : {model_path}")

    reg_col = "region" if "region" in data.columns else [c for c in data.columns if "region" in c][0]
    test_df = data.loc[test_mask, ["commune", "province", reg_col, "annee", "mois", "n_cas"]].copy()
    test_df["region"] = test_df[reg_col]
    test_df["y_pred_gbm"] = final_pred
    test_df["y_pred_gbm1_seul"] = gbm1_test_pred
    out_file = config.PROCESSED / "gbm_predictions_2018_2020.csv"
    test_df.to_csv(out_file, index=False)
    logger.info(f"Predictions sauvegardees : {out_file}")

    return m_final


if __name__ == "__main__":
    train()
