"""
model_io.py
============
Chargement + application du modele de prediction de cas officiel du projet
(outputs/processed/gbm_model.joblib), produit par
src/models/gbm_pinn_stacked.py : stacking residuel a 2 etages -- GBM_1
(LightGBM Tweedie, features standard) + GBM_2 (correcteur de residu, acces
aux features mecanistes du PINN SEIR-V). R2=0.552 sur le holdout 2018-2020
(0.260 hors la commune la plus lourde du test -- 20% des cas -- la mesure
la plus honnete de generalisation), contre 0.2265 avant les fix de donnees
de cette session -- voir le docstring de gbm_pinn_stacked.py pour le detail
de pourquoi cette architecture (et pas une simple injection de features)
est necessaire pour que le PINN apporte une vraie valeur.

Formats supportes par `predict_gbm_saved` (retro-compatible) :
  - 2 etages (officiel) : dict avec "model_1"/"model_2"/"base_feature_cols"/
    "full_feature_cols" -- necessite les features PINN dans `df` (ajoutees
    automatiquement via gbm_spatial_temporal.add_pinn_physics_features si
    absentes et si les colonnes source sont disponibles).
  - 1 seul modele (ancien format, gbm_spatial_temporal.py seul) : dict avec
    "model"/"feature_cols" -- gere pour ne pas casser d'anciens artefacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_gbm(processed_dir: Path):
    import joblib
    model_path = processed_dir / "gbm_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} introuvable. Lance d'abord : python src/models/gbm_pinn_stacked.py"
        )
    return joblib.load(model_path)


def _prep(df: pd.DataFrame, cols: list, uses_onehot: bool, final_cols) -> pd.DataFrame:
    X = df.reindex(columns=cols).copy()
    non_cat = [c for c in cols if c != "province"]
    X[non_cat] = X[non_cat].fillna(0.0)
    if "province" in X.columns:
        if uses_onehot:
            X = pd.get_dummies(X, columns=["province"])
            X = X.reindex(columns=final_cols, fill_value=0)
        else:
            X["province"] = X["province"].astype("category")
            X = X.reindex(columns=final_cols)
    return X


def predict_gbm_saved(saved: dict, df: pd.DataFrame) -> np.ndarray:
    """Applique le modele persiste (dict de load_gbm) a `df`."""
    uses_onehot = saved.get("uses_onehot_province", False)

    if "model_2" in saved:
        # ---- format officiel 2 etages ----
        full_cols = saved["full_feature_cols"]
        base_cols = saved["base_feature_cols"]
        if saved.get("uses_pinn_features") and not any(c.startswith("pinn_") and c in df.columns for c in full_cols):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
                from gbm_spatial_temporal import add_pinn_physics_features  # noqa: E402
                df = add_pinn_physics_features(df)
            except Exception as e:
                print(f"[WARN] features PINN non ajoutees ({e}) -- GBM_2 les recevra a 0/NaN, "
                      f"predictions degradees mais non bloquantes")

        X_base = _prep(df, base_cols, uses_onehot, base_cols)
        X_full = _prep(df, full_cols, uses_onehot, full_cols)
        pred_1 = np.clip(saved["model_1"].predict(X_base), 0, None)
        pred_2 = saved["model_2"].predict(X_full)
        pred = np.clip(pred_1 + pred_2, 0, None)

        if "model_hotspot" in saved and "commune" in df.columns:
            hotspot_communes = saved["hotspot_communes"]
            mask = df["commune"].isin(hotspot_communes).to_numpy()
            if mask.any():
                pred[mask] = np.clip(saved["model_hotspot"].predict(X_base.loc[mask]), 0, None)
        return pred

    # ---- retro-compatibilite : ancien format a 1 seul modele ----
    raw_cols = saved.get("raw_feature_cols", saved["feature_cols"])
    X = _prep(df, raw_cols, uses_onehot, saved["feature_cols"])
    return np.clip(saved["model"].predict(X), 0, None)
