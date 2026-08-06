     

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "pinn_seirv.py requires PyTorch to compute the autograd-based ODE "
        "residuals that make this a physics-informed model in the first "
        "place (`pip install torch`). The previous version of this file "
        "fell back to a hand-written formula when torch was missing and "
        "labeled its output 'PINN predictions' -- that was not a real "
        "model and should not be trusted for anything already generated "
        "by it."
    ) from e


# ---------------------------------------------------------------------------
# Climate-response sub-networks: small, strictly-positive, LEARNED functions
# of climate covariates (replace any fixed/hard-coded climate formula).
# ---------------------------------------------------------------------------
class ClimateResponse(nn.Module):
    """MLP -> softplus: a smooth, strictly-positive, learnable function of
    1-2 climate covariates. Used for emergence(T,P), mu_V(T), sigma_V(T)."""

    def __init__(self, n_in, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return nn.functional.softplus(self.net(x)) + 1e-6


class SEIRVPINN(nn.Module):
    """
    forward(t, lat, lon, temp, precip, humid) -> dict of the 7 state
    variables + the 3 climate-response outputs, all as smooth functions of
    continuous time t (t must allow gradient tracking upstream for the
    physics residual to be computable).
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.human_head = nn.Linear(hidden_dim, 4)    # -> softmax (S_H,E_H,I_H,R_H)
        self.vector_head = nn.Linear(hidden_dim, 3)   # -> softplus (S_V,E_V,I_V)

        self.emergence_fn = ClimateResponse(n_in=2)   # f(temp, precip)
        self.mortality_fn = ClimateResponse(n_in=1)   # f(temp)
        self.eip_fn = ClimateResponse(n_in=1)          # f(temp) -- extrinsic incubation rate

        # Disease-intrinsic rate constants: shared across all locations
        # (exactly like latent/infectious periods were treated as shared
        # constants in the earlier province-level SIR calibration in this
        # project -- here they are FITTED rather than fixed by hand).
        # Parametrized in "raw" (unconstrained) space and mapped through a
        # positive/[0,1] transform so gradient descent can't push them to
        # biologically impossible values (negative rates, probabilities >1).
        self._a_raw = nn.Parameter(torch.tensor(0.0))
        self._bh_raw = nn.Parameter(torch.tensor(-1.0))
        self._cv_raw = nn.Parameter(torch.tensor(-1.0))
        self._sigmaH_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 2.0))))
        self._gammaH_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 8.0))))

    @property
    def a(self):
        return nn.functional.softplus(self._a_raw) + 1e-3

    @property
    def b_h(self):
        return torch.sigmoid(self._bh_raw)

    @property
    def c_v(self):
        return torch.sigmoid(self._cv_raw)

    @property
    def sigma_H(self):
        return nn.functional.softplus(self._sigmaH_raw) + 1e-3

    @property
    def gamma_H(self):
        return nn.functional.softplus(self._gammaH_raw) + 1e-3

    def forward(self, t, lat, lon, temp, precip, humid):
        x = torch.cat([t, lat, lon, temp, precip, humid], dim=1)
        h = self.trunk(x)

        human = torch.softmax(self.human_head(h), dim=1)
        S_H, E_H, I_H, R_H = human[:, 0:1], human[:, 1:2], human[:, 2:3], human[:, 3:4]

        vector = nn.functional.softplus(self.vector_head(h)) + 1e-6
        S_V, E_V, I_V = vector[:, 0:1], vector[:, 1:2], vector[:, 2:3]

        emergence = self.emergence_fn(torch.cat([temp, precip], dim=1))
        mu_V = self.mortality_fn(temp)
        sigma_V = self.eip_fn(temp)

        return {
            "S_H": S_H, "E_H": E_H, "I_H": I_H, "R_H": R_H,
            "S_V": S_V, "E_V": E_V, "I_V": I_V,
            "emergence": emergence, "mu_V": mu_V, "sigma_V": sigma_V,
        }


def grad_wrt_t(y, t):
    """d(y)/dt via autograd. `t` must require_grad and be part of the graph
    that produced `y`. create_graph=True keeps the derivative itself
    differentiable, which is required to backprop the physics loss into the
    network's weights (this is the mechanism that was entirely missing
    before)."""
    return torch.autograd.grad(
        y, t, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def physics_residual(model, t, lat, lon, temp, precip, humid):
    """The true ODE residual: (autograd derivative) - (equation RHS) for
    each of the 7 SEIR-V equations. Driving this to zero during training is
    what enforces the mechanistic dynamics -- this is the part that makes
    the network a PINN rather than a plain regressor."""
    t = t.clone().requires_grad_(True)
    out = model(t, lat, lon, temp, precip, humid)
    S_H, E_H, I_H, R_H = out["S_H"], out["E_H"], out["I_H"], out["R_H"]
    S_V, E_V, I_V = out["S_V"], out["E_V"], out["I_V"]
    emergence, mu_V, sigma_V = out["emergence"], out["mu_V"], out["sigma_V"]

    a, b_h, c_v = model.a, model.b_h, model.c_v
    sigma_H, gamma_H = model.sigma_H, model.gamma_H

    dS_H_dt = grad_wrt_t(S_H, t)
    dE_H_dt = grad_wrt_t(E_H, t)
    dI_H_dt = grad_wrt_t(I_H, t)
    dR_H_dt = grad_wrt_t(R_H, t)
    dS_V_dt = grad_wrt_t(S_V, t)
    dE_V_dt = grad_wrt_t(E_V, t)
    dI_V_dt = grad_wrt_t(I_V, t)

    foi_h = b_h * a * I_V   # force of infection acting on humans
    foi_v = c_v * a * I_H   # force of infection acting on vectors

    res_SH = dS_H_dt - (-foi_h * S_H)
    res_EH = dE_H_dt - (foi_h * S_H - sigma_H * E_H)
    res_IH = dI_H_dt - (sigma_H * E_H - gamma_H * I_H)
    res_RH = dR_H_dt - (gamma_H * I_H)

    res_SV = dS_V_dt - (emergence - foi_v * S_V - mu_V * S_V)
    res_EV = dE_V_dt - (foi_v * S_V - sigma_V * E_V - mu_V * E_V)
    res_IV = dI_V_dt - (sigma_V * E_V - mu_V * I_V)

    residuals = torch.cat([res_SH, res_EH, res_IH, res_RH, res_SV, res_EV, res_IV], dim=1)
    return torch.mean(residuals ** 2), out


def load_population(df):
    """Look for a population column under a few likely names. Every model
    in this project so far has lacked one (flagged repeatedly) -- if it is
    still missing, WARN loudly and fall back to a shared placeholder rather
    than silently guessing per-location absolute numbers. This makes the
    limitation visible in the logs instead of hidden inside the output."""
    for col in ["population", "pop", "population_2024", "N_h"]:
        if col in df.columns:
            return df[col].values.astype(float), True
    logger.warning(
        "Aucune colonne de population trouvee (cherche 'population'/'pop'/"
        "'N_h'). Utilisation d'une valeur PARTAGEE arbitraire (200000) pour "
        "convertir les fractions E_H en nombre de cas -- les nombres "
        "ABSOLUS par commune/province ne seront pas fiables tant que cette "
        "colonne n'est pas fournie. Le RANG relatif des zones a risque "
        "reste interpretable."
    )
    return np.full(len(df), 200000.0), False


def train_pinn():
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        logger.error(f"Fichier panel introuvable: {panel_path}")
        return None

    df = pd.read_csv(panel_path)
    data = df[(df["annee"] >= 2009) & (df["annee"] <= 2020)].copy()
    data = data.dropna(subset=["temp_moy", "precip_mm", "humidite_pct"])

    pop_values, has_real_pop = load_population(data)
    data["_pop"] = pop_values


    data["t_months"] = (data["annee"] - data["annee"].min()) * 12 + (data["mois"] - 1)

    feature_cols = ["t_months", "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct"]
    X = data[feature_cols].values.astype(np.float32)
    
    t_raw, lat_raw, lon_raw = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    temp_raw, precip_raw, humid_raw = X[:, 3:4], X[:, 4:5], X[:, 5:6]

    y_cases = data["n_cas"].values.astype(np.float32)
    N_h = data["_pop"].values.astype(np.float32)

    train_mask = (data["annee"] <= 2017).values
    test_mask = (data["annee"] >= 2018).values

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    model = SEIRVPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t_tr = to_t(t_raw[train_mask])
    lat_tr, lon_tr = to_t(lat_raw[train_mask]), to_t(lon_raw[train_mask])
    temp_tr, precip_tr, humid_tr = to_t(temp_raw[train_mask]), to_t(precip_raw[train_mask]), to_t(humid_raw[train_mask])
    y_tr = to_t(y_cases[train_mask].reshape(-1, 1))
    Nh_tr = to_t(N_h[train_mask].reshape(-1, 1))

    n_epochs = 2000
    logger.info(f"Entrainement PINN SEIR-V (autograd, {n_epochs} epoques, {len(t_tr)} points d'entrainement)...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        phys_loss, out = physics_residual(model, t_tr, lat_tr, lon_tr, temp_tr, precip_tr, humid_tr)

        # Data loss: sigma_H * E_H is the per-capita rate of NEW symptomatic
        # infections; multiplying by the local population gives an expected
        # case COUNT, which is directly comparable to n_cas. This replaces
        # the previous "sigma_h * E_H * 1000" (an arbitrary constant, and a
        # SEPARATE, disconnected sigma_h parameter never used in the physics).
        predicted_incidence = model.sigma_H * out["E_H"] * Nh_tr
        data_loss = torch.mean((predicted_incidence - y_tr) ** 2)

        loss = data_loss + 0.1 * phys_loss
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            logger.info(f"  epoch {epoch:5d}  data_loss={data_loss.item():.4f}  phys_loss={phys_loss.item():.6f}")

    # ---------------------------------------------------------------- evaluate on the genuinely held-out 2018-2020 rows
    t_te = to_t(t_raw[test_mask]); lat_te, lon_te = to_t(lat_raw[test_mask]), to_t(lon_raw[test_mask])
    temp_te, precip_te, humid_te = to_t(temp_raw[test_mask]), to_t(precip_raw[test_mask]), to_t(humid_raw[test_mask])
    Nh_te = to_t(N_h[test_mask].reshape(-1, 1))

    with torch.no_grad():
        out_te = model(t_te, lat_te, lon_te, temp_te, precip_te, humid_te)
        y_pred_pinn = (model.sigma_H * out_te["E_H"] * Nh_te).numpy().flatten()
    y_pred_pinn = np.clip(y_pred_pinn, 0, None)

    reg_col = "region" if "region" in data.columns else [c for c in data.columns if "region" in c][0]
    test_df = data.loc[test_mask, ["commune", "province", reg_col, "annee", "mois", "n_cas"]].copy()
    test_df["region"] = test_df[reg_col]
    test_df["y_pred_pinn"] = y_pred_pinn
    if not has_real_pop:
        test_df["avertissement"] = "population non fournie -- valeurs absolues peu fiables, rang relatif OK"
    out_file = config.PROCESSED / "pinn_predictions_2018_2020.csv"
    test_df.to_csv(out_file, index=False)
    logger.info(f"Predictions PINN enregistrees dans : {out_file}")

    # Persist the trained model + the normalization/feature metadata needed
    # to re-run it later (e.g. to predict on 2021/2023/2024 climate rows in
    # robust_ensemble_recalibrated.py). The previous pipeline never saved
    # this, which is part of why the ensemble script could not honestly
    # generate forward predictions for the verification years and instead
    # re-used the 2018-2020 test predictions for a comparison that didn't
    # line up in time.
    weights_path = config.PROCESSED / "pinn_seirv_weights.pt"
    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols}, weights_path)
    logger.info(f"Poids du modele sauvegardes dans : {weights_path}")

    logger.info("\nParametres epidemiologiques appris (constantes partagees, a comparer a la litterature):")
    logger.info(f"  taux de piqure a         = {model.a.item():.4f} / mois")
    logger.info(f"  b_h (vecteur -> humain)  = {model.b_h.item():.4f}")
    logger.info(f"  c_v (humain -> vecteur)  = {model.c_v.item():.4f}")
    logger.info(f"  1/sigma_H (incubation)   = {1.0/model.sigma_H.item():.2f} mois")
    logger.info(f"  1/gamma_H (infectiosite) = {1.0/model.gamma_H.item():.2f} mois")

    return model


if __name__ == "__main__":
    train_pinn()