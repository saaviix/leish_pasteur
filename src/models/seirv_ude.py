"""
seirv_ude.py
=============
Reconstruction du modele mecaniste-neural hybride en suivant precisement la
methodologie de l'article de reference (structure universal differential
equations, Rackauckas et al. 2020 ; applique a la dengue par Zhang/Wang/Tang
2024, PLOS Comp Bio) -- PAS l'ancienne approche pinn_seirv.py (residu
physique sur points aleatoires, instable, R2 <0 meme apres 4 corrections).

Difference structurelle cle avec pinn_seirv.py :
  - UNE SEULE piece est neuronale : le recrutement effectif du vecteur
    Lambda_V(t) = kappa_p * q_theta(z_lag), q_theta : petit reseau tanh+
    sigmoid borne dans (0,1), entree = climat retarde standardise.
  - Toutes les autres constantes biologiques (piqure, transmission,
    incubation, mortalite) sont des CONSTANTES GLOBALES estimees (pas des
    fonctions temperature apprises point par point -- on n'a pas de courbe
    de reponse thermique publiee pour P. sergenti/L. tropica comme Mordecai
    et al. l'ont pour Aedes/dengue, donc on ne pretend pas en avoir une).
  - Integration RK4 REELLE d'une trajectoire continue par province (pas une
    perte de residu sur des points i.i.d.) -- la physique est BAKED IN par
    la simulation elle-meme, pas imposee comme penalite molle.
  - Modele d'observation Binomial Negatif explicite avec une VRAIE
    probabilite de rapportage rho -- repond proprement a la question de
    sous-declaration (pas besoin d'une tete d'observation ad hoc).

Resolution mensuelle (pas hebdomadaire comme l'article -- notre panel est
mensuel), niveau PROVINCE (pas commune -- trop clairseme pour un ajustement
par unite fiable, 76 provinces ~ 5 villes de l'article mais avec moins de
signal chacune, d'ou rho/phi partages plutot que par province).

Sortie :
  outputs/processed/seirv_ude_metrics.csv
  outputs/processed/seirv_ude_predictions.csv
  outputs/processed/seirv_ude_recruitment_by_month.csv

Usage :
  python src/models/seirv_ude.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config  # noqa: E402

torch.manual_seed(42)

N_SUB = 8           # sous-pas RK4 par mois (integration continue) -- reduit de 30 a 8
                     # pour rester dans un temps de calcul raisonnable (dt~3.8j,
                     # toujours largement suffisant pour la dynamique lente de ce
                     # systeme -- taux exprimes en mois, pas de raideur numerique)
LAG_MONTHS = 3       # memoire climatique (mois retardes en entree du reseau)
HIDDEN = 8
TRAIN_YEARS = (2009, 2017)
TEST_YEARS = (2018, 2020)


# --------------------------------------------------------------------------
# Constantes biologiques globales (estimees, contraintes positives/[0,1])
# --------------------------------------------------------------------------
class DiseaseConstants(nn.Module):
    def __init__(self):
        super().__init__()
        self._a_raw = nn.Parameter(torch.tensor(0.0))          # piqures/mois
        self._bvh_raw = nn.Parameter(torch.tensor(-1.0))       # P(vecteur->humain | piqure)
        self._bhv_raw = nn.Parameter(torch.tensor(-1.0))       # P(humain->vecteur | piqure)
        self._etaV_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 0.5))))   # 1/EIP (mois)
        self._muV_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 1.0))))    # mortalite vecteur /mois
        self._etaH_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 2.5))))   # 1/incubation humaine (mois)
        self._gammaH_raw = nn.Parameter(torch.tensor(float(np.log(1.0 / 15.0))))  # 1/duree infectiosite (mois)

    @property
    def a(self): return nn.functional.softplus(self._a_raw) + 1e-3
    @property
    def beta_VH(self): return torch.sigmoid(self._bvh_raw)
    @property
    def beta_HV(self): return torch.sigmoid(self._bhv_raw)
    @property
    def eta_V(self): return nn.functional.softplus(self._etaV_raw) + 1e-3
    @property
    def mu_V(self): return nn.functional.softplus(self._muV_raw) + 1e-3
    @property
    def eta_H(self): return nn.functional.softplus(self._etaH_raw) + 1e-3
    @property
    def gamma_H(self): return nn.functional.softplus(self._gammaH_raw) + 1e-3


class RecruitmentNet(nn.Module):
    """q_theta(z) -> (0,1), reseau PARTAGE entre toutes les provinces --
    seule piece neuronale du modele (eq. 6 de l'article de reference)."""

    def __init__(self, n_in, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(-1)


def neg_binomial_nll(y, mean, phi):
    """Log-vraisemblance Binomiale Negative, parametrisation moyenne-dispersion
    (eq. 21-22 de l'article) : Var(Y) = mean + mean^2/phi."""
    eps = 1e-8
    mean = mean.clamp(min=eps)
    ll = (
        torch.lgamma(y + phi) - torch.lgamma(phi) - torch.lgamma(y + 1)
        + phi * torch.log(phi / (phi + mean) + eps)
        + y * torch.log(mean / (phi + mean) + eps)
    )
    return -ll


def build_province_month_panel() -> pd.DataFrame:
    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv")
    prov = (
        panel.groupby(["province", "annee", "mois"])
        .agg(n_cas=("n_cas", "sum"), pop=("pop_total", "sum"),
             temp=("temp_moy", "mean"), precip=("precip_mm", "mean"), humid=("humidite_pct", "mean"))
        .reset_index()
    )
    prov = prov.sort_values(["province", "annee", "mois"]).reset_index(drop=True)
    return prov


def build_tensors(prov: pd.DataFrame):
    provinces = sorted(prov["province"].unique().tolist())
    p_idx = {p: i for i, p in enumerate(provinces)}
    months = sorted(prov[["annee", "mois"]].drop_duplicates().itertuples(index=False, name=None))
    m_idx = {m: i for i, m in enumerate(months)}
    n_p, n_m = len(provinces), len(months)

    T = np.full((n_m, n_p), np.nan, dtype=np.float64)
    P = np.full((n_m, n_p), np.nan, dtype=np.float64)
    H = np.full((n_m, n_p), np.nan, dtype=np.float64)
    Y = np.zeros((n_m, n_p), dtype=np.float64)
    for row in prov.itertuples(index=False):
        mi, pi = m_idx[(row.annee, row.mois)], p_idx[row.province]
        T[mi, pi] = row.temp
        P[mi, pi] = row.precip
        H[mi, pi] = row.humid
        Y[mi, pi] = row.n_cas

    # population par province (constante -- pas de demographie humaine, cf. docstring)
    N_h = prov.groupby("province")["pop"].first().reindex(provinces).values.astype(np.float64)
    N_h = np.where(np.isnan(N_h) | (N_h <= 0), np.nanmedian(N_h), N_h)

    return provinces, months, T, P, H, Y, N_h


def standardize_climate(T, P, H, train_mask):
    """Standardisation P* = log1p(P) puis (X-mean)/std, statistiques
    calculees UNIQUEMENT sur la periode d'entrainement (eq. 9-10 de
    l'article) -- jamais sur validation/test."""
    P_star = np.log1p(P)
    stats_ = {}
    out = {}
    for name, X in [("T", T), ("P", P_star), ("H", H)]:
        train_vals = X[train_mask]
        mu, sd = np.nanmean(train_vals), np.nanstd(train_vals)
        sd = sd if sd > 1e-6 else 1.0
        out[name] = (X - mu) / sd
        stats_[name] = (mu, sd)
    return out["T"], out["P"], out["H"], stats_


def build_lag_input(T_std, P_std, H_std, month_i, L):
    """z_w = [T_w..T_w-L+1, P*_w..P*_w-L+1, H_w..H_w-L+1] (eq. 8), avec
    repli sur le premier mois disponible pour le debut de serie (spin-up
    implicite, pas de perte de mois)."""
    idxs = [max(0, month_i - k) for k in range(L)]
    parts = [T_std[idxs], P_std[idxs], H_std[idxs]]
    return np.concatenate(parts, axis=0)  # shape (3L, n_provinces) -> transposer ensuite


def rk4_simulate(consts, recruit_net, kappa_p, T_std, P_std, H_std, N_h_t,
                  n_months, n_provinces, L, use_recruitment=True):
    """Simule TOUTES les provinces en parallele (dimension batch = province),
    integration RK4 a dt = 1/N_SUB mois, climat constant a l'interieur d'un
    mois (comme l'article, forcage constant a l'interieur d'une semaine)."""
    S_V = torch.full((n_provinces,), 0.7, dtype=torch.float64)
    E_V = torch.zeros(n_provinces, dtype=torch.float64)
    I_V = torch.zeros(n_provinces, dtype=torch.float64)
    I_H0 = torch.clamp(torch.tensor(1.0, dtype=torch.float64), min=1.0) / N_h_t
    S_H = 1.0 - I_H0
    E_H = torch.zeros(n_provinces, dtype=torch.float64)
    I_H = I_H0.clone()
    R_H = torch.zeros(n_provinces, dtype=torch.float64)
    # S_V/E_V/I_V en unites "fraction de N_h" (comme l'article, S_m(0)=0.7*N_h) ;
    # S_H/E_H/I_H/R_H en fraction de N_h (somme = 1, pas de demographie).

    a, beta_VH, beta_HV = consts.a, consts.beta_VH, consts.beta_HV
    eta_V, mu_V, eta_H, gamma_H = consts.eta_V, consts.mu_V, consts.eta_H, consts.gamma_H

    T_t = torch.tensor(T_std, dtype=torch.float64)
    P_t = torch.tensor(P_std, dtype=torch.float64)
    H_t = torch.tensor(H_std, dtype=torch.float64)

    C_months = []
    dt = 1.0 / N_SUB

    def rhs(state, recruitment):
        S_V, E_V, I_V, S_H, E_H, I_H, R_H = state
        foi_h = a * beta_VH * I_V          # force d'infection sur les humains (I_V deja en fraction)
        foi_v = a * beta_HV * I_H          # force d'infection sur les vecteurs
        dS_V = recruitment - foi_v * S_V - mu_V * S_V
        dE_V = foi_v * S_V - (eta_V + mu_V) * E_V
        dI_V = eta_V * E_V - mu_V * I_V
        dS_H = -foi_h * S_H
        dE_H = foi_h * S_H - eta_H * E_H
        dI_H = eta_H * E_H - gamma_H * I_H
        dR_H = gamma_H * I_H
        dC = eta_H * E_H
        return (dS_V, dE_V, dI_V, dS_H, dE_H, dI_H, dR_H), dC

    for month_i in range(n_months):
        idxs = [max(0, month_i - k) for k in range(L)]
        z = torch.cat([T_t[idxs], P_t[idxs], H_t[idxs]], dim=0).T.float()  # (n_provinces, 3L)
        if use_recruitment:
            recruitment = kappa_p * recruit_net(z).double()
        else:
            recruitment = kappa_p.double()  # ablation : recrutement constant

        C_month = torch.zeros(n_provinces, dtype=torch.float64)
        state = (S_V, E_V, I_V, S_H, E_H, I_H, R_H)
        for _ in range(N_SUB):
            k1, dC1 = rhs(state, recruitment)
            s2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
            k2, dC2 = rhs(s2, recruitment)
            s3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
            k3, dC3 = rhs(s3, recruitment)
            s4 = tuple(s + dt * k for s, k in zip(state, k3))
            k4, dC4 = rhs(s4, recruitment)
            state = tuple(
                torch.clamp(s + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4), min=0.0)
                for s, a1, a2, a3, a4 in zip(state, k1, k2, k3, k4)
            )
            C_month = C_month + (dt / 6.0) * (dC1 + 2 * dC2 + 2 * dC3 + dC4)
        S_V, E_V, I_V, S_H, E_H, I_H, R_H = state
        C_months.append(C_month * N_h_t)  # retour en nombre de cas (pas fraction)

    return torch.stack(C_months, dim=0)  # (n_months, n_provinces)


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
    sp = stats.spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Spearman": sp}


def train_one(use_recruitment: bool, label: str, T_std, P_std, H_std, N_h_t, Y, n_months, n_provinces,
              train_month_mask, n_epochs=400):
    consts = DiseaseConstants()
    recruit_net = RecruitmentNet(n_in=3 * LAG_MONTHS)
    kappa_raw = nn.Parameter(torch.zeros(n_provinces, dtype=torch.float64))
    rho_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))   # sigmoid -> ~0.12 au depart
    phi_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(10.0)))))  # softplus -> 10 au depart

    params = list(consts.parameters()) + [kappa_raw, rho_raw, phi_raw]
    if use_recruitment:
        params += list(recruit_net.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-3)

    Y_t = torch.tensor(Y, dtype=torch.float64)
    train_idx = np.where(train_month_mask)[0]

    print(f"\n--- Entrainement {label} ({n_epochs} epoques, {n_provinces} provinces, {n_months} mois) ---")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        kappa_p = nn.functional.softplus(kappa_raw)
        rho = torch.sigmoid(rho_raw)
        phi = nn.functional.softplus(phi_raw) + 1e-3

        C = rk4_simulate(consts, recruit_net, kappa_p, T_std, P_std, H_std, N_h_t,
                          n_months, n_provinces, LAG_MONTHS, use_recruitment=use_recruitment)
        mean_pred = rho * C[train_idx]
        nll = neg_binomial_nll(Y_t[train_idx], mean_pred, phi)

        l2 = sum((p ** 2).sum() for p in recruit_net.parameters()) if use_recruitment else torch.tensor(0.0)
        loss = nll.mean() + 1e-4 * l2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  NLL={nll.mean().item():.4f}  rho={rho.item():.4f}  "
                  f"phi={phi.item():.2f}  a={consts.a.item():.3f}  1/etaH={1/consts.eta_H.item():.2f}mo  "
                  f"1/gammaH={1/consts.gamma_H.item():.2f}mo")

    with torch.no_grad():
        kappa_p = nn.functional.softplus(kappa_raw)
        rho = torch.sigmoid(rho_raw)
        C_final = rk4_simulate(consts, recruit_net, kappa_p, T_std, P_std, H_std, N_h_t,
                                n_months, n_provinces, LAG_MONTHS, use_recruitment=use_recruitment)
        y_pred_all = (rho * C_final).numpy()

    return {
        "y_pred": y_pred_all, "rho": rho.item(), "phi": nn.functional.softplus(phi_raw).item() + 1e-3,
        "consts": consts, "kappa_p": kappa_p.detach().numpy(), "recruit_net": recruit_net,
    }


def main():
    config.ensure_dirs()
    prov = build_province_month_panel()
    provinces, months, T, P, H, Y, N_h = build_tensors(prov)
    n_months, n_provinces = len(months), len(provinces)
    years = np.array([m[0] for m in months])

    train_mask = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    test_mask = (years >= TEST_YEARS[0]) & (years <= TEST_YEARS[1])

    T = np.nan_to_num(T, nan=np.nanmean(T))
    P = np.nan_to_num(P, nan=np.nanmean(P))
    H = np.nan_to_num(H, nan=np.nanmean(H))
    T_std, P_std, H_std, clim_stats = standardize_climate(T, P, H, train_mask)
    N_h_t = torch.tensor(N_h, dtype=torch.float64)

    print(f"{n_provinces} provinces x {n_months} mois. Train={train_mask.sum()} mois, Test={test_mask.sum()} mois.")
    print(f"Stats climat (train uniquement) : {clim_stats}")

    results = {}
    for use_rec, label in [(True, "SEIR-V + recrutement neuronal (modele complet)"),
                            (False, "Ablation : recrutement constant (sans reseau)")]:
        out = train_one(use_rec, label, T_std, P_std, H_std, N_h_t, Y, n_months, n_provinces,
                         train_mask, n_epochs=200)
        y_pred_test = out["y_pred"][test_mask].flatten()
        y_true_test = Y[test_mask].flatten()
        m = evaluate(y_true_test, y_pred_test)
        results[label] = m
        out["metrics"] = m
        results[label + "_obj"] = out
        print(f"[{label}] TEST 2018-2020 : R2={m['R2']:+.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
              f"Spearman={m['Spearman']:.4f}  rho(rapportage)={out['rho']:.4f}")

    metrics_rows = [{"modele": k, **v} for k, v in results.items() if not k.endswith("_obj")]
    pd.DataFrame(metrics_rows).to_csv(config.PROCESSED / "seirv_ude_metrics.csv", index=False)

    best = results["SEIR-V + recrutement neuronal (modele complet)_obj"]
    pred_rows = []
    for mi, (annee, mois) in enumerate(months):
        for pi, province in enumerate(provinces):
            pred_rows.append({
                "province": province, "annee": annee, "mois": mois,
                "n_cas": Y[mi, pi], "y_pred_seirv_ude": best["y_pred"][mi, pi],
            })
    pd.DataFrame(pred_rows).to_csv(config.PROCESSED / "seirv_ude_predictions.csv", index=False)

    print(f"\n{'='*90}\nPARAMETRES BIOLOGIQUES FINAUX (modele complet)\n{'='*90}")
    c = best["consts"]
    print(f"  taux de piqure a         = {c.a.item():.4f} / mois")
    print(f"  beta_VH (vecteur->humain)= {c.beta_VH.item():.4f}")
    print(f"  beta_HV (humain->vecteur)= {c.beta_HV.item():.4f}")
    print(f"  1/eta_V (EIP)            = {1/c.eta_V.item():.2f} mois")
    print(f"  1/mu_V (survie vecteur)  = {1/c.mu_V.item():.2f} mois")
    print(f"  1/eta_H (incubation)     = {1/c.eta_H.item():.2f} mois")
    print(f"  1/gamma_H (infectiosite) = {1/c.gamma_H.item():.2f} mois")
    print(f"  rho (proba de rapportage)= {best['rho']:.4f}  <-- reponse a la sous-declaration")
    print(f"  phi (dispersion NegBin)  = {best['phi']:.2f}")

    print(f"\n[OK] outputs/processed/seirv_ude_metrics.csv")
    print(f"[OK] outputs/processed/seirv_ude_predictions.csv")


if __name__ == "__main__":
    main()
