     

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
    """MLP -> sigmoid bornee (0, scale) : fonction lisse, strictement
    positive ET BORNEE d'1-2 covariables climatiques. Utilisee pour
    emergence(T,P), mu_V(T), sigma_V(T).

    Remplace le softplus non borne de la version precedente (session
    2026-08-15) : audit comparatif contre un papier de reference sur un
    hybride mecaniste-neuronal similaire (recrutement moustique/dengue,
    Bresil) montre que leur fonction climat->recrutement equivalente est
    bornee par construction (sigmoide), pas juste par un terme de perte --
    diagnostic coherent avec nos propres symptomes (saisonnalite apprise en
    phase inversee, amplitude quasi nulle : signe typique d'une sortie
    neuronale insuffisamment contrainte, cf. Kao & Eisenberg 2018 sur
    l'identifiabilite pratique des modeles vectoriels)."""

    def __init__(self, n_in, hidden=16, scale=3.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)) * self.scale + 1e-6


class SEIRVPINN(nn.Module):
    """
    forward(t, lat, lon, temp, precip, humid, hist) -> dict of the 7 state
    variables + the 3 climate-response outputs, all as smooth functions of
    continuous time t (t must allow gradient tracking upstream for the
    physics residual to be computable). `hist` = [log1p(cases_lag1),
    log1p(cases_roll3), log1p(cases_roll6)], a fixed conditioning input.
    """

    def __init__(self, hidden_dim=64, n_provinces=76, prov_embed_dim=8):
        super().__init__()
        # 6 covariables physiques/climatiques (t, lat, lon, temp, precip,
        # humid) + 3 covariables d'historique local de cas (log1p(lag1),
        # log1p(roll3), log1p(roll6)) -- sans elles le reseau n'avait aucun
        # acces au signal le plus predictif du GBM (cases_roll6/cases_lag12
        # comptent parmi ses features les plus importantes, juste apres
        # province). Ce sont des covariables d'entree (conditionnement),
        # PAS des variables d'etat SEIR -- elles ne sont pas differentiees
        # par rapport a t dans physics_residual, donc n'affectent pas la
        # contrainte physique elle-meme, seulement le "prior" appris.
        # + un embedding de province (categorielle discrete) : lat/lon seuls
        # ne peuvent representer qu'un risque de base LISSE dans l'espace,
        # alors que "province" est de loin la feature la plus importante du
        # GBM (~20%) -- un effet local discret que la geometrie continue de
        # lat/lon ne peut pas capturer.
        self.province_embed = nn.Embedding(n_provinces, prov_embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(9 + prov_embed_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.human_head = nn.Linear(hidden_dim, 4)    # -> softmax (S_H,E_H,I_H,R_H)
        self.vector_head = nn.Linear(hidden_dim, 3)   # -> softplus (S_V,E_V,I_V)
        self.obs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )  # -> softplus, taux de cas observe (modele d'observation, voir forward())

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

        # sigma_H, gamma_H : FIXES a des valeurs cliniques de la litterature
        # (session 2026-08-15), plus appris librement -- suit la discipline
        # d'un papier de reference audite cette session (hybride mecaniste-
        # neuronal dengue, Bresil) : ne laisser le reseau apprendre QUE ce
        # qui est reellement inconnu (la reponse climat->vecteur, sans
        # equivalent des courbes de Briere de Mordecai et al. pour
        # P. sergenti/L. tropica), pas des constantes deja documentees
        # cliniquement -- reduit le nombre de mecanismes appris simultanement
        # et le risque de compensation entre eux (cf. Kao & Eisenberg 2018).
        # Valeurs (recherche web session 2026-08-15, pas de calibration sur
        # nos propres donnees) :
        #  - 1/sigma_H = 2 mois : incubation typique de la LCT urbaine/
        #    L. tropica (plage documentee ~10 jours a plusieurs mois,
        #    valeur usuelle ~2 mois -- CDC/AJTMH/JAMA Dermatology).
        #  - 1/gamma_H = 9 mois : guerison spontanee/duree d'infectiosite --
        #    plage documentee 6 mois a plusieurs annees selon l'espece,
        #    L. tropica specifiquement associee a des lesions plus longues
        #    (5-12 mois, voire >=13 mois dans les formes resistantes) --
        #    9 mois est une valeur mediane prudente, pas le point le plus
        #    long documente pour L. tropica.
        # A noter : les valeurs PRECEDEMMENT apprises librement (2.65 et
        # 14.22 mois, cf. rapport) atterrissaient deja pres de cette plage
        # -- le fix ne change donc pas radicalement le comportement du
        # modele, il retire seulement 2 degres de liberte du probleme
        # d'identifiabilite.
        self.register_buffer("_sigmaH_fixed", torch.tensor(1.0 / 2.0))
        self.register_buffer("_gammaH_fixed", torch.tensor(1.0 / 9.0))

        # Modele d'observation explicite (integre en interne l'idee de
        # l'article de reference de l'encadrant -- Y ~ NegBin(rho*C, phi) --
        # SANS jeter l'architecture existante : obs_head represente
        # desormais l'incidence VRAIE C (pas directement le nombre de cas
        # rapportes), rho est la probabilite de rapportage EXPLICITE
        # (comparable a p_epi=0.76 du modele bayesien ICAR, mais apprise ici
        # depuis les cas eux-memes plutot que depuis l'evidence de presence),
        # phi absorbe la surdispersion (variance >> moyenne, cf. les zeros
        # tres majoritaires + quelques pics a 51 cas).
        self._rho_raw = nn.Parameter(torch.tensor(-2.0))    # sigmoid -> ~0.12 au depart
        self._phi_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(10.0)))))  # softplus -> 10

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
        return self._sigmaH_fixed

    @property
    def gamma_H(self):
        return self._gammaH_fixed

    @property
    def rho(self):
        """Probabilite de rapportage : fraction de l'incidence vraie (obs_rate)
        qui se traduit en cas effectivement rapportes par la surveillance."""
        return torch.sigmoid(self._rho_raw)

    @property
    def phi(self):
        """Dispersion du modele Binomial Negatif (plus petit = plus surdisperse)."""
        return nn.functional.softplus(self._phi_raw) + 1e-3

    def forward(self, t, lat, lon, temp, precip, humid, hist, prov_idx):
        prov_e = self.province_embed(prov_idx.squeeze(-1))
        x = torch.cat([t, lat, lon, temp, precip, humid, hist, prov_e], dim=1)
        h = self.trunk(x)

        human = torch.softmax(self.human_head(h), dim=1)
        S_H, E_H, I_H, R_H = human[:, 0:1], human[:, 1:2], human[:, 2:3], human[:, 3:4]

        vector = nn.functional.softplus(self.vector_head(h)) + 1e-6
        S_V, E_V, I_V = vector[:, 0:1], vector[:, 1:2], vector[:, 2:3]

        emergence = self.emergence_fn(torch.cat([temp, precip], dim=1))
        mu_V = self.mortality_fn(temp)
        sigma_V = self.eip_fn(temp)

        # Tete d'observation : sigma_H*E_H*N_pop force une fraction bornee
        # (softmax) a varier sur 4 ordres de grandeur (population 13 a
        # 548421 hab.) pour reproduire des comptages rares -- impossible a
        # calibrer precisement (teste : R2 hors-echantillon reste <-4 apres
        # 3 corrections successives de la perte/features/embedding, le
        # decodeur lui-meme etait le goulot). obs_rate represente ici
        # l'INCIDENCE VRAIE C (pas directement les cas rapportes) -- recoit
        # la meme representation h, librement expressive (pas de contrainte
        # softmax). La conversion C -> cas rapportes se fait via rho (proba
        # de rapportage explicite, cf. modele d'observation NegBin dans
        # train_pinn()) plutot que d'etre absorbee silencieusement dans
        # cette tete -- integration interne de l'idee de l'article de
        # reference (Y ~ NegBin(rho*C, phi)) sans jeter cette architecture.
        obs_rate = nn.functional.softplus(self.obs_head(h)) + 1e-6

        return {
            "S_H": S_H, "E_H": E_H, "I_H": I_H, "R_H": R_H,
            "S_V": S_V, "E_V": E_V, "I_V": I_V,
            "emergence": emergence, "mu_V": mu_V, "sigma_V": sigma_V,
            "obs_rate": obs_rate,
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


def physics_residual(model, t, lat, lon, temp, precip, humid, hist, prov_idx):
    """The true ODE residual: (autograd derivative) - (equation RHS) for
    each of the 7 SEIR-V equations. Driving this to zero during training is
    what enforces the mechanistic dynamics -- this is the part that makes
    the network a PINN rather than a plain regressor. `hist`/`prov_idx`
    (recent case history, province identity) are conditioning covariates
    held fixed w.r.t. t -- not differentiated, so they do not enter the ODE
    residual itself."""
    t = t.clone().requires_grad_(True)
    out = model(t, lat, lon, temp, precip, humid, hist, prov_idx)
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
    for col in ["population", "pop", "population_2024", "N_h", "pop_total"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].astype(float)
            n_missing = int(vals.isna().sum())
            if n_missing:
                median_pop = vals.median()
                logger.warning(f"Population reelle trouvee ('{col}') mais manquante pour {n_missing}/{len(vals)} "
                                f"lignes (communes non reconciliees a mun_pop.csv) -> mediane ({median_pop:.0f}) "
                                f"utilisee pour ces lignes seulement")
                vals = vals.fillna(median_pop)
            return vals.values, True
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

    # Historique local de cas (memes features que gbm_spatial_temporal.py,
    # #1-2 en importance apres province) -- calcule AVANT tout dropna pour
    # que les decalages temporels restent correctement espaces par commune,
    # comme dans build_commune_panel.py/gbm_spatial_temporal.py.
    data = data.sort_values(["commune_id", "annee", "mois"])
    data["cases_lag1"] = data.groupby("commune_id")["n_cas"].shift(1)
    data["cases_roll3"] = data.groupby("commune_id")["n_cas"].transform(lambda x: x.shift(1).rolling(3).mean())
    data["cases_roll6"] = data.groupby("commune_id")["n_cas"].transform(lambda x: x.shift(1).rolling(6).mean())
    for c in ["cases_lag1", "cases_roll3", "cases_roll6"]:
        data[c] = np.log1p(data[c].fillna(0.0))

    data = data.dropna(subset=["temp_moy", "precip_mm", "humidite_pct"])
    # 2016/2018/2019 n'ont AUCUN mois de diagnostic dans la source LCT ->
    # n_cas=NaN pour ces annees dans le panel (voir build_commune_panel.py),
    # pas 0. Avant ce fix elles etaient silencieusement remplies a 0, ce qui
    # injectait 3 annees de faux "zero cas" dans l'entrainement du PINN --
    # exactement le genre de bruit qui peut noyer un vrai signal climatique
    # (temperature) dans la perte d'entrainement. Exclues ici.
    n_before = len(data)
    data = data.dropna(subset=["n_cas"])
    logger.info(f"Lignes exclues (n_cas=NaN, annee sans mois de diagnostic) : {n_before - len(data)}")

    pop_values, has_real_pop = load_population(data)
    data["_pop"] = pop_values

    provinces = sorted(data["province"].dropna().unique().tolist())
    prov_to_idx = {p: i for i, p in enumerate(provinces)}
    data["province_idx"] = data["province"].map(prov_to_idx).fillna(0).astype(int)

    data["t_months"] = (data["annee"] - data["annee"].min()) * 12 + (data["mois"] - 1)

    feature_cols = ["t_months", "latitude", "longitude", "temp_moy", "precip_mm", "humidite_pct"]
    hist_cols = ["cases_lag1", "cases_roll3", "cases_roll6"]
    X = data[feature_cols].values.astype(np.float32)
    H = data[hist_cols].values.astype(np.float32)
    P = data["province_idx"].values.astype(np.int64).reshape(-1, 1)

    t_raw, lat_raw, lon_raw = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    temp_raw, precip_raw, humid_raw = X[:, 3:4], X[:, 4:5], X[:, 5:6]

    y_cases = data["n_cas"].values.astype(np.float32)
    N_h = data["_pop"].values.astype(np.float32)

    train_mask = (data["annee"] <= 2017).values
    test_mask = (data["annee"] >= 2018).values

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    model = SEIRVPINN(n_provinces=len(provinces))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Grille de temperature fixe pour la penalite de lissage (session
    # 2026-08-15) : le papier de reference audite penalise les oscillations
    # semaine-a-semaine de sa fonction climat->recrutement directement dans
    # le temps -- inapplicable tel quel ici car nos 3 sous-reseaux climat
    # sont des fonctions STATIQUES de (temp, precip), pas d'une sequence
    # temporelle. L'equivalent naturel pour notre architecture : penaliser
    # la courbure (derivee seconde discrete) de chaque fonction le long de
    # l'axe temperature -- une fonction lisse en temperature produit
    # automatiquement une trajectoire lisse dans le temps (T(t) varie
    # lentement/saisonnierement), sans supposer une forme particuliere
    # (monotone, unimodale...). Cible directement le symptome diagnostique
    # sur cette session (saisonnalite apprise en phase inversee, amplitude
    # quasi nulle -- une fonction erratique/bruitee plutot qu'une vraie
    # reponse biologique lisse).
    temp_grid = torch.linspace(float(temp_raw.min()), float(temp_raw.max()), 40).view(-1, 1)
    precip_grid = torch.full_like(temp_grid, float(precip_raw[train_mask].mean()))
    LAMBDA_SMOOTH = 0.02  # poids empirique, pas de recherche formelle (contrainte de temps) -- a documenter comme tel

    # A priori saisonnier informe par Bacaer & Guernaoui (session 2026-08-16,
    # piste identifiee dans le rapport comme non implementee jusqu'ici) :
    # emergence vectorielle nulle decembre-mai, active juin-novembre. Le
    # sous-reseau emergence_fn n'a pas de notion directe de "mois" (fonction
    # statique de temp/precip) -- l'a priori est donc impose en evaluant
    # emergence_fn sur la climatologie mensuelle moyenne (train, 2009-2017)
    # et en penalisant l'ecart de FORME (normalisee, pas d'echelle absolue
    # imposee) avec un gabarit saisonnier cible. Choix deliberes :
    #  - climatologie calculee sur train uniquement (pas de fuite test)
    #  - gabarit cible normalise (moyenne 0, variance 1 sur les 12 mois)
    #    compare a l'emergence normalisee de la meme facon -- le reseau garde
    #    la liberte de choisir l'amplitude reelle, seule la PHASE est
    #    contrainte.
    _train_df = data.loc[train_mask]
    _clim = _train_df.groupby("mois")[["temp_moy", "precip_mm"]].mean().reindex(range(1, 13))
    seasonal_temp = torch.tensor(_clim["temp_moy"].values, dtype=torch.float32).view(-1, 1)
    seasonal_precip = torch.tensor(_clim["precip_mm"].values, dtype=torch.float32).view(-1, 1)
    # Gabarit Bacaer-Guernaoui : 0 decembre-mai (mois 12,1,2,3,4,5), bosse
    # juin-novembre (mois 6-11) culminant en aout/septembre -- forme
    # qualitative de la reference, pas une courbe mesuree.
    _target_raw = np.array([0, 0, 0, 0, 0, 0.4, 0.7, 0.9, 1.0, 0.8, 0.5, 0], dtype=np.float32)
    _target_norm = (_target_raw - _target_raw.mean()) / (_target_raw.std() + 1e-8)
    seasonal_target = torch.tensor(_target_norm, dtype=torch.float32).view(-1, 1)
    LAMBDA_SEASONAL = 0.3  # poids empirique (contrainte de temps, pas de recherche formelle) -- assez fort pour influencer la phase sans ecraser l'ajustement aux donnees

    t_tr = to_t(t_raw[train_mask])
    lat_tr, lon_tr = to_t(lat_raw[train_mask]), to_t(lon_raw[train_mask])
    temp_tr, precip_tr, humid_tr = to_t(temp_raw[train_mask]), to_t(precip_raw[train_mask]), to_t(humid_raw[train_mask])
    hist_tr = to_t(H[train_mask])
    prov_tr = torch.tensor(P[train_mask], dtype=torch.long)
    y_tr = to_t(y_cases[train_mask].reshape(-1, 1))
    Nh_tr = to_t(N_h[train_mask].reshape(-1, 1))

    # Mini-batch plutot que full-batch : le residu physique demande 7 appels
    # torch.autograd.grad(create_graph=True) par pas -- sur les 159732 lignes
    # d'entrainement en full-batch, chaque epoque coutait plusieurs dizaines
    # de secondes (2000 epoques n'auraient jamais fini en temps raisonnable
    # sur cette machine). Le mini-batching est de toute facon une pratique
    # standard pour entrainer un PINN, pas juste un contournement de vitesse.
    n_epochs = 4000
    batch_size = min(4096, len(t_tr))
    n_train = t_tr.shape[0]
    logger.info(f"Entrainement PINN SEIR-V (autograd, {n_epochs} epoques, batch={batch_size}, "
                f"{n_train} points d'entrainement au total)...")
    for epoch in range(n_epochs):
        idx = torch.randint(0, n_train, (batch_size,))
        t_b = t_tr[idx]
        lat_b, lon_b = lat_tr[idx], lon_tr[idx]
        temp_b, precip_b, humid_b = temp_tr[idx], precip_tr[idx], humid_tr[idx]
        hist_b = hist_tr[idx]
        prov_b = prov_tr[idx]
        y_b, Nh_b = y_tr[idx], Nh_tr[idx]

        optimizer.zero_grad()

        phys_loss, out = physics_residual(model, t_b, lat_b, lon_b, temp_b, precip_b, humid_b, hist_b, prov_b)

        # Data loss branchee sur la tete d'observation (voir forward()) --
        # plus sur sigma_H*E_H*Nh_b, garde comme quantite mecaniste interne
        # (contrainte par la physique) mais plus comme decodeur de la
        # prediction de cas.
        #
        # MSE brute (version d'origine) etait catastrophique sur ces donnees
        # (90% de zeros, quelques pics jusqu'a 51 cas) : R2 hors-echantillon
        # de -184. Poisson pondere (version intermediaire) a corrige
        # l'essentiel (R2=+0.30) mais suppose Var(Y)=E[Y], alors que nos
        # comptages sont TRES surdisperses (beaucoup de zeros + quelques pics
        # extremes -> variance >> moyenne). Modele d'observation Binomial
        # Negatif ci-dessous (integre l'idee de l'article de reference de
        # l'encadrant) : incidence_vraie = obs_rate, cas_rapportes ~
        # NegBin(rho*incidence_vraie, phi) -- rho separe explicitement "combien
        # de maladie il y a vraiment" de "combien on en detecte", phi absorbe
        # la surdispersion que Poisson ne peut pas representer.
        C_true = out["obs_rate"]
        mean_reported = model.rho * C_true
        phi = model.phi
        eps = 1e-6
        mean_c = mean_reported.clamp(min=eps)
        nll = -(
            torch.lgamma(y_b + phi) - torch.lgamma(phi) - torch.lgamma(y_b + 1)
            + phi * torch.log(phi / (phi + mean_c) + eps)
            + y_b * torch.log(mean_c / (phi + mean_c) + eps)
        )
        weight = torch.where(y_b > 0, torch.tensor(20.0), torch.tensor(1.0))
        data_loss = torch.sum(weight * nll) / torch.sum(weight)

        # Penalite de lissage (session 2026-08-15, voir commentaire pres de
        # temp_grid) : courbure discrete des 3 fonctions climat sur la
        # grille de temperature fixe.
        em_grid = model.emergence_fn(torch.cat([temp_grid, precip_grid], dim=1))
        mu_grid = model.mortality_fn(temp_grid)
        eip_grid = model.eip_fn(temp_grid)
        smooth_loss = (
            torch.mean(torch.diff(em_grid, n=2, dim=0) ** 2)
            + torch.mean(torch.diff(mu_grid, n=2, dim=0) ** 2)
            + torch.mean(torch.diff(eip_grid, n=2, dim=0) ** 2)
        )

        # A priori saisonnier Bacaer-Guernaoui (voir commentaire pres de
        # seasonal_target) : emergence evaluee sur la climatologie mensuelle,
        # normalisee (forme seule, pas d'echelle), comparee au gabarit cible.
        em_seasonal = model.emergence_fn(torch.cat([seasonal_temp, seasonal_precip], dim=1))
        em_seasonal_norm = (em_seasonal - em_seasonal.mean()) / (em_seasonal.std() + 1e-6)
        seasonal_loss = torch.mean((em_seasonal_norm - seasonal_target) ** 2)

        loss = data_loss + 0.1 * phys_loss + LAMBDA_SMOOTH * smooth_loss + LAMBDA_SEASONAL * seasonal_loss
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            logger.info(f"  epoch {epoch:5d}  data_loss={data_loss.item():.4f}  "
                        f"phys_loss={phys_loss.item():.6f}  smooth_loss={smooth_loss.item():.6f}  "
                        f"seasonal_loss={seasonal_loss.item():.4f}")

    # ---------------------------------------------------------------- evaluate on the genuinely held-out 2018-2020 rows
    t_te = to_t(t_raw[test_mask]); lat_te, lon_te = to_t(lat_raw[test_mask]), to_t(lon_raw[test_mask])
    temp_te, precip_te, humid_te = to_t(temp_raw[test_mask]), to_t(precip_raw[test_mask]), to_t(humid_raw[test_mask])
    hist_te = to_t(H[test_mask])
    prov_te = torch.tensor(P[test_mask], dtype=torch.long)
    Nh_te = to_t(N_h[test_mask].reshape(-1, 1))

    with torch.no_grad():
        out_te = model(t_te, lat_te, lon_te, temp_te, precip_te, humid_te, hist_te, prov_te)
        C_true_te = out_te["obs_rate"]
        y_pred_pinn = (model.rho * C_true_te).numpy().flatten()
        C_true_te = C_true_te.numpy().flatten()
    y_pred_pinn = np.clip(y_pred_pinn, 0, None)

    reg_col = "region" if "region" in data.columns else [c for c in data.columns if "region" in c][0]
    test_df = data.loc[test_mask, ["commune", "province", reg_col, "annee", "mois", "n_cas"]].copy()
    test_df["region"] = test_df[reg_col]
    test_df["y_pred_pinn"] = y_pred_pinn
    test_df["incidence_vraie_inferee"] = C_true_te
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
    torch.save({
        "state_dict": model.state_dict(), "feature_cols": feature_cols,
        "hist_cols": hist_cols, "provinces": provinces,
    }, weights_path)
    logger.info(f"Poids du modele sauvegardes dans : {weights_path}")

    logger.info("\nParametres epidemiologiques (constantes partagees) :")
    logger.info(f"  taux de piqure a         = {model.a.item():.4f} / mois  [appris]")
    logger.info(f"  b_h (vecteur -> humain)  = {model.b_h.item():.4f}  [appris]")
    logger.info(f"  c_v (humain -> vecteur)  = {model.c_v.item():.4f}  [appris]")
    logger.info(f"  1/sigma_H (incubation)   = {1.0/model.sigma_H.item():.2f} mois  [FIXE, litterature LCT]")
    logger.info(f"  1/gamma_H (infectiosite) = {1.0/model.gamma_H.item():.2f} mois  [FIXE, litterature LCT]")
    logger.info(f"  rho (proba de rapportage)= {model.rho.item():.4f}  [appris]  <-- reponse a la sous-declaration")
    logger.info(f"  phi (dispersion NegBin)  = {model.phi.item():.2f}  [appris]")

    return model


if __name__ == "__main__":
    train_pinn()