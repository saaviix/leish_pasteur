
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

UP = "/mnt/user-data/uploads"

vec = pd.read_csv(f"{UP}/phlebotomus_sergenti_par_province.csv")
vec.columns = [c.strip() for c in vec.columns]
zone_profile = pd.read_csv("/home/claude/zone_epi_final.csv")

def detection_strength(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if s.startswith("oui"): return 2
    if "indirecte" in s or "ancienne" in s: return 1
    return 0

def n_sources(row):
    src = row.get("Source", None)
    if pd.isna(src) or str(src).strip() == "-": return 0
    return len([s for s in str(src).split(";") if s.strip()])

vec["detection"] = vec["Statut_P_sergenti"].apply(detection_strength)
vec["effort"] = vec.apply(n_sources, axis=1)

data = zone_profile.merge(vec[["Province", "detection", "effort"]],
                            left_on="province", right_on="Province", how="left")
data["a_ete_etudiee"] = data["detection"].notna()

for col in ["temp_moy_an", "precip_totale_an", "humidite_moy_an", "elevation_m"]:
    data[f"{col}_z"] = (data[col] - data[col].mean()) / data[col].std()

# ---------------------------------------------------------------------------
# Matrice de distances (haversine, km) entre les 76 provinces -- pour le noyau du GP
# ---------------------------------------------------------------------------
def haversine_matrix(lon, lat):
    R = 6371
    lon, lat = np.radians(lon), np.radians(lat)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

D = haversine_matrix(data["longitude"].values, data["latitude"].values)
n = len(data)
print(f"{n} provinces, distances de {D[D>0].min():.0f} à {D.max():.0f} km")

obs_mask = data["a_ete_etudiee"].values
idx_obs = np.where(obs_mask)[0]
y_strong = (data.loc[obs_mask, "detection"] == 2).astype(int).values
effort_obs = data.loc[obs_mask, "effort"].clip(lower=0).values
X_psi = data[["temp_moy_an_z", "precip_totale_an_z", "humidite_moy_an_z", "elevation_m_z"]].values

# ---------------------------------------------------------------------------
# Modèle : logit(psi) = a0 + X.a_clim + f_spatial(coords),  f_spatial ~ GP(0, K_exp)
# Paramétrisation non centrée (f = L @ eta) pour un échantillonnage NUTS efficace
# ---------------------------------------------------------------------------
with pm.Model() as spatial_occ_model:
    a0 = pm.Normal("a0", 0, 2)
    a_clim = pm.Normal("a_clim", 0, 1, shape=4)

    rho = pm.HalfNormal("rho_km", 150)          # portée spatiale (km) -- combien loin la corrélation porte
    eta = pm.HalfNormal("eta", 1.5)             # écart-type du processus spatial

    K = eta**2 * pm.math.exp(-D / rho) + np.eye(n) * 1e-6
    L = pt.linalg.cholesky(K)
    f_raw = pm.Normal("f_raw", 0, 1, shape=n)
    f_spatial = pm.Deterministic("f_spatial", L @ f_raw)

    logit_psi = a0 + pm.math.dot(X_psi, a_clim) + f_spatial
    psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))

    b0 = pm.Normal("b0", -1, 1)
    b_effort = pm.HalfNormal("b_effort", 1)
    p_detect_given_present = pm.math.sigmoid(b0 + b_effort * effort_obs)
    p_obs_strong = psi[idx_obs] * p_detect_given_present
    pm.Bernoulli("y_strong", p=p_obs_strong, observed=y_strong)

    trace = pm.sample(1500, tune=1500, chains=4, cores=1, target_accept=0.95,
                       progressbar=False, random_seed=0)

print("\n" + az.summary(trace, var_names=["a0", "a_clim", "rho_km", "eta", "b0", "b_effort"]).to_string())

psi_samples = trace.posterior["psi"].values.reshape(-1, n)
data["psi_spatial_mean"] = psi_samples.mean(axis=0)
data["psi_spatial_low95"] = np.percentile(psi_samples, 2.5, axis=0)
data["psi_spatial_high95"] = np.percentile(psi_samples, 97.5, axis=0)

data[["region", "province", "a_ete_etudiee", "detection", "effort",
      "psi_spatial_mean", "psi_spatial_low95", "psi_spatial_high95"]].to_csv(
    "/home/claude/psi_par_province_spatial.csv", index=False)

rho_est = trace.posterior["rho_km"].values.mean()
print(f"\nPortée spatiale estimée (rho) = {rho_est:.0f} km -- distance à laquelle "
      f"la corrélation entre 2 provinces tombe à ~37% (1/e)")

# ---------------------------------------------------------------------------
# Comparaison climat-seul vs climat+spatial, pour les provinces jamais étudiées
# ---------------------------------------------------------------------------
psi_clim_only = pd.read_csv("/home/claude/psi_par_province.csv")
compare = data[["province", "a_ete_etudiee", "psi_spatial_mean"]].merge(
    psi_clim_only[["province", "psi_mean"]], on="province")
compare = compare.rename(columns={"psi_mean": "psi_climat_seul"})

print("\nProvinces jamais étudiées -- comparaison climat-seul vs climat+spatial :")
print(compare[~compare["a_ete_etudiee"]].sort_values("psi_spatial_mean", ascending=False).head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.scatter(compare["psi_climat_seul"], compare["psi_spatial_mean"],
           c=compare["a_ete_etudiee"].map({True: "green", False: "blue"}), alpha=0.6)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("ψ (climat seul)")
ax.set_ylabel("ψ (climat + spatial GP)")
ax.set_title("Effet de l'ajout du terme spatial\n(vert=documentée, bleu=jamais étudiée)")

ax = axes[1]
sc = ax.scatter(data["longitude"], data["latitude"], c=data["psi_spatial_mean"],
                 cmap="RdYlGn", s=60, edgecolors="k", linewidths=0.3, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="ψ (climat + spatial)")
ax.set_title("ψ spatial final, par province")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("/home/claude/occupancy_spatial_comparaison.png", dpi=130)
print("\nFigure sauvegardée : occupancy_spatial_comparaison.png")
