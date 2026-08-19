"""
Zonage épidémiologique/écologique FINAL — basé sur le vrai climat ERA5 + élévation.

Contrairement aux tentatives précédentes (limitées aux provinces avec des cas),
ceci couvre les 76 provinces car le climat est disponible partout, y compris
dans le Sud sans cas ni documentation vectorielle. C'est la vraie réponse à
"division admin -> division épidémiologique/écologique basée sur climat".

Le statut vecteur et le volume de cas sont ensuite ajoutés en SURCOUCHE
(pas comme critère de clustering) pour la carte et l'interprétation.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

climate = pd.read_csv("/home/claude/climat_reel_province_mois.csv")
env = pd.read_csv("/home/claude/env_static_province.csv")
communes = pd.read_csv("/mnt/user-data/uploads/communes_maroc_final.csv")

# ---------------------------------------------------------------------------
# 1. Profil bioclimatique annuel par province
# ---------------------------------------------------------------------------
clim_annual = (climate.groupby(["region", "province"])
               .agg(temp_moy_an=("temp_moy", "mean"),
                    temp_max_mois=("temp_moy", "max"),
                    temp_min_mois=("temp_moy", "min"),
                    precip_totale_an=("precip_mm", "sum"),
                    humidite_moy_an=("humidite_pct", "mean"))
               .reset_index())
clim_annual["amplitude_thermique"] = clim_annual["temp_max_mois"] - clim_annual["temp_min_mois"]
# moyenne sur les années -> diviser precip totale annuelle cumulée sur 13 ans par 13
clim_annual["precip_totale_an"] = clim_annual["precip_totale_an"] / climate["annee"].nunique()

profile = clim_annual.merge(env, on=["region", "province"])

feat_cols = ["temp_moy_an", "amplitude_thermique", "precip_totale_an", "humidite_moy_an", "elevation_m"]
X = StandardScaler().fit_transform(profile[feat_cols])

# ---------------------------------------------------------------------------
# 2. K-means, sélection de k par silhouette (sur TOUTES les 76 provinces)
# ---------------------------------------------------------------------------
best_k, best_score, scores = None, -1, {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
    score = silhouette_score(X, km.labels_)
    scores[k] = score
    if score > best_score:
        best_k, best_score = k, score

print("Silhouette par k :", {k: round(v, 3) for k, v in scores.items()})
print(f"-> k retenu = {best_k} (silhouette={best_score:.3f})")

km_final = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(X)
profile["zone_bioclim"] = km_final.labels_

# ---------------------------------------------------------------------------
# 3. Surcouche : volume de cas + statut vecteur (pour interprétation/carte, pas pour le clustering)
# ---------------------------------------------------------------------------
cases = pd.read_csv("/mnt/user-data/uploads/leish_LCT.csv")
case_counts = cases.groupby("Province").size().rename("total_cas").reset_index()
profile = profile.merge(case_counts, left_on="province", right_on="Province", how="left").drop(columns=["Province"])
profile["total_cas"] = profile["total_cas"].fillna(0).astype(int)

vec = pd.read_csv("/mnt/user-data/uploads/phlebotomus_sergenti_par_province.csv")
vec.columns = [c.strip() for c in vec.columns]
def classify_vector(s):
    if pd.isna(s): return "non_documente"
    s = str(s).lower()
    if s.startswith("oui"): return "confirme"
    if "indirecte" in s or "ancienne" in s: return "indice_indirect"
    return "non_documente"
vec["statut_vecteur"] = vec["Statut_P_sergenti"].apply(classify_vector)
profile = profile.merge(vec[["Province", "statut_vecteur"]], left_on="province", right_on="Province", how="left").drop(columns=["Province"])
profile["statut_vecteur"] = profile["statut_vecteur"].fillna("non_documente")

# Centroïde géographique par province (moyenne des communes) pour la carte
centroids = communes.groupby("province")[["latitude", "longitude"]].mean().reset_index()
profile = profile.merge(centroids, on="province", how="left")

profile.to_csv("/home/claude/zone_epi_final.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Résumé
# ---------------------------------------------------------------------------
print(f"\n{len(profile)} provinces classées en {best_k} zones bioclimatiques\n")
for z, g in profile.groupby("zone_bioclim"):
    print(f"Zone {z} (n={len(g)}): temp={g['temp_moy_an'].mean():.1f}°C  "
          f"precip={g['precip_totale_an'].mean():.0f}mm/an  humid={g['humidite_moy_an'].mean():.0f}%  "
          f"elev={g['elevation_m'].mean():.0f}m  |  cas cumulés={g['total_cas'].sum()}  |  "
          f"vecteur confirmé dans {(g['statut_vecteur']=='confirme').sum()}/{len(g)} provinces")
