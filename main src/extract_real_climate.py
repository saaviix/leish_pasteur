"""
Extraction du climat RÉEL ERA5 (2009-2021) + environnement statique, par commune
(plus proche voisin sur la grille) puis agrégation au niveau province x mois.

Remplace définitivement climat_placeholder.csv.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

UP = "/mnt/user-data/uploads"

communes = pd.read_csv(f"{UP}/communes_maroc_final.csv")
print(f"{len(communes)} communes, {communes['province'].nunique()} provinces, "
      f"{communes['region'].nunique()} régions")

# ---------------------------------------------------------------------------
# 1. Construire l'arbre KD sur la grille climat (fichiers annuels, 0.1°, 2009-2021)
# ---------------------------------------------------------------------------
ds_ref = xr.open_dataset(f"{UP}/era5_morocco_2009_monthly.nc")
lat_grid = ds_ref.latitude.values
lon_grid = ds_ref.longitude.values
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)  # (nlat, nlon)
grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
tree_climate = cKDTree(grid_points)

commune_coords = communes[["latitude", "longitude"]].values
dist, idx = tree_climate.query(commune_coords)
i_lat, i_lon = np.unravel_index(idx, lat_mesh.shape)
communes["_i_lat"], communes["_i_lon"] = i_lat, i_lon

# Diagnostic : communes hors de l'emprise de la grille (extrapolées au bord le plus proche)
out_of_bounds = (
    (communes["longitude"] < lon_grid.min()) | (communes["longitude"] > lon_grid.max()) |
    (communes["latitude"] < lat_grid.min()) | (communes["latitude"] > lat_grid.max())
)
print(f"Communes hors emprise grille climat (extrapolées au bord) : {out_of_bounds.sum()} "
      f"({100*out_of_bounds.mean():.1f}%) -- essentiellement Sahara/côte lointaine")

# ---------------------------------------------------------------------------
# 2. Extraire t2m, d2m, tp pour chaque commune, chaque année 2009-2021, 12 mois
# ---------------------------------------------------------------------------
years = range(2009, 2022)
rows = []
for year in years:
    ds = xr.open_dataset(f"{UP}/era5_morocco_{year}_monthly.nc")
    t2m = ds.t2m.values - 273.15          # K -> °C
    d2m = ds.d2m.values - 273.15          # K -> °C
    tp = ds.tp.values                     # m/jour (moyenne journalière du mois)
    days_in_month = ds.valid_time.dt.days_in_month.values

    # Formule de Magnus-Tetens pour l'humidité relative à partir de T et Td
    def es(temp_c):
        return 6.112 * np.exp((17.62 * temp_c) / (243.12 + temp_c))
    rh = 100 * es(d2m) / es(t2m)
    rh = np.clip(rh, 0, 100)

    for m_idx in range(12):
        month = m_idx + 1
        t_vals = t2m[m_idx, communes["_i_lat"], communes["_i_lon"]]
        rh_vals = rh[m_idx, communes["_i_lat"], communes["_i_lon"]]
        tp_vals = tp[m_idx, communes["_i_lat"], communes["_i_lon"]] * 1000 * days_in_month[m_idx]  # -> mm/mois

        rows.append(pd.DataFrame({
            "commune": communes["commune"].values,
            "province": communes["province"].values,
            "region": communes["region"].values,
            "annee": year,
            "mois": month,
            "temp_moy": t_vals,
            "humidite_pct": rh_vals,
            "precip_mm": tp_vals,
        }))
    print(f"  année {year} extraite")

climate_commune = pd.concat(rows, ignore_index=True)
climate_commune.to_csv("/home/claude/climat_reel_commune_mois.csv", index=False)
print(f"\nClimat commune x mois : {climate_commune.shape}")

# ---------------------------------------------------------------------------
# 3. Agrégation province x mois (moyenne des communes de la province)
# ---------------------------------------------------------------------------
climate_province = (climate_commune
                     .groupby(["region", "province", "annee", "mois"])
                     .agg(temp_moy=("temp_moy", "mean"),
                          humidite_pct=("humidite_pct", "mean"),
                          precip_mm=("precip_mm", "mean"))
                     .reset_index())
climate_province.to_csv("/home/claude/climat_reel_province_mois.csv", index=False)
print(f"Climat province x mois : {climate_province.shape}")

# Agrégation région x mois aussi (pour compat avec le pipeline déjà existant)
climate_region = (climate_commune
                   .groupby(["region", "annee", "mois"])
                   .agg(temp_moy=("temp_moy", "mean"),
                        humidite_pct=("humidite_pct", "mean"),
                        precip_mm=("precip_mm", "mean"))
                   .reset_index()
                   .rename(columns={"region": "Region"}))
climate_region.to_csv("/home/claude/climat_reel_region_mois.csv", index=False)
print(f"Climat région x mois : {climate_region.shape}")

print("\nAperçu (Marrakech-Safi devrait être chaud/sec, Tanger-Tétouan plus humide) :")
print(climate_region[climate_region["mois"] == 7].groupby("Region")[["temp_moy", "precip_mm", "humidite_pct"]].mean().round(1))
