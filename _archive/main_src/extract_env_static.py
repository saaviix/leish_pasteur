"""
Extraction des variables environnementales statiques (élévation, végétation, sol)
par commune (plus proche voisin sur la grille, résolution différente du climat),
puis agrégation au niveau province.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

UP = "/mnt/user-data/uploads"
G = 9.80665  # constante gravitationnelle, pour convertir le géopotentiel en élévation

communes = pd.read_csv(f"{UP}/communes_maroc_final.csv")
ds = xr.open_dataset(f"{UP}/era5_env_static.nc")

lat_grid = ds.latitude.values
lon_grid = ds.longitude.values
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
tree = cKDTree(grid_points)

dist, idx = tree.query(communes[["latitude", "longitude"]].values)
i_lat, i_lon = np.unravel_index(idx, lat_mesh.shape)

elevation = (ds.z.isel(valid_time=0).values / G)[i_lat, i_lon]
lai_hv = ds.lai_hv.isel(valid_time=0).values[i_lat, i_lon]
lai_lv = ds.lai_lv.isel(valid_time=0).values[i_lat, i_lon]
cvh = ds.cvh.isel(valid_time=0).values[i_lat, i_lon]
cvl = ds.cvl.isel(valid_time=0).values[i_lat, i_lon]
slt = ds.slt.isel(valid_time=0).values[i_lat, i_lon]

env_commune = communes[["commune", "province", "region"]].copy()
env_commune["elevation_m"] = elevation
env_commune["lai_haute_veg"] = lai_hv
env_commune["lai_basse_veg"] = lai_lv
env_commune["couverture_haute_veg"] = cvh
env_commune["couverture_basse_veg"] = cvl
env_commune["type_sol"] = slt

env_commune.to_csv("/home/claude/env_static_commune.csv", index=False)

env_province = (env_commune.groupby(["region", "province"])
                 .agg(elevation_m=("elevation_m", "mean"),
                      lai_haute_veg=("lai_haute_veg", "mean"),
                      lai_basse_veg=("lai_basse_veg", "mean"),
                      couverture_haute_veg=("couverture_haute_veg", "mean"),
                      couverture_basse_veg=("couverture_basse_veg", "mean"),
                      type_sol=("type_sol", lambda x: x.mode().iloc[0]))
                 .reset_index())
env_province.to_csv("/home/claude/env_static_province.csv", index=False)

print(f"Env statique : {len(env_commune)} communes -> {len(env_province)} provinces")
print("\nAperçu élévation par région (Atlas devrait être plus haut) :")
print(env_commune.groupby("region")["elevation_m"].mean().round(0).sort_values(ascending=False))
