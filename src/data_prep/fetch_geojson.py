"""
fetch_geojson.py
================
Produit un GeoJSON des communes pour la carte choroplethe du dashboard.

Strategie :
  1. Si le reseau est disponible, essaie de recuperer les limites administratives
     communales via l'API Overpass (OpenStreetMap) en utilisant osm_id.
  2. Sinon (ou en complement), genere un fallback : un petit polygone carre
     (buffer) autour du point lat/lon de chaque commune, pour que la carte
     fonctionne quand meme en mode "polygones approximatifs".

Chaque feature porte properties.commune_id et properties.name, comme attendu
par le dashboard.

Sortie :
  outputs/processed/communes_morocco.geojson

Usage :
  python src/data_prep/fetch_geojson.py            # fallback rapide (carres)
  python src/data_prep/fetch_geojson.py --osm      # tente OSM (lent)
"""

import json
import sys
import time

import pandas as pd

import config

try:
    import requests
except ImportError:
    requests = None

OVERPASS = "https://overpass-api.de/api/interpreter"
BUFFER_DEG = 0.05  # ~5.5 km, taille du carre fallback


def square_feature(commune_id, name, lat, lon):
    d = BUFFER_DEG
    ring = [
        [lon - d, lat - d], [lon + d, lat - d],
        [lon + d, lat + d], [lon - d, lat + d],
        [lon - d, lat - d],
    ]
    return {
        "type": "Feature",
        "properties": {"commune_id": int(commune_id), "name": name},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def fetch_osm_polygon(osm_id):
    """Tente de recuperer une relation OSM en GeoJSON. None si echec."""
    if requests is None or pd.isna(osm_id):
        return None
    query = f"[out:json];rel({int(osm_id)});out geom;"
    try:
        r = requests.get(OVERPASS, params={"data": query}, timeout=40)
        r.raise_for_status()
        data = r.json()
        for el in data.get("elements", []):
            coords = []
            for m in el.get("members", []):
                if m.get("type") == "way" and "geometry" in m:
                    coords.append([[p["lon"], p["lat"]] for p in m["geometry"]])
            if coords:
                return {"type": "MultiLineString", "coordinates": coords}
    except Exception as e:
        print(f"[WARN] OSM {osm_id}: {e}")
    return None


def main() -> None:
    config.ensure_dirs()
    use_osm = "--osm" in sys.argv
    communes = pd.read_csv(config.COMMUNES_CSV)

    features = []
    for _, c in communes.iterrows():
        geom = None
        if use_osm:
            geom = fetch_osm_polygon(c.get("osm_id"))
            time.sleep(0.5)
        if geom:
            features.append({
                "type": "Feature",
                "properties": {"commune_id": int(c["id"]), "name": c["commune"]},
                "geometry": geom,
            })
        else:
            features.append(square_feature(c["id"], c["commune"], c["latitude"], c["longitude"]))

    fc = {"type": "FeatureCollection", "features": features}
    with open(config.GEOJSON, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"[OK] ecrit {config.GEOJSON} ({len(features)} polygones)"
          f"{' [OSM]' if use_osm else ' [fallback carres]'}")


if __name__ == "__main__":
    main()
