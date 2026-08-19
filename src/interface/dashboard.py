"""
dashboard.py
============
Tableau de bord LeishSergenti (Flask) : carte choroplethe du Maroc par commune
(climat, environnement) + couche de RISQUE = probabilite de presence de
P. sergenti inferee par le modele bayesien (par province).

Sources de donnees (toutes generees par le pipeline, dans outputs/processed) :
  climate_morocco.db        <- extract_climate.py
  environment_morocco.db    <- build_environment.py
  communes_morocco.geojson  <- fetch_geojson.py
  psergenti_posterior_presence.csv <- src/models/bayesian_occupancy.py
  zone_bioclim_province.csv        <- src/models/bioclimatic_zoning.py
  province_elevation_classes.csv   <- src/analysis/elevation_classification.py
  ensemble_recalibrated_predictions.csv <- src/models/robust_ensemble_recalibrated.py
  forecast_2025_2045_communes.csv  <- src/analysis/forecast_future.py (modele officiel
  forecast_2025_2045_provinces.csv    GBM+PINN stacke, R2=0.588 holdout 2018-2020)
  forecast_2025_2045_regions.csv

Usage :
  python src/interface/dashboard.py
  puis ouvrir http://localhost:5050
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

# importer config depuis src/data_prep
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

app = Flask(__name__)

DB_CLIMATE = str(config.CLIMATE_DB)
DB_ENV = str(config.ENV_DB)
GEOJSON_FILE = str(config.GEOJSON)
POSTERIOR_FILE = str(config.POSTERIOR_CSV)
TABLE = "climate"


def detect_cols():
    if not os.path.exists(DB_CLIMATE):
        return []
    conn = sqlite3.connect(DB_CLIMATE)
    try:
        cols = pd.read_sql_query(f"SELECT * FROM {TABLE} LIMIT 1", conn).columns.tolist()
    except Exception:
        cols = []
    conn.close()
    return cols


COLS = detect_cols()


def col(*candidates):
    for c in candidates:
        if c in COLS:
            return c
    return None


C_TEMP = col("temp_mean")
C_DEW = col("dewpoint")
C_HUM = col("humidity")
C_PRECIP = col("precipitation")
C_WIND = col("wind_speed")
C_RAD = col("radiation")
C_EVAP = col("evapotrans")
C_REGION = col("region")
C_PROV = col("province")
C_COM = col("commune")
C_YEAR = col("annee")
C_MONTH = col("mois")
C_CID = col("commune_id")

VARIABLES = {
    "temp_mean":     {"label": "Temperature moyenne", "col": C_TEMP,   "unit": "°C",  "ramp": "thermal"},
    "humidity":      {"label": "Humidite relative",    "col": C_HUM,    "unit": "%",   "ramp": "blue"},
    "precipitation": {"label": "Precipitations",       "col": C_PRECIP, "unit": "mm",  "ramp": "blue"},
    "wind_speed":    {"label": "Vitesse du vent",      "col": C_WIND,   "unit": "m/s", "ramp": "purple"},
    "radiation":     {"label": "Rayonnement",          "col": C_RAD,    "unit": "J/m2","ramp": "amber"},
    "evapotrans":    {"label": "Evapotranspiration",   "col": C_EVAP,   "unit": "mm",  "ramp": "amber"},
}
VARIABLES = {k: v for k, v in VARIABLES.items() if v["col"]}

ENV_VARIABLES = {
    "altitude_m":    {"label": "Altitude",       "unit": "m", "ramp": "thermal"},
    "aridity_index": {"label": "Indice aridite", "unit": "",  "ramp": "amber"},
}

RISK_VARIABLE = {
    "psi_mean": {"label": "Risque P. sergenti (proba presence)", "unit": "", "ramp": "thermal"},
}

ZONE_VARIABLES = {
    "zone_bioclim": {"label": "Zone bioclimatique (K-means climat+altitude)", "unit": "", "ramp": "purple"},
    "classe_altitude_rank": {"label": "Classe d'altitude (plaine -> haute montagne)", "unit": "", "ramp": "amber"},
}

PRED_VARIABLE = {
    "cas_predits": {"label": "Cas predits (ensemble recalibre, dernier millesime dispo)", "unit": "cas/mois", "ramp": "thermal"},
}

FUTURE_VARIABLE = {
    "cas_futurs": {"label": "Cas predits (projection annuelle, modele officiel)", "unit": "cas/an", "ramp": "ember"},
}


def qc(sql, params=()):
    conn = sqlite3.connect(DB_CLIMATE)
    df = pd.read_sql_query(sql, conn, params=list(params))
    conn.close()
    return df


def qe(sql, params=()):
    if not os.path.exists(DB_ENV):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_ENV)
    df = pd.read_sql_query(sql, conn, params=list(params))
    conn.close()
    return df


# --- posterior de risque par province ---
RISK_BY_PROVINCE = {}
if os.path.exists(POSTERIOR_FILE):
    _pdf = pd.read_csv(POSTERIOR_FILE)
    RISK_BY_PROVINCE = dict(zip(_pdf["province"], _pdf["psi_mean"]))
    print(f"[INFO] risque charge pour {len(RISK_BY_PROVINCE)} provinces")

# --- zone bioclimatique + classe d'altitude par province ---
ZONE_BY_PROVINCE = {}
ZONE_LABELS = {}
if config.ZONE_BIOCLIM_CSV.exists():
    _zdf = pd.read_csv(config.ZONE_BIOCLIM_CSV)
    ZONE_BY_PROVINCE["zone_bioclim"] = dict(zip(_zdf["province"], _zdf["zone_bioclim"]))
    print(f"[INFO] zonage bioclimatique charge pour {len(_zdf)} provinces")
if config.ELEVATION_CLASSES_CSV.exists():
    _edf = pd.read_csv(config.ELEVATION_CLASSES_CSV)
    ZONE_BY_PROVINCE["classe_altitude_rank"] = dict(zip(_edf["province"], _edf["classe_altitude_rank"]))
    ZONE_LABELS["classe_altitude_rank"] = dict(zip(_edf["classe_altitude_rank"], _edf["classe_altitude"]))
    print(f"[INFO] classes d'altitude chargees pour {len(_edf)} provinces")

# --- predictions ensemble recalibrees, dernier millesime dispo, par commune ---
PRED_BY_COMMUNE = {}
if config.ENSEMBLE_PRED_CSV.exists() and config.COMMUNES_CSV.exists():
    _pred = pd.read_csv(config.ENSEMBLE_PRED_CSV)
    _last_year = int(_pred["annee"].max())
    _pred = _pred[_pred["annee"] == _last_year].groupby(["commune", "province"], as_index=False)["y_pred_recalibre"].mean()
    _ref = pd.read_csv(config.COMMUNES_CSV)[["id", "commune", "province"]]
    _pred = _pred.merge(_ref, on=["commune", "province"], how="inner")
    PRED_BY_COMMUNE = dict(zip(_pred["id"], _pred["y_pred_recalibre"]))
    print(f"[INFO] predictions {_last_year} chargees pour {len(PRED_BY_COMMUNE)} communes")

# --- projections futures 2025-2045, modele officiel GBM+PINN (src/analysis/forecast_future.py) ---
FUTURE_BY_COMMUNE = {}   # {annee: {commune_id: cas_predits_annuel}}
FUTURE_YEARS = []
FUTURE_TREND_REGIONS = pd.DataFrame()
if config.FORECAST_COMMUNES_CSV.exists() and config.COMMUNES_CSV.exists():
    _fut = pd.read_csv(config.FORECAST_COMMUNES_CSV)
    _fut_annual = _fut.groupby(["commune", "province", "annee"], as_index=False)["cas_predits"].sum()
    _ref2 = pd.read_csv(config.COMMUNES_CSV)[["id", "commune", "province"]]
    _fut_annual = _fut_annual.merge(_ref2, on=["commune", "province"], how="inner")
    FUTURE_YEARS = sorted(int(y) for y in _fut_annual["annee"].unique())
    for _y in FUTURE_YEARS:
        _sub = _fut_annual[_fut_annual["annee"] == _y]
        FUTURE_BY_COMMUNE[_y] = dict(zip(_sub["id"], _sub["cas_predits"]))
    print(f"[INFO] projections futures {FUTURE_YEARS[0]}-{FUTURE_YEARS[-1]} chargees pour {_fut_annual['id'].nunique()} communes")

if config.FORECAST_REGIONS_CSV.exists():
    FUTURE_TREND_REGIONS = pd.read_csv(config.FORECAST_REGIONS_CSV)
    # variance implicite par ligne region x annee x mois, retro-deduite de l'IC95%
    # deja agrege correctement (cf. forecast_future.py::aggregate_with_ci) -- permet
    # de re-agreger par annee (across mois) sans re-sommer les bornes directement.
    FUTURE_TREND_REGIONS["_var"] = (
        (FUTURE_TREND_REGIONS["ci_upper_95"] - FUTURE_TREND_REGIONS["ci_lower_95"]) / (2 * 1.96)
    ) ** 2

# --- metriques du modele officiel, pour badge d'entete (jamais codees en dur : lues
# depuis l'artefact persiste par gbm_pinn_stacked.py) ---
MODEL_METRICS = {}
_model_path = config.PROCESSED / "gbm_model.joblib"
if _model_path.exists():
    try:
        import joblib
        MODEL_METRICS = {k: round(float(v), 4) for k, v in joblib.load(_model_path).get("metrics", {}).items()}
        print(f"[INFO] modele officiel charge : R2={MODEL_METRICS.get('R2')}")
    except Exception as e:
        print(f"[WARN] metriques du modele officiel non chargees ({e})")

# --- R2 du modele officiel a plusieurs resolutions spatio-temporelles, calcule
# en direct depuis les vraies predictions du holdout 2018-2020 (jamais code en
# dur) -- meme calcul que myvisuals.py::viz 21, reutilise ici pour le dashboard.
MODEL_RES_METRICS = {}
_gbm_pred_path = config.PROCESSED / "gbm_predictions_2018_2020.csv"
if _gbm_pred_path.exists():
    try:
        from sklearn.metrics import r2_score
        _gp = pd.read_csv(_gbm_pred_path)
        _g_annee = _gp.groupby(["commune", "province", "annee"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
        _g_prov_mois = _gp.groupby(["province", "annee", "mois"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
        _g_prov_annee = _gp.groupby(["province", "annee"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
        MODEL_RES_METRICS = {
            "commune_mois": round(float(r2_score(_gp["n_cas"], _gp["y_pred_gbm"])), 4),
            "province_mois": round(float(r2_score(_g_prov_mois["y"], _g_prov_mois["p"])), 4),
            "commune_annee": round(float(r2_score(_g_annee["y"], _g_annee["p"])), 4),
            "province_annee": round(float(r2_score(_g_prov_annee["y"], _g_prov_annee["p"])), 4),
        }
        print(f"[INFO] R2 multi-resolution : {MODEL_RES_METRICS}")
    except Exception as e:
        print(f"[WARN] R2 multi-resolution non calcule ({e})")

# --- geojson ---
GEOJSON_DATA = None
GEOJSON_AVAILABLE = os.path.exists(GEOJSON_FILE)
if GEOJSON_AVAILABLE:
    with open(GEOJSON_FILE, encoding="utf-8") as f:
        GEOJSON_DATA = json.load(f)
    print(f"[INFO] GeoJSON : {len(GEOJSON_DATA['features'])} polygones")
else:
    print(f"[WARN] {GEOJSON_FILE} introuvable -> mode points (cercles)")


def build_points_fallback():
    df = qc(f"SELECT DISTINCT {C_CID} AS commune_id, {C_COM} AS commune, latitude, longitude FROM {TABLE}")
    return df.to_dict(orient="records")


@app.route("/api/regions")
def api_regions():
    df = qc(f"SELECT DISTINCT region FROM {TABLE} WHERE region IS NOT NULL ORDER BY region")
    return jsonify(df["region"].tolist())


@app.route("/api/provinces")
def api_provinces():
    r = request.args.get("region", "")
    df = qc(f"SELECT DISTINCT province FROM {TABLE} WHERE region=? ORDER BY province", [r])
    return jsonify(df["province"].tolist())


@app.route("/api/communes")
def api_communes():
    p = request.args.get("province", "")
    df = qc(f"SELECT DISTINCT commune FROM {TABLE} WHERE province=? ORDER BY commune", [p])
    return jsonify(df["commune"].tolist())


@app.route("/api/years")
def api_years():
    df = qc(f"SELECT DISTINCT annee FROM {TABLE} ORDER BY annee")
    return jsonify(df["annee"].tolist())


@app.route("/api/variables")
def api_variables():
    return jsonify({
        "climate": VARIABLES, "environment": ENV_VARIABLES, "risk": RISK_VARIABLE,
        "zone": {k: v for k, v in ZONE_VARIABLES.items() if k in ZONE_BY_PROVINCE},
        "prediction": PRED_VARIABLE if PRED_BY_COMMUNE else {},
        "future": FUTURE_VARIABLE if FUTURE_BY_COMMUNE else {},
    })


@app.route("/api/future_years")
def api_future_years():
    return jsonify(FUTURE_YEARS)


@app.route("/api/future_trend")
def api_future_trend():
    """Serie annuelle 2025-2045 (national ou filtre par region), agregee
    correctement a partir des variances implicites par ligne (cf. chargement
    de FUTURE_TREND_REGIONS ci-dessus) -- pas une somme naive des IC deja
    agreges au niveau mois, qui surestimerait l'incertitude."""
    if FUTURE_TREND_REGIONS.empty:
        return jsonify({"years": [], "predicted": [], "ci_lower": [], "ci_upper": []})
    region = request.args.get("region", "")
    df = FUTURE_TREND_REGIONS
    if region:
        df = df[df["region"] == region]
    annual = df.groupby("annee").agg(cas_predits=("cas_predits", "sum"), _var=("_var", "sum")).reset_index().sort_values("annee")
    ci_lower = np.clip(annual["cas_predits"] - 1.96 * np.sqrt(annual["_var"]), 0, None)
    ci_upper = annual["cas_predits"] + 1.96 * np.sqrt(annual["_var"])
    return jsonify({
        "years": annual["annee"].astype(int).tolist(),
        "predicted": [round(float(v), 1) for v in annual["cas_predits"]],
        "ci_lower": [round(float(v), 1) for v in ci_lower],
        "ci_upper": [round(float(v), 1) for v in ci_upper],
    })


@app.route("/api/map_data")
def api_map_data():
    var_key = request.args.get("variable", "temp_mean")
    source = request.args.get("source", "climate")
    annee = request.args.get("annee", "")
    mois = request.args.get("mois", "")
    region = request.args.get("region", "")

    if source == "risk":
        # risque par province -> propage a chaque commune de la province
        sql = f"SELECT {C_CID} AS commune_id, {C_PROV} AS province FROM {TABLE}"
        params = []
        if region:
            sql += f" WHERE {C_REGION}=?"
            params.append(region)
        df = qc(sql, params)
        df["value"] = df["province"].map(RISK_BY_PROVINCE)
        df = df.dropna(subset=["value"])
    elif source == "zone":
        if var_key not in ZONE_BY_PROVINCE:
            return jsonify({"error": "unknown zone variable"}), 400
        sql = f"SELECT {C_CID} AS commune_id, {C_PROV} AS province FROM {TABLE}"
        params = []
        if region:
            sql += f" WHERE {C_REGION}=?"
            params.append(region)
        df = qc(sql, params)
        df["value"] = df["province"].map(ZONE_BY_PROVINCE[var_key])
        df = df.dropna(subset=["value"])
    elif source == "prediction":
        sql = f"SELECT {C_CID} AS commune_id, {C_PROV} AS province FROM {TABLE}"
        params = []
        if region:
            sql += f" WHERE {C_REGION}=?"
            params.append(region)
        df = qc(sql, params)
        df["value"] = df["commune_id"].map(PRED_BY_COMMUNE)
        df = df.dropna(subset=["value"])
    elif source == "future":
        fut_year = int(annee) if annee else (FUTURE_YEARS[0] if FUTURE_YEARS else 2025)
        year_map = FUTURE_BY_COMMUNE.get(fut_year, {})
        sql = f"SELECT {C_CID} AS commune_id, {C_PROV} AS province FROM {TABLE}"
        params = []
        if region:
            sql += f" WHERE {C_REGION}=?"
            params.append(region)
        df = qc(sql, params)
        df["value"] = df["commune_id"].map(year_map)
        df = df.dropna(subset=["value"])
    elif source == "environment":
        if var_key not in ENV_VARIABLES:
            return jsonify({"error": "unknown env variable"}), 400
        sql = f"SELECT commune_id, commune, region, {var_key} AS value FROM environment WHERE {var_key} IS NOT NULL"
        params = []
        if region:
            sql += " AND region=?"
            params.append(region)
        df = qe(sql, params)
    else:
        if var_key not in VARIABLES:
            return jsonify({"error": "unknown climate variable"}), 400
        dbcol = VARIABLES[var_key]["col"]
        sql = f"SELECT {C_CID} AS commune_id, {C_COM} AS commune, {C_REGION} AS region, AVG({dbcol}) AS value FROM {TABLE} WHERE {dbcol} IS NOT NULL"
        params = []
        if annee:
            sql += f" AND {C_YEAR}=?"; params.append(int(annee))
        if mois:
            sql += f" AND {C_MONTH}=?"; params.append(int(mois))
        if region:
            sql += f" AND {C_REGION}=?"; params.append(region)
        sql += f" GROUP BY {C_CID}"
        df = qc(sql, params)

    if df.empty:
        return jsonify({"values": {}, "min": 0, "max": 0, "geojson_available": GEOJSON_AVAILABLE})

    values = {
        str(int(row["commune_id"])): round(float(row["value"]), 3)
        for _, row in df.iterrows()
        if pd.notna(row["value"]) and pd.notna(row["commune_id"])
    }
    vmin = min(values.values()) if values else 0
    vmax = max(values.values()) if values else 0

    response = {"values": values, "min": vmin, "max": vmax, "geojson_available": GEOJSON_AVAILABLE}
    if not GEOJSON_AVAILABLE:
        points = build_points_fallback()
        response["points"] = [p for p in points if str(p["commune_id"]) in values]
    return jsonify(response)


@app.route("/api/geojson")
def api_geojson():
    if not GEOJSON_AVAILABLE:
        return jsonify({"type": "FeatureCollection", "features": []})
    return jsonify(GEOJSON_DATA)


@app.route("/api/stats")
def api_stats():
    region = request.args.get("region", "")
    province = request.args.get("province", "")
    commune = request.args.get("commune", "")
    if not region or not C_YEAR or not C_TEMP:
        return jsonify([])
    aggs = [f"AVG({C_TEMP}) AS temp_mean"]
    if C_HUM: aggs.append(f"AVG({C_HUM}) AS humidity")
    if C_WIND: aggs.append(f"AVG({C_WIND}) AS wind_speed")
    if C_PRECIP: aggs.append(f"AVG({C_PRECIP}) AS precipitation")
    sql = f"SELECT {C_YEAR} AS annee, {', '.join(aggs)}, COUNT(DISTINCT {C_COM}) AS n_communes FROM {TABLE} WHERE {C_REGION}=?"
    params = [region]
    if province and C_PROV: sql += f" AND {C_PROV}=?"; params.append(province)
    if commune and C_COM: sql += f" AND {C_COM}=?"; params.append(commune)
    sql += f" GROUP BY {C_YEAR} ORDER BY {C_YEAR}"
    df = qc(sql, params)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))


@app.route("/api/monthly_profile")
def api_monthly_profile():
    region = request.args.get("region", "")
    province = request.args.get("province", "")
    commune = request.args.get("commune", "")
    annee = request.args.get("annee", "")
    if not region or not C_MONTH:
        return jsonify([])
    aggs = []
    if C_TEMP: aggs.append(f"AVG({C_TEMP}) AS temp_mean")
    if C_HUM: aggs.append(f"AVG({C_HUM}) AS humidity")
    if C_PRECIP: aggs.append(f"AVG({C_PRECIP}) AS precipitation")
    sql = f"SELECT {C_MONTH} AS mois, {', '.join(aggs)} FROM {TABLE} WHERE {C_REGION}=?"
    params = [region]
    if province and C_PROV: sql += f" AND {C_PROV}=?"; params.append(province)
    if commune and C_COM: sql += f" AND {C_COM}=?"; params.append(commune)
    if annee and C_YEAR: sql += f" AND {C_YEAR}=?"; params.append(int(annee))
    sql += f" GROUP BY {C_MONTH} ORDER BY {C_MONTH}"
    df = qc(sql, params)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))


@app.route("/api/db_info")
def api_db_info():
    conn = sqlite3.connect(DB_CLIMATE)
    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    coms = conn.execute(f"SELECT COUNT(DISTINCT commune) FROM {TABLE}").fetchone()[0]
    ymin = conn.execute(f"SELECT MIN(annee) FROM {TABLE}").fetchone()[0]
    ymax = conn.execute(f"SELECT MAX(annee) FROM {TABLE}").fetchone()[0]
    conn.close()
    return jsonify({
        "total_rows": total, "communes": coms, "year_min": ymin, "year_max": ymax,
        "geojson_ok": GEOJSON_AVAILABLE,
        "geojson_features": len(GEOJSON_DATA["features"]) if GEOJSON_DATA else 0,
        "risk_ok": bool(RISK_BY_PROVINCE),
        "future_ok": bool(FUTURE_BY_COMMUNE),
        "future_year_min": FUTURE_YEARS[0] if FUTURE_YEARS else None,
        "future_year_max": FUTURE_YEARS[-1] if FUTURE_YEARS else None,
        "model_r2": MODEL_METRICS.get("R2"),
        "model_res": MODEL_RES_METRICS,
    })


HTML = r"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LeishSergenti — Institut Pasteur du Maroc</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{
  --blue:#0E4C86;--blue-deep:#082D52;--blue-pale:#E7F0FA;
  --gold:#D89A3E;--ember:#B5502E;
  --paper:#F5F7FA;--surface:#FFFFFF;
  --ink:#0F1F30;--ink-2:#4C6076;--ink-3:#8398AC;
  --line:#E1E7EE;--line-2:#CBD6E2;
  --good:#1E8E5A;--warn:#C98A2C;--critical:#C1443B;
  --r:10px;--r-lg:14px;
  --shadow-sm:0 1px 2px rgba(15,31,48,.07);
  --shadow-md:0 8px 24px -8px rgba(15,31,48,.20);
  --font-display:'Fraunces',Georgia,serif;
  --font-body:'IBM Plex Sans',system-ui,sans-serif;
  --font-mono:'IBM Plex Mono',ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --blue:#5A9AD6;--blue-deep:#0B2E4F;--blue-pale:#152E46;
    --gold:#E4B369;--ember:#D6926F;
    --paper:#0B1420;--surface:#121E2E;
    --ink:#EAF1F8;--ink-2:#A9BBCC;--ink-3:#6E8299;
    --line:#22344A;--line-2:#324A65;
    --good:#4CBE86;--warn:#E0A94F;--critical:#E17A72;
    --shadow-sm:0 1px 2px rgba(0,0,0,.35);
    --shadow-md:0 10px 28px -8px rgba(0,0,0,.6);
  }
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:var(--font-body);-webkit-font-smoothing:antialiased}
@media (prefers-reduced-motion: reduce){*{animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important}}

.masthead{position:relative;overflow:hidden;background:linear-gradient(120deg,#0A3D6E,#2E77B5 45%,#4A93CE 55%,#0A3D6E);background-size:220% 220%;animation:sheen 18s ease-in-out infinite;border-bottom:3px solid #fff}
@keyframes sheen{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.masthead-photo{position:absolute;inset:0;background-image:url('/static/pasteur_campus.jpg');background-size:cover;background-position:center 58%;opacity:.24;mix-blend-mode:luminosity}
.masthead-scrim{position:absolute;inset:0;background:linear-gradient(175deg,rgba(8,45,82,.32),rgba(8,45,82,.88))}
.masthead-inner{position:relative;max-width:1440px;margin:0 auto;padding:24px 28px 18px;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:18px}
.brand{display:flex;align-items:center;gap:14px}
.brand-logo{height:50px;width:50px;border-radius:12px;background:#fff;padding:6px;box-shadow:var(--shadow-md);object-fit:contain;flex:none}
.brand-org{font-family:var(--font-body);font-size:13px;font-weight:600;color:#fff;letter-spacing:.01em}
.brand-dept{font-family:var(--font-body);font-size:10.5px;color:rgba(255,255,255,.72);margin-top:3px;max-width:320px;line-height:1.4}
.brand-title{text-align:right}
.brand-title h1{font-family:var(--font-display);font-weight:600;font-size:27px;color:#fff;letter-spacing:.005em;text-wrap:balance}
.brand-title h1 .dot{color:var(--gold)}
.brand-title p{font-family:var(--font-body);font-size:11.5px;color:rgba(255,255,255,.72);margin-top:3px}

.app{display:grid;grid-template-columns:280px 1fr;max-width:1440px;margin:0 auto;min-height:calc(100vh - 128px)}
aside{background:var(--surface);border-right:1px solid var(--line);padding:18px 14px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.s-title{font-family:var(--font-body);font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--line);font-weight:600}
.var-section-sub{font-size:10.5px;color:var(--ink-2);line-height:1.45;margin:-4px 0 10px;padding:8px 10px;background:var(--blue-pale);border-radius:8px}
.fg{display:flex;flex-direction:column;gap:5px;margin-bottom:8px}
.fg label{font-size:11px;color:var(--ink-2)}
select{width:100%;background:var(--paper);border:1px solid var(--line-2);color:var(--ink);padding:8px 10px;border-radius:var(--r);font-size:12px;font-family:var(--font-body);transition:border-color .15s,box-shadow .15s}
select:hover{border-color:var(--blue)}
select:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px var(--blue-pale)}
.var-btn{background:var(--paper);border:1px solid var(--line-2);color:var(--ink-2);padding:9px 11px;border-radius:var(--r);font-size:11.5px;cursor:pointer;display:flex;justify-content:space-between;margin-bottom:5px;transition:border-color .15s,transform .15s,color .15s,background .15s}
.var-btn:hover{border-color:var(--blue);transform:translateX(2px)}
.var-btn.on{background:var(--blue-pale);border-color:var(--blue);color:var(--blue);font-weight:600}
.source-toggle{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}
.source-btn{flex:1;min-width:74px;background:var(--paper);border:1px solid var(--line-2);color:var(--ink-2);padding:7px;border-radius:var(--r);font-size:10px;cursor:pointer;text-align:center;transition:all .15s;font-family:var(--font-body)}
.source-btn:hover{border-color:var(--blue);color:var(--blue)}
.source-btn.on{background:var(--blue);border-color:var(--blue);color:#fff;box-shadow:var(--shadow-sm)}
#src-future.on{background:var(--ember);border-color:var(--ember)}
main{padding:22px 26px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}
.panel,.map-panel,.future-panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:18px;box-shadow:var(--shadow-sm)}
#map{width:100%;height:520px;border-radius:10px;background:#E7F0FA}
.legend-bar{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--paper);border:1px solid var(--line-2);border-radius:var(--r);margin-top:10px}
.legend-grad{flex:1;height:8px;border-radius:5px}
.legend-val{font-family:var(--font-mono);font-size:10px;color:var(--ink-2);min-width:42px}
.panel{transition:box-shadow .2s}
.cw{position:relative;height:210px;background:#fff;border:1px solid #E1E7EE;border-radius:var(--r);padding:10px}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.map-title{font-family:var(--font-body);font-size:13px;font-weight:600;color:var(--ink)}
.future-panel{border-top:3px solid var(--ember)}
.res-panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:18px;box-shadow:var(--shadow-sm);border-top:3px solid var(--gold)}
.res-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
.res-card{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);padding:16px 12px;text-align:center;transition:transform .2s,border-color .2s,box-shadow .2s}
.res-card:hover{transform:translateY(-3px);border-color:var(--blue);box-shadow:var(--shadow-md)}
.res-v{font-family:var(--font-mono);font-size:26px;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.res-l{font-size:10.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.05em;margin-top:6px}
.res-best{background:var(--blue-pale);border-color:var(--blue)}
@media (max-width:820px){.res-grid{grid-template-columns:1fr 1fr}}
.future-head{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:4px}
.future-sub{font-size:11px;color:var(--ink-3)}
.future-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.fstat{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px;transition:border-color .2s}
.fstat:hover{border-color:var(--ember)}
.fstat .fv{font-family:var(--font-mono);font-size:20px;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.fstat .fl{font-size:9.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;margin-top:3px}
.cw-lg{position:relative;height:260px;background:#fff;border:1px solid #E1E7EE;border-radius:var(--r);padding:12px}
.hidden{display:none !important}
footer{max-width:1440px;margin:0 auto;padding:20px 28px 34px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;color:var(--ink-3);font-size:10.5px;border-top:1px solid var(--line)}
footer strong{color:var(--ink-2);font-weight:600}
.reveal{opacity:0;animation:revealUp .65s cubic-bezier(.16,1,.3,1) forwards}
@keyframes revealUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* leaflet chrome, restyled to match the token system instead of default white/black */
.leaflet-control-zoom{border:1px solid #E1E7EE!important;box-shadow:0 1px 2px rgba(15,31,48,.07)!important;border-radius:8px!important;overflow:hidden}
.leaflet-control-zoom a{background:#fff!important;color:#4C6076!important;border-color:#E1E7EE!important;font-family:var(--font-body)!important}
.leaflet-control-zoom a:hover{background:#E7F0FA!important;color:#082D52!important}
.leaflet-control-attribution{background:rgba(255,255,255,.8)!important;color:#4C6076!important;font-family:var(--font-body)!important;font-size:9.5px!important;border-radius:6px 0 0 0!important}
.leaflet-control-attribution a{color:#0E4C86!important}
.leaflet-tooltip.lf-tip{background:#fff;color:#0F1F30;border:1px solid #CBD6E2;border-radius:8px;box-shadow:0 8px 24px -8px rgba(15,31,48,.2);font-family:var(--font-body);font-size:11.5px;padding:5px 10px}
.leaflet-tooltip.lf-tip::before{display:none}
.leaflet-popup.lf-pop-wrap .leaflet-popup-content-wrapper{background:#fff;color:#0F1F30;border-radius:12px;box-shadow:0 10px 28px -8px rgba(15,31,48,.28);padding:2px}
.leaflet-popup.lf-pop-wrap .leaflet-popup-tip{background:#fff}
.lf-pop{font-family:var(--font-body);padding:6px 8px;min-width:150px}
.lf-pop-name{font-size:13px;font-weight:600;color:#0F1F30;margin-bottom:4px}
.lf-pop-val{font-family:var(--font-mono);font-size:19px;font-weight:600;color:#0E4C86}
.lf-pop-val span{font-family:var(--font-body);font-size:11px;font-weight:400;color:#8398AC;margin-left:3px}
.lf-pop-meta{font-size:10.5px;color:#8398AC;margin-top:4px;border-top:1px solid #E1E7EE;padding-top:4px}
</style></head><body>
<header class="masthead">
  <div class="masthead-photo" aria-hidden="true"></div>
  <div class="masthead-scrim" aria-hidden="true"></div>
  <div class="masthead-inner">
    <div class="brand">
      <img class="brand-logo" src="/static/pasteur_logo.png" alt="Institut Pasteur du Maroc" onerror="this.style.display='none'"/>
      <div>
        <div class="brand-org">Institut Pasteur du Maroc</div>
        <div class="brand-dept">Département de recherche — Service de Parasitologie et Maladies Vectorielles</div>
      </div>
    </div>
    <div class="brand-title">
      <h1>Projet Pasteur LCT<span class="dot">.</span></h1>
      <p>Surveillance et prédiction spatio-temporelle de la leishmaniose cutanée au Maroc</p>
    </div>
  </div>
</header>
<div class="app">
<aside class="reveal" style="animation-delay:.05s">
  <div><div class="s-title">Source</div>
    <div class="source-toggle">
      <div class="source-btn on" id="src-climate" onclick="setSource('climate')">Climat</div>
      <div class="source-btn" id="src-env" onclick="setSource('environment')">Env.</div>
      <div class="source-btn" id="src-risk" onclick="setSource('risk')">Risque</div>
      <div class="source-btn" id="src-zone" onclick="setSource('zone')">Zone</div>
      <div class="source-btn" id="src-prediction" onclick="setSource('prediction')">Prediction</div>
      <div class="source-btn" id="src-future" onclick="setSource('future')">Futur</div>
    </div>
  </div>
  <div><div class="s-title" id="var-section-title">Variable</div><div class="var-section-sub hidden" id="var-section-sub"></div><div id="var-grid"></div></div>
  <div class="fg hidden" id="future-year-group"><label>Annee de projection</label><select id="sel-future-year"></select></div>
  <div id="climate-time-group">
    <div class="s-title">Periode (climat)</div>
    <div class="fg"><label>Mois</label><select id="sel-mois"><option value="">Annee entiere (moyenne 12 mois)</option></select></div>
    <div class="fg"><label>Annee</label><select id="sel-annee"><option value="">Moyenne 2009-2021</option></select></div>
  </div>
  <div><div class="s-title">Filtrer</div>
    <div class="fg"><label>Region</label><select id="sel-region"><option value="">Tout le Maroc</option></select></div>
    <div class="fg"><label>Province</label><select id="sel-province" disabled><option value="">Toutes</option></select></div>
    <div class="fg"><label>Commune</label><select id="sel-commune" disabled><option value="">Toutes</option></select></div>
  </div>
</aside>
<main>
  <div class="res-panel reveal" style="animation-delay:.04s">
    <div class="future-head">
      <div class="map-title">Précision du modèle selon la résolution d'analyse</div>
    </div>
    <div class="res-grid">
      <div class="res-card"><div class="res-v" id="res-commune-mois">-</div><div class="res-l">Commune × mois</div></div>
      <div class="res-card"><div class="res-v" id="res-province-mois">-</div><div class="res-l">Province × mois</div></div>
      <div class="res-card"><div class="res-v" id="res-commune-annee">-</div><div class="res-l">Commune × année</div></div>
      <div class="res-card res-best"><div class="res-v" id="res-province-annee">-</div><div class="res-l">Province × année</div></div>
    </div>
  </div>
  <div class="map-panel reveal" style="animation-delay:.1s">
    <div class="map-title" id="map-title">Carte</div>
    <div id="map"></div>
    <div class="legend-bar"><span class="legend-val" id="leg-min">-</span><div class="legend-grad" id="leg-grad"></div><span class="legend-val" id="leg-max">-</span></div>
  </div>
  <div class="future-panel reveal" style="animation-delay:.16s">
    <div class="future-head">
      <div class="map-title">Projection des cas predits (modele officiel GBM+PINN)</div>
      <div class="future-sub" id="future-region-label">Maroc entier</div>
    </div>
    <div class="future-sub">Serie 2025-2045 (horizon 20 ans, dont les 10 premieres annees demandees) — bande = intervalle de confiance a 95%, agregee correctement par sommation de variance (pas des bornes).</div>
    <div class="future-stats">
      <div class="fstat"><div class="fv" id="fstat-10y">-</div><div class="fl">Cas cumules, 2025-2034</div></div>
      <div class="fstat"><div class="fv" id="fstat-avg">-</div><div class="fl">Cas / an en moyenne</div></div>
      <div class="fstat"><div class="fv" id="fstat-r2">-</div><div class="fl">R2 modele (holdout 2018-2020)</div></div>
    </div>
    <div class="cw-lg"><canvas id="c-future"></canvas></div>
  </div>
  <div class="cg">
    <div class="panel reveal" style="animation-delay:.22s"><div class="map-title">Temperature (serie annuelle)</div><div class="cw"><canvas id="c-temp"></canvas></div></div>
    <div class="panel reveal" style="animation-delay:.26s"><div class="map-title">Precipitations (serie annuelle)</div><div class="cw"><canvas id="c-precip"></canvas></div></div>
  </div>
</main>
</div>
<footer>
  <div class="foot-brand"><strong>Institut Pasteur du Maroc</strong> — Service de Parasitologie et Maladies Vectorielles</div>
  <div>Modèle officiel GBM + PINN SEIR-V</div>
</footer>
<script>
const S={region:"",province:"",commune:"",annee:null,mois:null,variable:"temp_mean",source:"climate",futureYear:null};
const RAMPS={thermal:["#0B3D66","#2E77B5","#8FC1DE","#E8B85A","#B5502E"],blue:["#082D52","#0E4C86","#2E77B5","#7FB0DA","#C7E0F2"],purple:["#2B1F45","#5A4179","#8A6BA8","#B79BD1","#DDCBEE"],amber:["#4A2F0C","#8A5A1C","#C1852E","#E8B85A","#F5D9A0"],ember:["#3A1509","#6E2E17","#B5502E","#D6926F","#EFC3AC"]};
let VARS=null,map=null,gjLayer=null,gjData=null,ptMarkers=[],FUTURE_YEARS=[];
const MOIS_NOMS=["","Janvier","Fevrier","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Decembre"];
Chart.defaults.plugins.legend.display=false;
function mkLine(id,color){return new Chart(document.getElementById(id),{type:"line",data:{labels:[],datasets:[{data:[],borderColor:color,backgroundColor:color+"22",borderWidth:2,pointRadius:2,fill:true,tension:.35}]},options:{responsive:true,maintainAspectRatio:false}});}
function mkBandChart(id){return new Chart(document.getElementById(id),{type:"line",data:{labels:[],datasets:[
  {label:"IC95 haut",data:[],borderWidth:0,pointRadius:0,fill:false,tension:.3},
  {label:"IC95 bas",data:[],borderWidth:0,pointRadius:0,fill:"-1",backgroundColor:"rgba(181,80,46,.14)",tension:.3},
  {label:"Cas predits",data:[],borderColor:"#B5502E",backgroundColor:"rgba(181,80,46,.06)",borderWidth:2.5,pointRadius:2,fill:false,tension:.3},
]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
  scales:{y:{beginAtZero:true}},
  plugins:{legend:{display:false},tooltip:{filter:it=>it.datasetIndex===2,callbacks:{label:it=>`${num(it.raw,1)} cas predits`}}}}});}
const C={temp:mkLine("c-temp","#B5502E"),precip:mkLine("c-precip","#0E4C86"),future:mkBandChart("c-future")};
async function api(ep,p={}){const qs=new URLSearchParams(p).toString();try{const r=await fetch(`/api/${ep}${qs?"?"+qs:""}`);return await r.json();}catch(e){return[];}}
function num(v,d=1){return(v!=null&&!isNaN(v))?(+v).toFixed(d):"-";}
function countUp(el,target,decimals=3,duration=900){
  if(target==null||isNaN(target)){el.textContent="-";return;}
  const start=performance.now();
  function tick(now){
    const t=Math.min(1,(now-start)/duration);
    const eased=1-Math.pow(1-t,3);
    el.textContent=(target*eased).toFixed(decimals);
    if(t<1)requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
async function init(){
  const info=await api("db_info");
  document.getElementById("fstat-r2").textContent=info.model_r2!=null?num(info.model_r2,3):"-";
  if(info.model_res){
    countUp(document.getElementById("res-commune-mois"),info.model_res.commune_mois);
    countUp(document.getElementById("res-province-mois"),info.model_res.province_mois);
    countUp(document.getElementById("res-commune-annee"),info.model_res.commune_annee);
    countUp(document.getElementById("res-province-annee"),info.model_res.province_annee);
  }
  VARS=await api("variables");renderVars();
  const regs=await api("regions");const sel=document.getElementById("sel-region");
  regs.forEach(r=>{const o=document.createElement("option");o.value=r;o.textContent=r;sel.appendChild(o);});
  FUTURE_YEARS=await api("future_years");
  const fy=document.getElementById("sel-future-year");
  FUTURE_YEARS.forEach(y=>{const o=document.createElement("option");o.value=y;o.textContent=y;fy.appendChild(o);});
  if(FUTURE_YEARS.length)S.futureYear=FUTURE_YEARS[0];
  const moisSel=document.getElementById("sel-mois");
  for(let m=1;m<=12;m++){const o=document.createElement("option");o.value=m;o.textContent=MOIS_NOMS[m];moisSel.appendChild(o);}
  const annees=await api("years");const anneeSel=document.getElementById("sel-annee");
  annees.forEach(y=>{const o=document.createElement("option");o.value=y;o.textContent=y;anneeSel.appendChild(o);});
  initMap();await loadMap();
  await loadFutureTrend();
}
function curVars(){return VARS[S.source==="env"?"environment":S.source]||{};}
const SOURCE_TITLES={risk:"Risque",zone:"Zone",prediction:"Prediction",future:"Futur"};
const SOURCE_SUB={
  prediction:"Dernière estimation validée (2024), recalibrée sur les vérités-terrain régionales réelles 2021/2023/2024.",
  future:"Projection prospective 2025-2045, recalculée mois par mois à partir des prédictions du modèle lui-même — pas une mesure.",
};
function renderVars(){const d=curVars();document.getElementById("var-section-title").textContent=SOURCE_TITLES[S.source]||"Variable";const keys=Object.keys(d);if(!keys.includes(S.variable))S.variable=keys[0];
  const sub=document.getElementById("var-section-sub");
  if(SOURCE_SUB[S.source]){sub.textContent=SOURCE_SUB[S.source];sub.classList.remove("hidden");}else{sub.classList.add("hidden");}
  document.getElementById("var-grid").innerHTML=keys.map(k=>`<div class="var-btn ${k===S.variable?"on":""}" onclick="pickVar('${k}')"><span>${d[k].label}</span><span>${d[k].unit}</span></div>`).join("");}
function setSource(s){S.source=s;["climate","env","risk","zone","prediction","future"].forEach(x=>document.getElementById("src-"+x).classList.toggle("on",(x==="env"?"environment":x)===s));
  document.getElementById("future-year-group").classList.toggle("hidden",s!=="future");
  document.getElementById("climate-time-group").classList.toggle("hidden",s!=="climate");
  renderVars();loadMap();}
function pickVar(k){S.variable=k;renderVars();loadMap();}
document.getElementById("sel-mois").onchange=function(){S.mois=this.value||null;loadMap();};
document.getElementById("sel-annee").onchange=function(){S.annee=this.value||null;loadMap();};
document.getElementById("sel-future-year").onchange=function(){S.futureYear=this.value;loadMap();};
function initMap(){map=L.map("map",{zoomControl:true}).setView([31.5,-6.5],6);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",{attribution:"© OSM © CARTO",maxZoom:12,minZoom:5}).addTo(map);
  map.createPane("labels").style.zIndex=650;
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png",{attribution:"",pane:"labels",maxZoom:12,minZoom:5,opacity:.6}).addTo(map);}
function shapeStroke(){return "rgba(14,76,134,.55)";}
function noDataColor(){return "#e4ecf5";}
function colorFor(v,mn,mx,ramp){if(v==null||isNaN(v))return noDataColor();const c=RAMPS[ramp]||RAMPS.thermal;const t=mx>mn?(v-mn)/(mx-mn):.5;return c[Math.max(0,Math.min(c.length-1,Math.floor(t*c.length)))];}
async function loadMap(){
  const p={variable:S.variable,source:S.source};
  if(S.source==="climate"){if(S.annee)p.annee=S.annee;if(S.mois)p.mois=S.mois;}
  if(S.source==="future"&&S.futureYear)p.annee=S.futureYear;
  if(S.region)p.region=S.region;
  const data=await api("map_data",p);const vi=curVars()[S.variable];
  let title=vi.label;
  if(S.source==="future")title=`${vi.label} — ${S.futureYear}`;
  else if(S.source==="climate"&&(S.mois||S.annee))title=`${vi.label} — ${S.mois?MOIS_NOMS[S.mois]:"annee entiere"}${S.annee?" "+S.annee:" (moy. 2009-2021)"}`;
  document.getElementById("map-title").textContent=title;
  const c=RAMPS[vi.ramp]||RAMPS.thermal;document.getElementById("leg-grad").style.background=`linear-gradient(to right,${c.join(",")})`;
  document.getElementById("leg-min").textContent=num(data.min,2);document.getElementById("leg-max").textContent=num(data.max,2);
  if(data.geojson_available){if(!gjData)gjData=await api("geojson");renderChoro(gjData,data.values,data.min,data.max,vi);}
  else if(data.points)renderPts(data.points,data.values,data.min,data.max,vi);
}
function popupHtml(name,v,vi){
  const periodBit=S.source==="future"?` · projection ${S.futureYear}`:(S.source==="climate"&&(S.mois||S.annee)?` · ${S.mois?MOIS_NOMS[S.mois]:"annee entiere"}${S.annee?" "+S.annee:""}`:"");
  return `<div class="lf-pop"><div class="lf-pop-name">${name}</div><div class="lf-pop-val">${v!=null?`${v} <span>${vi.unit}</span>`:"pas de donnee"}</div><div class="lf-pop-meta">${vi.label}${periodBit}</div></div>`;
}
function renderChoro(gj,vals,mn,mx,vi){if(gjLayer)map.removeLayer(gjLayer);
  const stroke=shapeStroke();
  gjLayer=L.geoJSON(gj,{style:f=>{const v=vals[String(f.properties.commune_id)];return{fillColor:colorFor(v,mn,mx,vi.ramp),fillOpacity:v!=null?.82:.35,color:stroke,weight:.85,lineJoin:"round",lineCap:"round"};},
  onEachFeature:(f,l)=>{const v=vals[String(f.properties.commune_id)];
    l.on("mouseover",function(){this.setStyle({weight:1.6,color:"#0E4C86"});this.bringToFront();});
    l.on("mouseout",function(){if(!this._popupOpen)this.setStyle({weight:.85,color:stroke});});
    l.on("click",function(e){this.setStyle({weight:2.2,color:"#D89A3E"});this._popupOpen=true;
      this.once("popupclose",()=>{this._popupOpen=false;this.setStyle({weight:.85,color:stroke});});
      L.popup({className:"lf-pop-wrap",closeButton:true}).setLatLng(e.latlng).setContent(popupHtml(f.properties.name,v,vi)).openOn(map);});
    l.bindTooltip(`${f.properties.name}${v!=null?` — ${v} ${vi.unit}`:" — pas de donnee"}`,{sticky:true,className:"lf-tip"});}}).addTo(map);}
function renderPts(pts,vals,mn,mx,vi){ptMarkers.forEach(m=>map.removeLayer(m));ptMarkers=[];
  const stroke=shapeStroke();
  pts.forEach(p=>{const v=vals[String(p.commune_id)];if(v==null)return;const m=L.circleMarker([p.latitude,p.longitude],{radius:5,fillColor:colorFor(v,mn,mx,vi.ramp),fillOpacity:.88,color:stroke,weight:.8}).addTo(map);
    m.on("click",()=>{L.popup({className:"lf-pop-wrap",closeButton:true}).setLatLng([p.latitude,p.longitude]).setContent(popupHtml(p.commune,v,vi)).openOn(map);});
    m.bindTooltip(`${p.commune} — ${v} ${vi.unit}`,{sticky:true,className:"lf-tip"});ptMarkers.push(m);});}
document.getElementById("sel-region").onchange=async function(){S.region=this.value;S.province="";S.commune="";
  const sp=document.getElementById("sel-province"),sc=document.getElementById("sel-commune");
  sp.innerHTML='<option value="">Toutes</option>';sp.disabled=true;sc.innerHTML='<option value="">Toutes</option>';sc.disabled=true;
  if(S.region){const pr=await api("provinces",{region:S.region});pr.forEach(p=>{const o=document.createElement("option");o.value=p;o.textContent=p;sp.appendChild(o);});sp.disabled=false;}
  loadMap();refreshCharts();loadFutureTrend();};
document.getElementById("sel-province").onchange=async function(){S.province=this.value;S.commune="";const sc=document.getElementById("sel-commune");sc.innerHTML='<option value="">Toutes</option>';sc.disabled=true;
  if(S.province){const co=await api("communes",{province:S.province});co.forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;sc.appendChild(o);});sc.disabled=false;}refreshCharts();};
document.getElementById("sel-commune").onchange=function(){S.commune=this.value;refreshCharts();};
async function refreshCharts(){if(!S.region)return;const d=await api("stats",{region:S.region,province:S.province,commune:S.commune});if(!d.length)return;
  const labels=d.map(x=>x.annee);C.temp.data.labels=labels;C.temp.data.datasets[0].data=d.map(x=>x.temp_mean);C.temp.update();
  C.precip.data.labels=labels;C.precip.data.datasets[0].data=d.map(x=>x.precipitation);C.precip.update();}
async function loadFutureTrend(){
  const d=await api("future_trend",S.region?{region:S.region}:{});
  document.getElementById("future-region-label").textContent=S.region||"Maroc entier";
  if(!d.years||!d.years.length)return;
  C.future.data.labels=d.years;
  C.future.data.datasets[0].data=d.ci_upper;
  C.future.data.datasets[1].data=d.ci_lower;
  C.future.data.datasets[2].data=d.predicted;
  C.future.update();
  const first10=d.predicted.slice(0,10);
  const sum10=first10.reduce((a,b)=>a+b,0);
  document.getElementById("fstat-10y").textContent=num(sum10,0);
  document.getElementById("fstat-avg").textContent=num(sum10/Math.max(first10.length,1),1);
}
init();
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    if not os.path.exists(DB_CLIMATE):
        print(f"ERREUR : {DB_CLIMATE} introuvable. Lance d'abord extract_climate.py.")
    else:
        print("=" * 50)
        print("  LeishSergenti Dashboard")
        print(f"  Climat  : {DB_CLIMATE}")
        print(f"  GeoJSON : {'OK' if GEOJSON_AVAILABLE else 'MANQUANT (mode points)'}")
        print(f"  Risque  : {'OK' if RISK_BY_PROVINCE else 'absent (lance le modele)'}")
        print("  http://localhost:5050")
        print("=" * 50)
        app.run(host="0.0.0.0", port=5050, debug=False)
