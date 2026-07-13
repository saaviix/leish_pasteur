from flask import Flask, jsonify, request, render_template_string
import sqlite3, os, json
import pandas as pd

app = Flask(__name__)
DB_ENV     = "environment_morocco.db"
DB_CLIMATE = "climate_morocco.db" if os.path.exists("climate_morocco.db") else "climate_morocco_era5_final.db"
GEOJSON_FILE = "communes_morocco.geojs
def detect_table():
    if not os.path.exists(DB_CLIMATE):
        return "climate"
    conn = sqlite3.connect(DB_CLIMATE)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    conn.close()
    for t in ["climate", "climate_monthly"]:
        if t in tables:
            return t
    return tables[0] if tables else "climate"

TABLE = detect_table()

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

C_TEMP   = col("temp_mean", "t2m_c", "temperature_2m_mean")
C_DEW    = col("dewpoint", "d2m_c", "2m_dewpoint_temperature")
C_TMAX   = col("temp_max", "temperature_2m_max")
C_TMIN   = col("temp_min", "temperature_2m_min")
C_HUM    = col("humidity", "humidity_rel", "rh_percent")
C_PRECIP = col("precipitation", "tp", "precipitation_sum")
C_WIND   = col("wind_speed", "si10_ms")
C_RAD    = col("radiation", "ssrd")
C_EVAP   = col("evapotrans", "pev")
C_REGION = col("region")
C_PROV   = col("province")
C_COM    = col("commune")
C_YEAR   = col("annee", "year")
C_MONTH  = col("mois", "month")
C_CID    = col("commune_id")

print(f"Using DB -> {DB_CLIMATE} ; table -> {TABLE}")
print(f"Columns: {', '.join(COLS)}")

VARIABLES = {
    "temp_mean":     {"label": "Température moyenne",  "col": C_TEMP,   "unit": "°C",   "ramp": "thermal"},
    "temp_max":      {"label": "Température maximale",  "col": C_TMAX,   "unit": "°C",   "ramp": "thermal"},
    "humidity":      {"label": "Humidité relative",     "col": C_HUM,    "unit": "%",    "ramp": "blue"},
    "precipitation": {"label": "Précipitations",        "col": C_PRECIP, "unit": "mm",   "ramp": "blue"},
    "wind_speed":    {"label": "Vitesse du vent",       "col": C_WIND,   "unit": "m/s",  "ramp": "purple"},
    "radiation":     {"label": "Rayonnement solaire",   "col": C_RAD,    "unit": "MJ/m²","ramp": "amber"},
    "evapotrans":    {"label": "Évapotranspiration",    "col": C_EVAP,   "unit": "mm",   "ramp": "amber"},
}
VARIABLES = {k: v for k, v in VARIABLES.items() if v["col"]}

ENV_VARIABLES = {
    "altitude_m":    {"label": "Altitude",            "unit": "m",        "ramp": "thermal"},
    "pop_density":   {"label": "Densité population",   "unit": "hab/km²",  "ramp": "purple"},
    "dist_water_km": {"label": "Distance cours d'eau", "unit": "km",       "ramp": "blue"},
    "aridity_index": {"label": "Indice aridité",       "unit": "",         "ramp": "amber"},
}

def qc(sql, params=()):
    conn = sqlite3.connect(DB_CLIMATE)
    df   = pd.read_sql_query(sql, conn, params=list(params))
    conn.close()
    return df

def qe(sql, params=()):
    if not os.path.exists(DB_ENV):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_ENV)
    df   = pd.read_sql_query(sql, conn, params=list(params))
    conn.close()
    return df


GEOJSON_DATA = None
GEOJSON_AVAILABLE = os.path.exists(GEOJSON_FILE)

if GEOJSON_AVAILABLE:
    with open(GEOJSON_FILE, encoding="utf-8") as f:
        GEOJSON_DATA = json.load(f)
    print(f"GeoJSON chargé : {len(GEOJSON_DATA['features'])} polygones")
else:
    print(f"ATTENTION : {GEOJSON_FILE} introuvable. Lance 08_fetch_geojson.py d'abord.")
    print("Le dashboard fonctionnera en mode 'points' (cercles) en attendant.")

# Fallback : si pas de geojson, on construit des points depuis la DB
def build_points_fallback():
    df = qc(f"SELECT DISTINCT {C_CID} AS commune_id, {C_COM} AS commune, latitude, longitude FROM {TABLE}")
    return df.to_dict(orient="records")


@app.route("/api/regions")
def api_regions():
    df = qc(f"SELECT DISTINCT region FROM {TABLE} WHERE region IS NOT NULL ORDER BY region")
    return jsonify(df["region"].tolist())

@app.route("/api/provinces")
def api_provinces():
    r = request.args.get("region","")
    df = qc(f"SELECT DISTINCT province FROM {TABLE} WHERE region=? ORDER BY province", [r])
    return jsonify(df["province"].tolist())

@app.route("/api/communes")
def api_communes():
    p = request.args.get("province","")
    df = qc(f"SELECT DISTINCT commune FROM {TABLE} WHERE province=? ORDER BY commune", [p])
    return jsonify(df["commune"].tolist())

@app.route("/api/years")
def api_years():
    df = qc(f"SELECT DISTINCT annee FROM {TABLE} ORDER BY annee")
    return jsonify(df["annee"].tolist())

@app.route("/api/variables")
def api_variables():
    return jsonify({
        "climate": VARIABLES,
        "environment": ENV_VARIABLES,
    })



@app.route("/api/map_data")
def api_map_data():
    var_key  = request.args.get("variable", "temp_mean")
    source   = request.args.get("source", "climate")
    annee    = request.args.get("annee", "")
    mois     = request.args.get("mois", "")
    region   = request.args.get("region", "")

    if source == "environment":
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
            sql += f" AND {C_YEAR}=?"
            params.append(int(annee))
        if mois:
            sql += f" AND {C_MONTH}=?"
            params.append(int(mois))
        if region:
            sql += f" AND {C_REGION}=?"
            params.append(region)
        sql += f" GROUP BY {C_CID}"
        df = qc(sql, params)

    if df.empty:
        return jsonify({"values": {}, "min": 0, "max": 0, "geojson_available": GEOJSON_AVAILABLE})

    values = {
        str(int(row["commune_id"])): round(float(row["value"]), 2)
        for _, row in df.iterrows()
        if pd.notna(row["value"]) and pd.notna(row["commune_id"])
    }

    vmin = min(values.values()) if values else 0
    vmax = max(values.values()) if values else 0

    response = {
        "values": values,
        "min": vmin,
        "max": vmax,
        "geojson_available": GEOJSON_AVAILABLE,
    }

    if not GEOJSON_AVAILABLE:
        points = build_points_fallback()
        response["points"] = [
            p for p in points if str(p["commune_id"]) in values
        ]

    return jsonify(response)

@app.route("/api/geojson")
def api_geojson():
    if not GEOJSON_AVAILABLE:
        return jsonify({"type": "FeatureCollection", "features": []})
    return jsonify(GEOJSON_DATA)



@app.route("/api/stats")
def api_stats():
    region   = request.args.get("region","")
    province = request.args.get("province","")
    commune  = request.args.get("commune","")
    if not region or not C_YEAR or not C_TEMP:
        return jsonify([])

    aggs = [f"AVG({C_TEMP}) AS temp_mean"]
    if C_TMAX: aggs.append(f"MAX({C_TMAX}) AS temp_max")
    if C_TMIN: aggs.append(f"MIN({C_TMIN}) AS temp_min")
    if C_HUM:  aggs.append(f"AVG({C_HUM}) AS humidity")
    if C_WIND: aggs.append(f"AVG({C_WIND}) AS wind_speed")
    if C_PRECIP: aggs.append(f"AVG({C_PRECIP}) AS precipitation")
    sql = f"""
        SELECT {C_YEAR} AS annee,
               {', '.join(aggs)},
               COUNT(DISTINCT {C_COM}) AS n_communes
        FROM {TABLE}
        WHERE {C_REGION}=?
    """
    params = [region]
    if province and C_PROV:
        sql += f" AND {C_PROV}=?"
        params.append(province)
    if commune and C_COM:
        sql += f" AND {C_COM}=?"
        params.append(commune)
    sql += f" GROUP BY {C_YEAR} ORDER BY {C_YEAR}"

    df = qc(sql, params)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))

@app.route("/api/monthly_profile")
def api_monthly_profile():
    region   = request.args.get("region","")
    province = request.args.get("province","")
    commune  = request.args.get("commune","")
    annee    = request.args.get("annee","")
    if not region or not C_MONTH:
        return jsonify([])

    aggs = []
    if C_TEMP: aggs.append(f"AVG({C_TEMP}) AS temp_mean")
    if C_TMAX: aggs.append(f"AVG({C_TMAX}) AS temp_max")
    if C_TMIN: aggs.append(f"AVG({C_TMIN}) AS temp_min")
    if C_DEW:  aggs.append(f"AVG({C_DEW}) AS dewpoint")
    if C_HUM:  aggs.append(f"AVG({C_HUM}) AS humidity")
    if C_WIND: aggs.append(f"AVG({C_WIND}) AS wind_speed")
    if C_PRECIP: aggs.append(f"AVG({C_PRECIP}) AS precipitation")
    if C_RAD:  aggs.append(f"AVG({C_RAD}) AS radiation")
    if C_EVAP: aggs.append(f"AVG({C_EVAP}) AS evapotrans")

    sql    = f"SELECT {C_MONTH} AS mois, {', '.join(aggs)} FROM {TABLE} WHERE {C_REGION}=?"
    params = [region]
    if province and C_PROV: sql += f" AND {C_PROV}=?"; params.append(province)
    if commune  and C_COM:  sql += f" AND {C_COM}=?";   params.append(commune)
    if annee    and C_YEAR: sql += f" AND {C_YEAR}=?";   params.append(int(annee))
    sql += f" GROUP BY {C_MONTH} ORDER BY {C_MONTH}"

    df = qc(sql, params)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))

@app.route("/api/climate")
def api_climate():
    region   = request.args.get("region","")
    province = request.args.get("province","")
    commune  = request.args.get("commune","")
    annee    = request.args.get("annee","")
    mois     = request.args.get("mois","")

    sql    = f"SELECT * FROM {TABLE} WHERE region=?"
    params = [region]
    if province: sql += " AND province=?"; params.append(province)
    if commune:  sql += " AND commune=?";  params.append(commune)
    if annee:    sql += " AND annee=?";    params.append(int(annee))
    if mois:     sql += " AND mois=?";     params.append(int(mois))
    sql += " ORDER BY annee, mois LIMIT 2000"

    df = qc(sql, params)
    rename_map = {}
    if C_TEMP and C_TEMP != "temp_mean": rename_map[C_TEMP] = "temp_mean"
    if C_DEW  and C_DEW  != "dewpoint":  rename_map[C_DEW]  = "dewpoint"
    if C_HUM  and C_HUM  != "humidity":  rename_map[C_HUM]  = "humidity"
    if C_WIND and C_WIND != "wind_speed": rename_map[C_WIND] = "wind_speed"
    if C_PRECIP and C_PRECIP != "precipitation": rename_map[C_PRECIP] = "precipitation"
    if C_RAD  and C_RAD  != "radiation":  rename_map[C_RAD]  = "radiation"
    if C_EVAP and C_EVAP != "evapotrans": rename_map[C_EVAP] = "evapotrans"
    df = df.rename(columns=rename_map)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))

@app.route("/api/db_info")
def api_db_info():
    conn  = sqlite3.connect(DB_CLIMATE)
    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    coms  = conn.execute(f"SELECT COUNT(DISTINCT commune) FROM {TABLE}").fetchone()[0]
    ymin  = conn.execute(f"SELECT MIN(annee) FROM {TABLE}").fetchone()[0]
    ymax  = conn.execute(f"SELECT MAX(annee) FROM {TABLE}").fetchone()[0]
    conn.close()

    env_ok = os.path.exists(DB_ENV)
    env_n  = 0
    if env_ok:
        conn2 = sqlite3.connect(DB_ENV)
        tbls  = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn2
        )["name"].tolist()
        if tbls:
            env_n = conn2.execute(f"SELECT COUNT(*) FROM {tbls[0]}").fetchone()[0]
        conn2.close()

    return jsonify({
        "total_rows": total, "communes": coms,
        "year_min": ymin,    "year_max": ymax,
        "env_ok": env_ok,    "env_rows": env_n,
        "geojson_ok": GEOJSON_AVAILABLE,
        "geojson_features": len(GEOJSON_DATA["features"]) if GEOJSON_DATA else 0,
    })

@app.route("/api/env")
def api_env():
    commune = request.args.get("commune","")
    region  = request.args.get("region","")
    if not os.path.exists(DB_ENV):
        return jsonify([])
    conn  = sqlite3.connect(DB_ENV)
    tbls  = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    conn.close()
    if not tbls:
        return jsonify([])
    tbl    = tbls[0]
    sql    = f"SELECT * FROM {tbl} WHERE 1=1"
    params = []
    if commune: sql += " AND commune=?"; params.append(commune)
    elif region: sql += " AND region=?"; params.append(region)
    sql += " LIMIT 100"
    df = qe(sql, params)
    return jsonify(df.where(pd.notna(df), None).to_dict(orient="records"))

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LeishNet · Morocco Climate Map</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg0:#F4F8FC;--bg1:#FFFFFF;--bg2:#EEF4FA;--bg3:#E3EDF7;
  --border:#D6E3F0;--border2:#C5D5E8;
  --teal:#00A896;--teal2:#00866F;
  --amber:#F59E0B;--rose:#E53E6A;--blue:#0857C3;--purple:#8B5CF6;
  --t1:#0B1F3A;--t2:#3D5570;--t3:#7E94AC;
  --mono:'JetBrains Mono',monospace;
  --sans:'Inter',sans-serif;
  --r:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg0);color:var(--t1);font-family:var(--sans);min-height:100vh;overflow-x:hidden}

header{
  background:rgba(255,255,255,.95);
  border-bottom:1px solid var(--border);
  box-shadow:0 1px 12px rgba(8,87,195,0.06);
}
.hlogo{display:flex;align-items:center;gap:12px}
.hring{
  width:30px;height:30px;border-radius:50%;flex-shrink:0;
  background:conic-gradient(var(--teal) 0%,var(--blue) 50%,var(--teal) 100%);
  animation:spin 6s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.htitle{font-size:15px;font-weight:700;letter-spacing:-.02em}
.hsub{font-size:10px;color:var(--teal);font-family:var(--mono);letter-spacing:.1em;margin-top:1px}
.hright{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tag{
  background:rgba(0,212,170,.08);border:1px solid rgba(0,212,170,.2);
  color:var(--teal);padding:3px 10px;border-radius:100px;
  font-size:10px;font-family:var(--mono);letter-spacing:.05em;white-space:nowrap;
}
.tag.warn{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.25);color:var(--amber)}

.app{display:grid;grid-template-columns:280px 1fr;min-height:calc(100vh - 56px)}

aside{
  background:var(--bg1);border-right:1px solid var(--border);
  padding:18px 12px;display:flex;flex-direction:column;gap:18px;overflow-y:auto;
}
.s-title{
  font-size:9px;font-family:var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--t3);margin-bottom:10px;padding-bottom:7px;border-bottom:1px solid var(--border);
}
.fg{display:flex;flex-direction:column;gap:5px;margin-bottom:9px}
.fg label{font-size:11px;color:var(--t2);font-weight:500}
select{
  width:100%;background:var(--bg2);border:1px solid var(--border2);
  color:var(--t1);padding:8px 10px;border-radius:var(--r);
  font-family:var(--sans);font-size:12px;cursor:pointer;transition:border-color .2s;
  appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%233d5070' d='M5 7L1 2h8z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 9px center;
}
select:focus{outline:none;border-color:var(--teal)}
select:disabled{opacity:.4;cursor:default}

.var-grid{display:flex;flex-direction:column;gap:5px}
.var-btn{
  background:var(--bg2);border:1px solid var(--border2);color:var(--t2);
  padding:8px 10px;border-radius:var(--r);font-size:11px;text-align:left;
  cursor:pointer;transition:all .15s;display:flex;justify-content:space-between;align-items:center;
}
.var-btn:hover{border-color:var(--teal2);color:var(--teal)}
.var-btn.on{background:rgba(0,212,170,.14);border-color:var(--teal);color:var(--teal);font-weight:600}
.var-btn .unit{font-size:9px;color:var(--t3);font-family:var(--mono)}
.var-btn.on .unit{color:var(--teal2)}

.source-toggle{display:flex;gap:4px;margin-bottom:10px}
.source-btn{
  flex:1;background:var(--bg2);border:1px solid var(--border2);color:var(--t2);
  padding:6px 8px;border-radius:var(--r);font-size:10px;cursor:pointer;text-align:center;
  font-family:var(--mono);letter-spacing:.05em;
}
.source-btn.on{background:rgba(139,92,246,.15);border-color:var(--purple);color:var(--purple)}

main{background:var(--bg0);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px}

.bc{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--t3);font-family:var(--mono);flex-wrap:wrap}
.bc .node{color:var(--teal)}
.bc .sep{color:var(--border2)}

.map-panel{
  background:var(--bg1);border:1px solid var(--border);border-radius:12px;
  padding:16px;display:flex;flex-direction:column;gap:12px;
}
.map-head{display:flex;align-items:baseline;justify-content:space-between}
.map-title{font-size:13px;font-weight:600;color:var(--t1)}
.map-sub{font-size:10px;color:var(--t3);font-family:var(--mono)}

#map{
  width:100%;height:520px;border-radius:10px;
  background:var(--bg2);border:1px solid var(--border2);
}
.leaflet-container{background:var(--bg2) !important;font-family:var(--sans) !important}
.leaflet-popup-content-wrapper{background:var(--bg1) !important;color:var(--t1) !important;border-radius:8px !important}
.leaflet-popup-tip{background:var(--bg1) !important}
.leaflet-control-zoom a{background:var(--bg1) !important;color:var(--t1) !important;border-color:var(--border2) !important}
.leaflet-control-attribution{background:rgba(8,14,28,.8) !important;color:var(--t3) !important}
.leaflet-control-attribution a{color:var(--teal) !important}

.legend-bar{
  display:flex;align-items:center;gap:10px;padding:10px 14px;
  background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r);
}
.legend-grad{flex:1;height:10px;border-radius:5px}
.legend-val{font-size:10px;font-family:var(--mono);color:var(--t2);min-width:42px}

.timeline{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tl-head{font-size:9px;font-family:var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--t3);margin-bottom:10px}
.ypills{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.ybtn{
  background:var(--bg2);border:1px solid var(--border2);color:var(--t2);
  padding:3px 9px;border-radius:100px;font-family:var(--mono);
  font-size:10px;cursor:pointer;transition:all .15s;
}
.ybtn:hover{border-color:var(--teal2);color:var(--teal)}
.ybtn.on{background:rgba(0,212,170,.15);border-color:var(--teal);color:var(--teal);font-weight:600}
.mpills{display:flex;flex-wrap:wrap;gap:5px}
.mbtn{
  background:var(--bg2);border:1px solid var(--border2);color:var(--t3);
  padding:3px 8px;border-radius:var(--r);font-family:var(--mono);
  font-size:10px;cursor:pointer;transition:all .15s;
}
.mbtn:hover{border-color:var(--purple);color:var(--purple)}
.mbtn.on{background:rgba(139,92,246,.15);border-color:var(--purple);color:var(--purple);font-weight:600}

.cg{display:grid;gap:12px}
.cg.two{grid-template-columns:1fr 1fr}

.panel{
  background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:16px;
  transition:border-color .2s;
}
.panel:hover{border-color:var(--teal2)}
.ph{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px}
.pt{font-size:12px;font-weight:600;color:var(--t1)}
.ps{font-size:10px;color:var(--t3);font-family:var(--mono)}
.cw{position:relative;height:210px}

.kpi-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.kpi{
  background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r);
  padding:10px 8px;text-align:center;
}
.kv{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--teal);display:block;line-height:1.1}
.kl{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-top:3px;display:block}

.env-grid{display:flex;flex-direction:column;gap:4px}
.env-row{
  display:flex;justify-content:space-between;align-items:center;
  background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--r);padding:6px 10px;font-size:11px;
}
.env-key{color:var(--t2)}
.env-val{font-family:var(--mono);color:var(--teal);font-size:10px}

.empty-msg{
  display:flex;align-items:center;justify-content:center;
  height:80px;color:var(--t3);font-family:var(--mono);font-size:11px;letter-spacing:.05em;
}

@media(max-width:860px){
  .app{grid-template-columns:1fr}
  .cg.two{grid-template-columns:1fr}
  #map{height:380px}
}
</style>
</head>
<!-- ============================================================
     BOUTON INSECTES ANIMÉS — Easter Egg
     ============================================================ -->
<button id="bug-trigger" title="Ne pas cliquer... ou si">🦟</button>

<div id="envelope-container"></div>

<div id="letter-overlay">
  <div id="letter-card">
    <div id="letter-stamp">⚠️</div>
    <h2>Attention !</h2>
    <p>Des insectes rôdent dehors.</p>
    <p class="letter-sub">Protégez-vous des piqûres de phlébotomes — vecteurs de la leishmaniose au Maroc.</p>
    <button id="letter-close">Fermer</button>
  </div>
</div>

<style>
/* ===== BOUTON DÉCLENCHEUR ===== */
#bug-trigger{
  position:fixed; bottom:24px; right:24px; z-index:9999;
  width:58px; height:58px; border-radius:50%;
  background:linear-gradient(135deg, var(--teal), var(--blue));
  border:none; font-size:26px; cursor:pointer;
  box-shadow:0 6px 20px rgba(0,168,150,0.35);
  transition:transform .2s;
  display:flex; align-items:center; justify-content:center;
}
#bug-trigger:hover{ transform:scale(1.1) rotate(-8deg); }
#bug-trigger:active{ transform:scale(0.92); }

/* ===== INSECTES VOLANTS ===== */
.flying-bug{
  position:fixed; font-size:22px; z-index:9998;
  pointer-events:none; user-select:none;
  filter:drop-shadow(0 2px 4px rgba(0,0,0,0.15));
}

@keyframes flyPath1{
  0%   { transform:translate(0,0) rotate(0deg) scale(0.6); opacity:0; }
  10%  { opacity:1; }
  25%  { transform:translate(30vw,-15vh) rotate(40deg) scale(1); }
  50%  { transform:translate(60vw,10vh) rotate(-20deg) scale(1.1); }
  75%  { transform:translate(20vw,40vh) rotate(60deg) scale(0.9); }
  90%  { opacity:1; }
  100% { transform:translate(-10vw,60vh) rotate(0deg) scale(0.5); opacity:0; }
}
@keyframes flyPath2{
  0%   { transform:translate(0,0) rotate(0deg) scale(0.6); opacity:0; }
  10%  { opacity:1; }
  30%  { transform:translate(-40vw,20vh) rotate(-30deg) scale(1); }
  55%  { transform:translate(-20vw,-30vh) rotate(50deg) scale(1.05); }
  80%  { transform:translate(35vw,-10vh) rotate(-15deg) scale(0.95); }
  90%  { opacity:1; }
  100% { transform:translate(50vw,30vh) rotate(0deg) scale(0.5); opacity:0; }
}
@keyframes flyPath3{
  0%   { transform:translate(0,0) rotate(0deg) scale(0.6); opacity:0; }
  10%  { opacity:1; }
  20%  { transform:translate(15vw,-40vh) rotate(70deg) scale(1.1); }
  45%  { transform:translate(-30vw,-20vh) rotate(-40deg) scale(0.9); }
  70%  { transform:translate(-10vw,30vh) rotate(20deg) scale(1); }
  90%  { opacity:1; }
  100% { transform:translate(40vw,50vh) rotate(0deg) scale(0.5); opacity:0; }
}

.flying-bug.p1{ animation:flyPath1 4.5s ease-in-out forwards; }
.flying-bug.p2{ animation:flyPath2 5s ease-in-out forwards; }
.flying-bug.p3{ animation:flyPath3 4.8s ease-in-out forwards; }

/* ===== ENVELOPPE ===== */
#envelope-container{
  position:fixed; bottom:24px; right:24px; z-index:9997;
}
.envelope-pop{
  position:fixed; bottom:90px; right:24px; z-index:9999;
  font-size:42px; cursor:pointer;
  animation:envelopeBounceIn .6s cubic-bezier(.34,1.56,.64,1) forwards, envelopeFloat 1.8s ease-in-out infinite 0.6s;
  filter:drop-shadow(0 6px 14px rgba(0,0,0,0.2));
}
@keyframes envelopeBounceIn{
  0%   { transform:scale(0) translateY(40px); opacity:0; }
  60%  { transform:scale(1.15) translateY(-8px); opacity:1; }
  100% { transform:scale(1) translateY(0); opacity:1; }
}
@keyframes envelopeFloat{
  0%,100% { transform:translateY(0); }
  50%     { transform:translateY(-6px); }
}

/* ===== OVERLAY LETTRE ===== */
#letter-overlay{
  position:fixed; inset:0; z-index:10000;
  background:rgba(11,31,58,0.55); backdrop-filter:blur(4px);
  display:none; align-items:center; justify-content:center;
}
#letter-overlay.show{ display:flex; animation:fadeIn .25s ease; }
@keyframes fadeIn{ from{opacity:0} to{opacity:1} }

#letter-card{
  background:#FFFFFF; border-radius:16px; padding:36px 40px;
  max-width:380px; text-align:center;
  box-shadow:0 20px 60px rgba(0,0,0,0.25);
  animation:letterPop .4s cubic-bezier(.34,1.56,.64,1);
  position:relative;
}
@keyframes letterPop{
  0%   { transform:scale(0.7) translateY(30px); opacity:0; }
  100% { transform:scale(1) translateY(0); opacity:1; }
}
#letter-stamp{
  font-size:48px; margin-bottom:8px;
  animation:stampWiggle 1s ease-in-out infinite;
}
@keyframes stampWiggle{
  0%,100% { transform:rotate(-4deg); }
  50%     { transform:rotate(4deg); }
}
#letter-card h2{ color:var(--rose); font-size:22px; margin-bottom:10px; font-family:'Cambria',serif; }
#letter-card p{ color:var(--t2); font-size:14px; line-height:1.5; margin-bottom:6px; }
.letter-sub{ font-size:12px !important; color:var(--t3) !important; margin-top:8px !important; }
#letter-close{
  margin-top:18px; background:var(--teal); color:white; border:none;
  padding:10px 28px; border-radius:100px; font-size:13px; font-weight:600;
  cursor:pointer; transition:transform .15s, background .2s;
}
#letter-close:hover{ background:var(--teal2); transform:scale(1.05); }
</style>

<script>
// ============================================================
// EASTER EGG — Insectes animés + lettre
// ============================================================
const BUG_EMOJIS = ["🦟","🐛","🦋","🐜","🐝"];
const BUG_PATHS  = ["p1","p2","p3"];

document.getElementById("bug-trigger").addEventListener("click", function() {
  // Empêcher double-clic pendant l'animation
  this.disabled = true;
  this.style.opacity = "0.5";

  // Spawn 8 insectes à des positions de départ aléatoires
  const bugCount = 8;
  for (let i = 0; i < bugCount; i++) {
    setTimeout(() => spawnBug(), i * 150);
  }

  // Après l'animation des insectes, faire apparaître l'enveloppe
  setTimeout(() => {
    showEnvelope();
    document.getElementById("bug-trigger").disabled = false;
    document.getElementById("bug-trigger").style.opacity = "1";
  }, 3200);
});

function spawnBug() {
  const bug = document.createElement("div");
  const emoji = BUG_EMOJIS[Math.floor(Math.random() * BUG_EMOJIS.length)];
  const path  = BUG_PATHS[Math.floor(Math.random() * BUG_PATHS.length)];

  bug.className = "flying-bug " + path;
  bug.textContent = emoji;

  // Position de départ près du bouton
  bug.style.bottom = (20 + Math.random()*20) + "px";
  bug.style.right  = (20 + Math.random()*20) + "px";

  document.body.appendChild(bug);

  // Nettoyage après l'animation
  setTimeout(() => bug.remove(), 5500);
}

function showEnvelope() {
  const container = document.getElementById("envelope-container");
  const envelope = document.createElement("div");
  envelope.className = "envelope-pop";
  envelope.textContent = "✉️";
  envelope.title = "Ouvrir";
  envelope.addEventListener("click", openLetter);
  container.appendChild(envelope);

  // L'enveloppe disparaît si pas cliquée après 8s
  setTimeout(() => {
    if (envelope.parentNode) envelope.remove();
  }, 8000);
}

function openLetter() {
  document.getElementById("letter-overlay").classList.add("show");
  // Retirer l'enveloppe
  const env = document.querySelector(".envelope-pop");
  if (env) env.remove();
}

document.getElementById("letter-close").addEventListener("click", () => {
  document.getElementById("letter-overlay").classList.remove("show");
});

// Fermer en cliquant en dehors de la carte
document.getElementById("letter-overlay").addEventListener("click", function(e) {
  if (e.target === this) this.classList.remove("show");
});
</script>
<body>

<header>
  <div class="hlogo">
    <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/d/d3/Institut_Pasteur_du_Maroc_logo.png/200px-Institut_Pasteur_du_Maroc_logo.png" 
      alt="Institut Pasteur du Maroc" style="height:36px;width:auto" 
      onerror="this.style.display='none'"/>
    <div>
      <div class="htitle">Leishmaniose pasteur </div>
      <div class="hsub">Morocco Climate Map</div>
    </div>
  </div>
  <div class="hright">
    <span class="tag" id="tag-rows">chargement…</span>
    <span class="tag" id="tag-communes"></span>
    <span class="tag" id="tag-geo"></span>
  </div>
</header>

<div class="app">
<aside>
  <div>
    <div class="s-title">Source des données</div>
    <div class="source-toggle">
      <div class="source-btn on" id="src-climate" onclick="setSource('climate')">Climat</div>
      <div class="source-btn" id="src-env" onclick="setSource('environment')">Environnement</div>
    </div>
  </div>

  <div>
    <div class="s-title" id="var-section-title">Variable climatique</div>
    <div class="var-grid" id="var-grid"></div>
  </div>

  <div>
    <div class="s-title">Filtrer la zone</div>
    <div class="fg"><label>Région</label>
      <select id="sel-region"><option value="">— Tout le Maroc —</option></select>
    </div>
    <div class="fg"><label>Province</label>
      <select id="sel-province" disabled><option value="">— Toutes —</option></select>
    </div>
    <div class="fg"><label>Commune</label>
      <select id="sel-commune" disabled><option value="">— Toutes —</option></select>
    </div>
  </div>

  <div>
    <div class="s-title">KPI — commune sélectionnée</div>
    <div class="kpi-row">
      <div class="kpi"><span class="kv" id="k-temp">—</span><span class="kl">T° moy (°C)</span></div>
      <div class="kpi"><span class="kv" id="k-hum">—</span><span class="kl">Humidité %</span></div>
      <div class="kpi"><span class="kv" id="k-precip">—</span><span class="kl">Précip mm</span></div>
      <div class="kpi"><span class="kv" id="k-wind">—</span><span class="kl">Vent m/s</span></div>
    </div>
  </div>

  <div>
    <div class="s-title">Environnement</div>
    <div class="env-grid" id="env-panel">
      <div class="empty-msg" style="height:50px;font-size:10px">Choisir une commune</div>
    </div>
  </div>
</aside>

<main>
  <div class="bc" id="bc"><span class="node">Maroc</span></div>

  <div class="timeline">
    <div class="tl-head">Année</div>
    <div class="ypills" id="ypills"></div>
    <div class="tl-head" style="margin-top:10px">Mois (optionnel)</div>
    <div class="mpills" id="mpills"></div>
  </div>

  <div class="map-panel">
    <div class="map-head">
      <span class="map-title" id="map-title">Température moyenne — toutes années</span>
      <span class="map-sub" id="map-sub">cliquer une commune pour le détail</span>
    </div>
    <div id="map"></div>
    <div class="legend-bar">
      <span class="legend-val" id="leg-min">—</span>
      <div class="legend-grad" id="leg-grad"></div>
      <span class="legend-val" id="leg-max">—</span>
    </div>
  </div>

  <div class="panel">
    <div class="ph">
      <span class="pt">Profil mensuel</span>
      <span class="ps" id="monthly-sub">sélectionner une commune ou région</span>
    </div>
    <div class="cw"><canvas id="c-monthly"></canvas></div>
  </div>

  <div class="cg two">
    <div class="panel">
      <div class="ph"><span class="pt">Température (°C)</span><span class="ps">série annuelle</span></div>
      <div class="cw"><canvas id="c-temp"></canvas></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="pt">Précipitations (mm)</span><span class="ps">série annuelle</span></div>
      <div class="cw"><canvas id="c-precip"></canvas></div>
    </div>
  </div>
</main>
</div>

<script>
const S = { region:"", province:"", commune:"", annee:null, mois:null, variable:"temp_mean", source:"climate" };
const ML = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Aoû","Sep","Oct","Nov","Déc"];
const MF = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"];

const RAMPS = {
  thermal: ["#1d3557","#3b82f6","#f59e0b","#e8593c","#a32d2d"],
  blue:    ["#0d2240","#185fa5","#378add","#85b7eb","#b5d4f4"],
  purple:  ["#26215c","#534ab7","#7f77dd","#afa9ec","#cecbf6"],
  amber:   ["#412402","#854f0b","#ba7517","#ef9f27","#fac775"],
};

Chart.defaults.color       = "#8a9bb8";
Chart.defaults.borderColor = "#162038";
Chart.defaults.font.family = "'Inter',sans-serif";
Chart.defaults.font.size   = 10;
Chart.defaults.plugins.legend.display = false;

function mkLine(id, color) {
  const ctx = document.getElementById(id).getContext("2d");
  return new Chart(ctx, {
    type:"line",
    data:{ labels:[], datasets:[{
      data:[], borderColor:color, backgroundColor:color+"22",
      borderWidth:2, pointRadius:2, pointHoverRadius:5,
      fill:true, tension:0.35,
    }]},
    options: {
      responsive:true, maintainAspectRatio:false,
      animation:{ duration:300 },
      plugins:{ tooltip:{ backgroundColor:"#0d1526", borderColor:"#1e2d45", borderWidth:1, padding:10 } },
      scales:{
        x:{ grid:{ color:"#111d33" }, ticks:{ color:"#3d5070", maxTicksLimit:14 } },
        y:{ grid:{ color:"#0d1526" }, ticks:{ color:"#3d5070" } },
      }
    }
  });
}

const ctx_m = document.getElementById("c-monthly").getContext("2d");
const C_monthly = new Chart(ctx_m, {
  type:"line",
  data:{ labels: ML, datasets:[
    { label:"T° moy (°C)",  data:[], borderColor:"#00d4aa", backgroundColor:"#00d4aa18",
      borderWidth:2.5, pointRadius:3, fill:true, tension:0.4 },
    { label:"Humidité (%)", data:[], borderColor:"#f59e0b", backgroundColor:"transparent",
      borderWidth:1.5, pointRadius:2, borderDash:[4,3], tension:0.4 },
  ]},
  options:{
    responsive:true, maintainAspectRatio:false, animation:{ duration:300 },
    plugins:{ legend:{ display:true, labels:{ color:"#8a9bb8", font:{ size:10 }, boxWidth:16 } } },
    scales:{
      x:{ grid:{ color:"#111d33" }, ticks:{ color:"#3d5070" } },
      y:{ grid:{ color:"#0d1526" }, ticks:{ color:"#3d5070" } },
    }
  }
});

const C = { temp: mkLine("c-temp","#00d4aa"), precip: mkLine("c-precip","#3b82f6") };

function updLine(chart, labels, data) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.update("active");
}

async function api(ep, params={}) {
  const qs = new URLSearchParams(params).toString();
  try {
    const r = await fetch(`/api/${ep}${qs?"?"+qs:""}`);
    return await r.json();
  } catch(e) { return []; }
}

function num(v, d=1) { return (v != null && !isNaN(v)) ? (+v).toFixed(d) : "—"; }
function safeArr(arr, key) { return arr.map(r => { const v = r[key]; return (v != null && !isNaN(v)) ? +v : null; }); }

let VARIABLES_DATA = null;
let leafletMap = null;
let geojsonLayer = null;
let geojsonData = null;

async function init() {
  const info = await api("db_info");
  if (info.total_rows != null) {
    document.getElementById("tag-rows").textContent     = `${(info.total_rows/1e6).toFixed(2)}M lignes`;
    document.getElementById("tag-communes").textContent = `${info.communes} communes`;
    document.getElementById("tag-geo").textContent      = info.geojson_ok
      ? `Carte: ${info.geojson_features} polygones`
      : "Carte: mode points";
    if (!info.geojson_ok) document.getElementById("tag-geo").classList.add("warn");
  }

  VARIABLES_DATA = await api("variables");
  renderVarButtons();

  const regions = await api("regions");
  const sel = document.getElementById("sel-region");
  regions.forEach(r => {
    const o = document.createElement("option");
    o.value = r; o.textContent = r; sel.appendChild(o);
  });

  const years = await api("years");
  buildYearPills(years);
  buildMonthPills();

  initMap();
  await loadMapData();
}

function renderVarButtons() {
  const data = S.source === "climate" ? VARIABLES_DATA.climate : VARIABLES_DATA.environment;
  document.getElementById("var-section-title").textContent =
    S.source === "climate" ? "Variable climatique" : "Variable environnementale";

  const keys = Object.keys(data);
  if (!keys.includes(S.variable)) S.variable = keys[0];

  const grid = document.getElementById("var-grid");
  grid.innerHTML = keys.map(k => {
    const v = data[k];
    const on = k === S.variable ? "on" : "";
    return `<div class="var-btn ${on}" onclick="pickVariable('${k}')">
      <span>${v.label}</span><span class="unit">${v.unit}</span>
    </div>`;
  }).join("");
}

function setSource(src) {
  S.source = src;
  document.getElementById("src-climate").classList.toggle("on", src === "climate");
  document.getElementById("src-env").classList.toggle("on", src === "environment");
  renderVarButtons();
  loadMapData();
}

function pickVariable(key) {
  S.variable = key;
  renderVarButtons();
  loadMapData();
}

function initMap() {
  leafletMap = L.map("map", { zoomControl: true, attributionControl: true })
    .setView([31.5, -6.5], 6);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '© OpenStreetMap, © CARTO',
    maxZoom: 12,
    minZoom: 5,
  }).addTo(leafletMap);
}

function colorForValue(value, vmin, vmax, ramp) {
  if (value == null || isNaN(value)) return "#1a2234";
  const colors = RAMPS[ramp] || RAMPS.thermal;
  const t = vmax > vmin ? (value - vmin) / (vmax - vmin) : 0.5;
  const idx = Math.min(colors.length - 1, Math.floor(t * colors.length));
  return colors[Math.max(0, idx)];
}

async function loadMapData() {
  const params = {
    variable: S.variable,
    source: S.source,
  };
  if (S.source === "climate") {
    if (S.annee) params.annee = S.annee;
    if (S.mois)  params.mois  = S.mois;
  }
  if (S.region) params.region = S.region;

  const data = await api("map_data", params);
  const varInfo = (S.source === "climate" ? VARIABLES_DATA.climate : VARIABLES_DATA.environment)[S.variable];

  updateMapTitle(varInfo);
  updateLegend(data.min, data.max, varInfo.ramp);

  if (data.geojson_available) {
    if (!geojsonData) {
      geojsonData = await api("geojson");
    }
    renderChoropleth(geojsonData, data.values, data.min, data.max, varInfo);
  } else if (data.points) {
    renderPointsMap(data.points, data.values, data.min, data.max, varInfo);
  }
}

function updateMapTitle(varInfo) {
  let suffix = "";
  if (S.source === "climate") {
    suffix = S.annee ? ` — ${S.annee}` : " — moy. 1990–2025";
    if (S.mois) suffix += ` — ${MF[S.mois-1]}`;
  } else {
    suffix = " — donnée statique";
  }
  document.getElementById("map-title").textContent = varInfo.label + suffix;
}

function updateLegend(vmin, vmax, ramp) {
  const colors = RAMPS[ramp] || RAMPS.thermal;
  document.getElementById("leg-grad").style.background =
    `linear-gradient(to right, ${colors.join(",")})`;
  document.getElementById("leg-min").textContent = num(vmin, 1);
  document.getElementById("leg-max").textContent = num(vmax, 1);
}

function renderChoropleth(geojson, values, vmin, vmax, varInfo) {
  if (geojsonLayer) {
    leafletMap.removeLayer(geojsonLayer);
  }

  geojsonLayer = L.geoJSON(geojson, {
    style: (feature) => {
      const cid = String(feature.properties.commune_id);
      const value = values[cid];
      return {
        fillColor: colorForValue(value, vmin, vmax, varInfo.ramp),
        fillOpacity: value != null ? 0.78 : 0.08,
        color: "#0d1526",
        weight: 0.6,
      };
    },
    onEachFeature: (feature, layer) => {
      const cid = String(feature.properties.commune_id);
      const value = values[cid];
      const name = feature.properties.name;

      layer.on("click", () => {
        selectCommuneFromMap(name);
      });

      layer.bindTooltip(
        `${name}${value != null ? ` — ${value} ${varInfo.unit}` : " — pas de donnée"}`,
        { sticky: true, className: "map-tooltip" }
      );

      layer.on("mouseover", function() {
        this.setStyle({ weight: 1.8, color: "#00d4aa" });
      });
      layer.on("mouseout", function() {
        geojsonLayer.resetStyle(this);
      });
    }
  }).addTo(leafletMap);
}

let pointMarkers = [];
function renderPointsMap(points, values, vmin, vmax, varInfo) {
  pointMarkers.forEach(m => leafletMap.removeLayer(m));
  pointMarkers = [];

  points.forEach(p => {
    const cid = String(p.commune_id);
    const value = values[cid];
    if (value == null) return;

    const marker = L.circleMarker([p.latitude, p.longitude], {
      radius: 5,
      fillColor: colorForValue(value, vmin, vmax, varInfo.ramp),
      fillOpacity: 0.85,
      color: "#0d1526",
      weight: 0.5,
    }).addTo(leafletMap);

    marker.bindTooltip(`${p.commune} — ${value} ${varInfo.unit}`, { sticky: true });
    marker.on("click", () => selectCommuneFromMap(p.commune));

    pointMarkers.push(marker);
  });
}

async function selectCommuneFromMap(communeName) {
  const sel = document.getElementById("sel-commune");
  let found = false;
  for (const opt of sel.options) {
    if (opt.value === communeName) { found = true; break; }
  }
  if (!found) {
    const o = document.createElement("option");
    o.value = communeName; o.textContent = communeName;
    sel.appendChild(o);
  }
  sel.disabled = false;
  sel.value = communeName;
  S.commune = communeName;
  updateBc();
  await refreshDetail();
  await refreshEnv();
}

function buildYearPills(years) {
  const c = document.getElementById("ypills");
  c.innerHTML = "";
  const all = document.createElement("button");
  all.className = "ybtn on"; all.textContent = "Toutes"; all.dataset.y = "";
  all.onclick = () => pickYear("");
  c.appendChild(all);
  years.forEach(y => {
    const b = document.createElement("button");
    b.className = "ybtn"; b.textContent = y; b.dataset.y = y;
    b.onclick = () => pickYear(y);
    c.appendChild(b);
  });
}

function buildMonthPills() {
  const c = document.getElementById("mpills");
  c.innerHTML = "";
  const all = document.createElement("button");
  all.className = "mbtn on"; all.textContent = "Année entière"; all.dataset.m = "";
  all.onclick = () => pickMonth("");
  c.appendChild(all);
  ML.forEach((m, i) => {
    const b = document.createElement("button");
    b.className = "mbtn"; b.textContent = m; b.dataset.m = i+1;
    b.onclick = () => pickMonth(i+1);
    c.appendChild(b);
  });
}

function pickYear(y) {
  S.annee = y || null;
  document.querySelectorAll(".ybtn").forEach(b =>
    b.classList.toggle("on", b.dataset.y === String(y) || (y==="" && b.dataset.y===""))
  );
  updateBc();
  loadMapData();
  refreshDetail();
}

function pickMonth(m) {
  S.mois = m || null;
  document.querySelectorAll(".mbtn").forEach(b =>
    b.classList.toggle("on", b.dataset.m === String(m) || (m==="" && b.dataset.m===""))
  );
  updateBc();
  loadMapData();
}

document.getElementById("sel-region").onchange = async function() {
  S.region = this.value; S.province = ""; S.commune = "";
  const sp = document.getElementById("sel-province");
  const sc = document.getElementById("sel-commune");
  sp.innerHTML = '<option value="">— Toutes —</option>'; sp.disabled = true;
  sc.innerHTML = '<option value="">— Toutes —</option>'; sc.disabled = true;

  if (S.region) {
    const provs = await api("provinces", { region: S.region });
    provs.forEach(p => {
      const o = document.createElement("option");
      o.value = p; o.textContent = p; sp.appendChild(o);
    });
    sp.disabled = false;

    leafletMap.fitBounds(getRegionBounds(S.region) || [[27.5,-13.5],[36,-0.9]]);
  } else {
    leafletMap.setView([31.5, -6.5], 6);
  }

  updateBc();
  loadMapData();
};

function getRegionBounds(region) {
  if (!geojsonData) return null;
  return null;
}

document.getElementById("sel-province").onchange = async function() {
  S.province = this.value; S.commune = "";
  const sc = document.getElementById("sel-commune");
  sc.innerHTML = '<option value="">— Toutes —</option>'; sc.disabled = true;

  if (S.province) {
    const coms = await api("communes", { province: S.province });
    coms.forEach(c => {
      const o = document.createElement("option");
      o.value = c; o.textContent = c; sc.appendChild(o);
    });
    sc.disabled = false;
  }
  updateBc();
};

document.getElementById("sel-commune").onchange = function() {
  S.commune = this.value;
  updateBc();
  refreshDetail();
  refreshEnv();
};

function updateBc() {
  let h = '<span class="node">Maroc</span>';
  if (S.region)   h += '<span class="sep"> &gt; </span><span class="node">'+S.region+'</span>';
  if (S.province) h += '<span class="sep"> &gt; </span><span class="node">'+S.province+'</span>';
  if (S.commune)  h += '<span class="sep"> &gt; </span><span class="node">'+S.commune+'</span>';
  if (S.annee)    h += '<span class="sep"> &gt; </span><span class="node">'+S.annee+'</span>';
  if (S.mois)     h += '<span class="sep"> &gt; </span><span class="node">'+MF[S.mois-1]+'</span>';
  document.getElementById("bc").innerHTML = h;
}

async function refreshDetail() {
  const region = S.region || (await getRegionForCommune(S.commune));
  if (!region) return;

  const data = await api("stats", { region, province: S.province, commune: S.commune });
  if (!data.length) return;

  const labels = data.map(d => d.annee);
  updLine(C.temp,   labels, safeArr(data, "temp_mean"));
  updLine(C.precip, labels, safeArr(data, "precipitation"));

  const row = S.annee ? data.find(d => String(d.annee) === String(S.annee)) : data[data.length-1];
  if (row) {
    document.getElementById("k-temp").textContent   = num(row.temp_mean, 1);
    document.getElementById("k-hum").textContent    = num(row.humidity, 0);
    document.getElementById("k-precip").textContent = num(row.precipitation, 0);
    document.getElementById("k-wind").textContent   = num(row.wind_speed, 1);
  }

  const mp = await api("monthly_profile", { region, province: S.province, commune: S.commune, annee: S.annee || "" });
  if (mp.length) {
    const bm = {};
    mp.forEach(d => { bm[d.mois] = d; });
    C_monthly.data.datasets[0].data = Array.from({length:12}, (_,i) => {
      const v = bm[i+1]?.temp_mean; return v!=null ? +parseFloat(v).toFixed(1) : null;
    });
    C_monthly.data.datasets[1].data = Array.from({length:12}, (_,i) => {
      const v = bm[i+1]?.humidity; return v!=null ? +parseFloat(v).toFixed(0) : null;
    });
    C_monthly.update("active");
    document.getElementById("monthly-sub").textContent =
      (S.commune || S.region) + (S.annee ? ` · ${S.annee}` : " · moy. toutes années");
  }
}

let regionCache = {};
async function getRegionForCommune(commune) {
  if (!commune) return S.region;
  if (regionCache[commune]) return regionCache[commune];
  const rows = await api("climate", { region: "", commune });
  return S.region;
}

async function refreshEnv() {
  if (!S.commune) return;
  const data = await api("env", { commune: S.commune });
  const panel = document.getElementById("env-panel");

  if (!data.length) {
    panel.innerHTML = `<div class="empty-msg" style="height:50px;font-size:10px">Non disponible</div>`;
    return;
  }

  const r = data[0];
  const rows = [
    ["Altitude",     r.altitude_m    != null ? (+r.altitude_m).toFixed(0)+" m"   : "—"],
    ["Type de sol",  r.soil_type     ?? "—"],
    ["Couverture",   r.land_cover    ?? "—"],
    ["Dist. eau",    r.dist_water_km != null ? (+r.dist_water_km).toFixed(1)+" km" : "—"],
    ["Population",   r.pop_density   != null ? (+r.pop_density).toFixed(0)+" hab/km²" : "—"],
    ["Aridité",      r.aridity_index != null ? (+r.aridity_index).toFixed(2)      : "—"],
    ["Classe",       r.urban_class   ?? "—"],
  ];
  panel.innerHTML = rows.map(([k,v]) =>
    `<div class="env-row"><span class="env-key">${k}</span><span class="env-val">${v}</span></div>`
  ).join("");
}

init();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    if not os.path.exists(DB_CLIMATE):
        print(f"ERREUR : {DB_CLIMATE} introuvable.")
    else:
        print("="*50)
        print("  LeishNet Climate Map Dashboard")
        print(f"  Table   : {TABLE}")
        print(f"  GeoJSON : {'OK' if GEOJSON_AVAILABLE else 'MANQUANT - mode points'}")
        print("  http://localhost:5050")
        print("="*50)
        app.run(host="0.0.0.0", port=5050, debug=False)
