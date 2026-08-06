"""
dashboard.py
============
Tableau de bord LeishNet (Flask) : carte choroplethe du Maroc par commune
(climat, environnement) + couche de RISQUE = probabilite de presence de
P. sergenti inferee par le modele bayesien (par province).

Sources de donnees (toutes generees par le pipeline, dans outputs/processed) :
  climate_morocco.db        <- extract_climate.py
  environment_morocco.db    <- build_environment.py
  communes_morocco.geojson  <- fetch_geojson.py
  psergenti_posterior_presence.csv <- src/models/bayesian_occupancy.py

Usage :
  python src/interface/dashboard.py
  puis ouvrir http://localhost:5050
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

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
    return jsonify({"climate": VARIABLES, "environment": ENV_VARIABLES, "risk": RISK_VARIABLE})


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
    })


HTML = r"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LeishNet - Carte Maroc</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg0:#F4F8FC;--bg1:#fff;--bg2:#EEF4FA;--border:#D6E3F0;--border2:#C5D5E8;
--teal:#00A896;--teal2:#00866F;--amber:#F59E0B;--rose:#E53E6A;--blue:#0857C3;--purple:#8B5CF6;
--t1:#0B1F3A;--t2:#3D5570;--t3:#7E94AC;--r:10px;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg0);color:var(--t1);font-family:Inter,system-ui,sans-serif}
header{background:#fff;border-bottom:1px solid var(--border);padding:10px 16px;display:flex;justify-content:space-between;align-items:center}
.htitle{font-size:15px;font-weight:700}
.hsub{font-size:10px;color:var(--teal);letter-spacing:.1em}
.tag{background:rgba(0,168,150,.08);border:1px solid rgba(0,168,150,.2);color:var(--teal);padding:3px 10px;border-radius:100px;font-size:10px;margin-left:6px}
.app{display:grid;grid-template-columns:280px 1fr;min-height:calc(100vh - 52px)}
aside{background:var(--bg1);border-right:1px solid var(--border);padding:16px 12px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.s-title{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--t3);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.fg{display:flex;flex-direction:column;gap:5px;margin-bottom:8px}
.fg label{font-size:11px;color:var(--t2)}
select{width:100%;background:var(--bg2);border:1px solid var(--border2);color:var(--t1);padding:8px;border-radius:var(--r);font-size:12px}
.var-btn{background:var(--bg2);border:1px solid var(--border2);color:var(--t2);padding:8px 10px;border-radius:var(--r);font-size:11px;cursor:pointer;display:flex;justify-content:space-between;margin-bottom:5px}
.var-btn.on{background:rgba(0,168,150,.14);border-color:var(--teal);color:var(--teal);font-weight:600}
.source-toggle{display:flex;gap:4px;margin-bottom:10px}
.source-btn{flex:1;background:var(--bg2);border:1px solid var(--border2);color:var(--t2);padding:6px;border-radius:var(--r);font-size:10px;cursor:pointer;text-align:center}
.source-btn.on{background:rgba(139,92,246,.15);border-color:var(--purple);color:var(--purple)}
main{padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.map-panel{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:16px}
#map{width:100%;height:520px;border-radius:10px;background:var(--bg2)}
.legend-bar{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r);margin-top:10px}
.legend-grad{flex:1;height:10px;border-radius:5px}
.legend-val{font-size:10px;color:var(--t2);min-width:42px}
.panel{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:16px}
.cw{position:relative;height:210px}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.map-title{font-size:13px;font-weight:600}
</style></head><body>
<header>
  <div><div class="htitle">Leishmaniose - Institut Pasteur Maroc</div><div class="hsub">LeishNet Climate & Risk Map</div></div>
  <div><span class="tag" id="tag-rows">...</span><span class="tag" id="tag-communes"></span><span class="tag" id="tag-risk"></span></div>
</header>
<div class="app">
<aside>
  <div><div class="s-title">Source</div>
    <div class="source-toggle">
      <div class="source-btn on" id="src-climate" onclick="setSource('climate')">Climat</div>
      <div class="source-btn" id="src-env" onclick="setSource('environment')">Env.</div>
      <div class="source-btn" id="src-risk" onclick="setSource('risk')">Risque</div>
    </div>
  </div>
  <div><div class="s-title" id="var-section-title">Variable</div><div id="var-grid"></div></div>
  <div><div class="s-title">Filtrer</div>
    <div class="fg"><label>Region</label><select id="sel-region"><option value="">Tout le Maroc</option></select></div>
    <div class="fg"><label>Province</label><select id="sel-province" disabled><option value="">Toutes</option></select></div>
    <div class="fg"><label>Commune</label><select id="sel-commune" disabled><option value="">Toutes</option></select></div>
  </div>
</aside>
<main>
  <div class="map-panel">
    <div class="map-title" id="map-title">Carte</div>
    <div id="map"></div>
    <div class="legend-bar"><span class="legend-val" id="leg-min">-</span><div class="legend-grad" id="leg-grad"></div><span class="legend-val" id="leg-max">-</span></div>
  </div>
  <div class="cg">
    <div class="panel"><div class="map-title">Temperature (serie annuelle)</div><div class="cw"><canvas id="c-temp"></canvas></div></div>
    <div class="panel"><div class="map-title">Precipitations (serie annuelle)</div><div class="cw"><canvas id="c-precip"></canvas></div></div>
  </div>
</main>
</div>
<script>
const S={region:"",province:"",commune:"",annee:null,mois:null,variable:"temp_mean",source:"climate"};
const RAMPS={thermal:["#1d3557","#3b82f6","#f59e0b","#e8593c","#a32d2d"],blue:["#0d2240","#185fa5","#378add","#85b7eb","#b5d4f4"],purple:["#26215c","#534ab7","#7f77dd","#afa9ec","#cecbf6"],amber:["#412402","#854f0b","#ba7517","#ef9f27","#fac775"]};
let VARS=null,map=null,gjLayer=null,gjData=null,ptMarkers=[];
Chart.defaults.plugins.legend.display=false;
function mkLine(id,color){return new Chart(document.getElementById(id),{type:"line",data:{labels:[],datasets:[{data:[],borderColor:color,backgroundColor:color+"22",borderWidth:2,pointRadius:2,fill:true,tension:.35}]},options:{responsive:true,maintainAspectRatio:false}});}
const C={temp:mkLine("c-temp","#00A896"),precip:mkLine("c-precip","#0857C3")};
async function api(ep,p={}){const qs=new URLSearchParams(p).toString();try{const r=await fetch(`/api/${ep}${qs?"?"+qs:""}`);return await r.json();}catch(e){return[];}}
function num(v,d=1){return(v!=null&&!isNaN(v))?(+v).toFixed(d):"-";}
async function init(){
  const info=await api("db_info");
  if(info.total_rows!=null){document.getElementById("tag-rows").textContent=`${info.total_rows} lignes`;document.getElementById("tag-communes").textContent=`${info.communes} communes`;document.getElementById("tag-risk").textContent=info.risk_ok?"Risque OK":"Risque absent";}
  VARS=await api("variables");renderVars();
  const regs=await api("regions");const sel=document.getElementById("sel-region");
  regs.forEach(r=>{const o=document.createElement("option");o.value=r;o.textContent=r;sel.appendChild(o);});
  initMap();await loadMap();
}
function curVars(){return S.source==="climate"?VARS.climate:(S.source==="risk"?VARS.risk:VARS.environment);}
function renderVars(){const d=curVars();document.getElementById("var-section-title").textContent=S.source==="risk"?"Risque":"Variable";const keys=Object.keys(d);if(!keys.includes(S.variable))S.variable=keys[0];
  document.getElementById("var-grid").innerHTML=keys.map(k=>`<div class="var-btn ${k===S.variable?"on":""}" onclick="pickVar('${k}')"><span>${d[k].label}</span><span>${d[k].unit}</span></div>`).join("");}
function setSource(s){S.source=s;["climate","env","risk"].forEach(x=>document.getElementById("src-"+x).classList.toggle("on",(x==="env"?"environment":x)===s));renderVars();loadMap();}
function pickVar(k){S.variable=k;renderVars();loadMap();}
function initMap(){map=L.map("map").setView([31.5,-6.5],6);L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{attribution:"© OSM © CARTO",maxZoom:12,minZoom:5}).addTo(map);}
function colorFor(v,mn,mx,ramp){if(v==null||isNaN(v))return"#dfe8f2";const c=RAMPS[ramp]||RAMPS.thermal;const t=mx>mn?(v-mn)/(mx-mn):.5;return c[Math.max(0,Math.min(c.length-1,Math.floor(t*c.length)))];}
async function loadMap(){
  const p={variable:S.variable,source:S.source};if(S.source==="climate"){if(S.annee)p.annee=S.annee;if(S.mois)p.mois=S.mois;}if(S.region)p.region=S.region;
  const data=await api("map_data",p);const vi=curVars()[S.variable];
  document.getElementById("map-title").textContent=vi.label;
  const c=RAMPS[vi.ramp]||RAMPS.thermal;document.getElementById("leg-grad").style.background=`linear-gradient(to right,${c.join(",")})`;
  document.getElementById("leg-min").textContent=num(data.min,2);document.getElementById("leg-max").textContent=num(data.max,2);
  if(data.geojson_available){if(!gjData)gjData=await api("geojson");renderChoro(gjData,data.values,data.min,data.max,vi);}
  else if(data.points)renderPts(data.points,data.values,data.min,data.max,vi);
}
function renderChoro(gj,vals,mn,mx,vi){if(gjLayer)map.removeLayer(gjLayer);
  gjLayer=L.geoJSON(gj,{style:f=>{const v=vals[String(f.properties.commune_id)];return{fillColor:colorFor(v,mn,mx,vi.ramp),fillOpacity:v!=null?.78:.08,color:"#0d1526",weight:.5};},
  onEachFeature:(f,l)=>{const v=vals[String(f.properties.commune_id)];l.bindTooltip(`${f.properties.name}${v!=null?` - ${v} ${vi.unit}`:" - pas de donnee"}`,{sticky:true});}}).addTo(map);}
function renderPts(pts,vals,mn,mx,vi){ptMarkers.forEach(m=>map.removeLayer(m));ptMarkers=[];
  pts.forEach(p=>{const v=vals[String(p.commune_id)];if(v==null)return;const m=L.circleMarker([p.latitude,p.longitude],{radius:5,fillColor:colorFor(v,mn,mx,vi.ramp),fillOpacity:.85,color:"#0d1526",weight:.5}).addTo(map);m.bindTooltip(`${p.commune} - ${v} ${vi.unit}`,{sticky:true});ptMarkers.push(m);});}
document.getElementById("sel-region").onchange=async function(){S.region=this.value;S.province="";S.commune="";
  const sp=document.getElementById("sel-province"),sc=document.getElementById("sel-commune");
  sp.innerHTML='<option value="">Toutes</option>';sp.disabled=true;sc.innerHTML='<option value="">Toutes</option>';sc.disabled=true;
  if(S.region){const pr=await api("provinces",{region:S.region});pr.forEach(p=>{const o=document.createElement("option");o.value=p;o.textContent=p;sp.appendChild(o);});sp.disabled=false;}
  loadMap();refreshCharts();};
document.getElementById("sel-province").onchange=async function(){S.province=this.value;S.commune="";const sc=document.getElementById("sel-commune");sc.innerHTML='<option value="">Toutes</option>';sc.disabled=true;
  if(S.province){const co=await api("communes",{province:S.province});co.forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;sc.appendChild(o);});sc.disabled=false;}refreshCharts();};
document.getElementById("sel-commune").onchange=function(){S.commune=this.value;refreshCharts();};
async function refreshCharts(){if(!S.region)return;const d=await api("stats",{region:S.region,province:S.province,commune:S.commune});if(!d.length)return;
  const labels=d.map(x=>x.annee);C.temp.data.labels=labels;C.temp.data.datasets[0].data=d.map(x=>x.temp_mean);C.temp.update();
  C.precip.data.labels=labels;C.precip.data.datasets[0].data=d.map(x=>x.precipitation);C.precip.update();}
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
        print("  LeishNet Dashboard")
        print(f"  Climat  : {DB_CLIMATE}")
        print(f"  GeoJSON : {'OK' if GEOJSON_AVAILABLE else 'MANQUANT (mode points)'}")
        print(f"  Risque  : {'OK' if RISK_BY_PROVINCE else 'absent (lance le modele)'}")
        print("  http://localhost:5050")
        print("=" * 50)
        app.run(host="0.0.0.0", port=5050, debug=False)
