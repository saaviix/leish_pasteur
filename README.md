# Modelisation climat-sante de la leishmaniose cutanee (P. sergenti) au Maroc

Projet de recherche : cartographier et **inferer** la presence du phlebotome
*Phlebotomus sergenti* (vecteur de la leishmaniose cutanee a *Leishmania tropica*)
par province au Maroc, y compris la ou **les donnees manquent**, via un modele
bayesien d'occupation a lissage spatial, couple a un tableau de bord climatique.

---

## Architecture

```
E:\leishpasteur
|-- README.md
|-- requirements.txt
|-- run_pipeline.py               <- ORCHESTRATEUR (lance tout dans l'ordre)
|-- .gitignore
|
|-- data/
|   |-- raw/                      <- source de verite
|   |   |-- communes_maroc_final.csv         (1503 communes + lat/lon + region/province)
|   |   |-- leish_LCT.csv                     (cas LCT 2009-2021)
|   |   |-- phlebotomus_sergenti_par_province.csv  (statuts entomo compiles a la main)
|   |   |-- era5_morocco_2009_2021_hourly.nc  (ERA5 horaire 2009-2021, ignore par git)
|   \-- external/articles/        <- PDF d'articles (ignore par git)
|
|-- src/
|   |-- data_prep/
|   |   |-- config.py                 <- chemins centralises (importe partout)
|   |   |-- validate_inputs.py        <- VERIFICATION coherence des donnees brutes
|   |   |-- communes_by_region.py     <- REQUETE communes par region/province
|   |   |-- clean_lct.py              <- nettoyage cas + rapport donnees manquantes
|   |   |-- download_era5.py          <- TELECHARGEMENT ERA5 (climat + env) depuis CDS
|   |   |-- extract_climate.py        <- ERA5 NetCDF -> climate_morocco.db + covariables
|   |   |-- build_environment.py      <- altitude/sol/vegetation/aridite -> environment_morocco.db
|   |   |-- fetch_geojson.py          <- polygones communes -> geojson (carte)
|   |   \-- build_province_table.py   <- table province + climat + voisinage ICAR
|   |-- scraping/
|   |   |-- main.py / matching.py / text_extraction.py / urls.txt
|   |-- models/
|   |   |-- bayesian_occupancy.py     <- MODELE bayesien (inference des manques)
|   |   |-- summarize_results.py      <- resume lisible des resultats
|   |-- analysis/                    <- couche d'analyse complete
|   |   |-- seasonality.py           <- saisonnalite LCT + climat
|   |   |-- climate_response.py      <- correlations climat / presence
|   |   |-- projections.py           <- projections 2030-2100 (SSP126/SSP585)
|   |   |-- spatial.py               <- clusters spatiaux, Moran's I
|   |   |-- figures.py               <- generateur de figures publication
|   |   \-- run_analysis.py          <- lance toute l'analyse
|   \-- interface/
|       \-- dashboard.py              <- tableau de bord Flask (carte + risque)
|
|-- outputs/
|   |-- processed/   <- communes_by_region.*, lct_clean.csv, lct_missing_report.csv,
|   |                   validation_report.txt, communes_climate.csv, province_table.csv,
|   |                   adjacency_edges.npy, climate_morocco.db, environment_morocco.db,
|   |                   communes_morocco.geojson
|   |-- posterior/   <- psergenti_posterior_presence.csv, occupancy_trace.nc
|   \-- figures/     <- results_summary.txt, (figures a venir)
\-- docs/
```

---

## Telecharger les donnees ERA5 (climat + environnement) depuis le CDS

Les donnees climatiques viennent d'ERA5 (Copernicus), en **donnees horaires**
pour la periode 2009-2021, agregees en **quotidiennes** (moyenne des 24h) dans
la base SQLite, puis en moyennes mensuelles long-terme pour les covariables du
modele. **A faire une seule fois :**

1. Creer un compte gratuit : https://cds.climate.copernicus.eu/
2. Accepter la licence du dataset *"ERA5 hourly data on single levels"* ET
   *"ERA5 monthly averaged data on single levels"* (onglet Download).
3. Recuperer ta cle : https://cds.climate.copernicus.eu/how-to-api
4. Creer le fichier `~/.cdsapirc` (sous Windows : `C:\Users\<toi>\.cdsapirc`) :
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <TON-UID>:<TA-CLE-API>
   ```
5. `pip install cdsapi xarray netcdf4`

Puis :

```bash
# climat HORAIRE Maroc 2009-2021 -> data/raw/era5_morocco_2009_2021_hourly.nc
# (extract_climate.py agrege ensuite en daily -> monthly)
python src/data_prep/download_era5.py

# variante ERA5-Land (0.1 deg, plus fin)
python src/data_prep/download_era5.py --land

# couches ENVIRONNEMENTALES statiques (sol, vegetation, altitude) -> era5_env_static.nc
python src/data_prep/download_era5.py --env
```

Zone Maroc : `[Nord, Ouest, Sud, Est] = [36, -13.5, 21, -1]`.
Variables climat : temperature 2m, point de rosee (humidite), precipitations,
vent u/v, rayonnement solaire, evapotranspiration.

Format des donnees : **horaire** (24 pas par jour) dans le NetCDF brut, puis
**1 ligne par commune par jour** dans `climate_morocco.db` pour le dashboard,
et **moyennes mensuelles long-terme** dans `communes_climate.csv` pour le modele.

---

## Pipeline complet (bout en bout)

Le plus simple : l'orchestrateur.

```bash
pip install -r requirements.txt

# tout sauf telechargement / climat lourd / reseau :
python run_pipeline.py

# inclure le telechargement ERA5 (necessite ~/.cdsapirc) :
python run_pipeline.py --download --with-climate --with-env

# TOUT :
python run_pipeline.py --all
```

### Etape par etape (equivalent)

```bash
# 1. REQUETE : communes par region -> province -> communes
python src/data_prep/communes_by_region.py

# 2. Nettoyage cas + RAPPORT des donnees manquantes
python src/data_prep/clean_lct.py

# 3. TELECHARGEMENT ERA5 (climat + env) depuis le CDS (cf section ci-dessus)
python src/data_prep/download_era5.py
python src/data_prep/download_era5.py --env

# 4. Climat ERA5 -> base + covariables
python src/data_prep/extract_climate.py

# 5. Environnement (altitude/sol/vegetation ERA5, aridite)
python src/data_prep/build_environment.py

# 6. GeoJSON des communes pour la carte
python src/data_prep/fetch_geojson.py         # carres approx (rapide)
# python src/data_prep/fetch_geojson.py --osm  # vraies limites OSM (lent)

# 7. (optionnel) Scraping des articles P. sergenti
python src/scraping/main.py

# 8. Table province + voisinage spatial (ICAR) + fusion climat
python src/data_prep/build_province_table.py

# 9. MODELE BAYESIEN : inference de la presence (provinces sans donnee incluses)
python src/models/bayesian_occupancy.py

# 10. ANALYSE : graphes, correlations, projections, patterns spatiaux
python src/analysis/run_analysis.py

# 11. Resume des resultats
python src/models/summarize_results.py

# 12. TABLEAU DE BORD (carte climat + couche risque)
python src/interface/dashboard.py     # http://localhost:5050
```

---

## Traitement des donnees manquantes (le coeur)

`src/models/bayesian_occupancy.py` estime pour chaque province une probabilite
de presence `psi` de *P. sergenti*, **meme sans donnee locale**, en combinant :

- **lissage spatial ICAR** : les provinces voisines se ressemblent (graphe construit
  par `build_province_table.py`) ;
- **covariables** : latitude + climat (temperature, precipitations, aridite) quand
  `extract_climate.py` a tourne, latitude seule sinon ;
- **evidence epidemiologique** : cas LCT autochtones (detection MacKenzie 2002) ;
- **evidence entomologique** : captures confirmees (`hard`) et a verifier (`soft`).

Les provinces `no_data_gap` dans la sortie sont celles sans aucune donnee :
leur `psi_mean` est la **valeur inferee** (avec intervalle q05-q95).

---

## Tableau de bord

`src/interface/dashboard.py` (Flask + Leaflet + Chart.js) affiche :
- une carte choroplethe par commune : variables **climatiques** et **environnementales** ;
- une couche **Risque** = probabilite de presence P. sergenti issue du modele bayesien ;
- series annuelles temperature / precipitations par region/province/commune.

Il lit les bases generees dans `outputs/processed/`. Sans GeoJSON, il bascule
automatiquement en mode "points".

---

## Nettoyage du projet

Lance `CLEANUP.bat` a la racine pour deplacer tous les anciens dossiers
(`data_MSANTE/`, `modelesself/`, `modeles_mathematiques/`, `phlebotomes/`,
`articles/`, etc.) dans `_archive/`. Rien n'est supprime : tu peux verifier
puis supprimer `_archive/` une fois que tout marche.

```powershell
# depuis E:\leishpasteur
CLEANUP.bat
```

---

## Commandes rapides

```bash
# Setup initial (une seule fois)
pip install -r requirements.txt

# Verification des donnees (doit retourner 0 erreurs)
python src/data_prep/validate_inputs.py

# Pipeline complet (recommande)
python run_pipeline.py --all

# Pipeline leger (sans climat/env lourd)
python run_pipeline.py

# Uniquement le modele (si les donnees sont deja pretes)
python src/data_prep/build_province_table.py
python src/models/bayesian_occupancy.py
python src/models/summarize_results.py

# Dashboard
python src/interface/dashboard.py
# ouvrir http://localhost:5050

# Telecharger ERA5 (separement, necessite ~/.cdsapirc)
python src/data_prep/download_era5.py          # daily 2009-2021 (defaut)
python src/data_prep/download_era5.py --env    # couches environnementales statiques
```

## Notes

- Les fichiers volumineux (`.nc`, `.pdf`) restent sur disque mais sont ignores par git.
- Les anciens dossiers (`modelesself/`, `modeles_mathematiques/`, `phlebotomes/`,
  `data_MSANTE/`, `facteurs_*`, `interface/`) sont des versions historiques ; le code
  canonique vit desormais dans `src/`. Ils peuvent etre archives ou supprimes.
