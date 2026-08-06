# Notes du projet

## Setup initial (une seule fois)

1. Creer un compte CDS (gratuit) : https://cds.climate.copernicus.eu/
2. Creer `C:\Users\<toi>\.cdsapirc` :
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <UID>:<API-KEY>
   ```
3. `pip install -r requirements.txt`
4. Lancer `CLEANUP.bat` pour ranger les anciens dossiers

## Lancer le pipeline

```bash
# Verification des donnees
python src/data_prep/validate_inputs.py

# Pipeline complet (avec telechargement ERA5)
python run_pipeline.py --all

# Sans telechargement (ERA5 deja present)
python run_pipeline.py --with-climate --with-env --with-analysis

# Dashboard
python src/interface/dashboard.py
# -> http://localhost:5050
```

## Commandes git

```bash
git add .
git commit -m "update"
git push origin main
```

## Problemes courants

- `ModuleNotFoundError` : verifier que `pip install -r requirements.txt` a bien tourne
- `FileNotFoundError` pour ERA5 : lancer `python src/data_prep/download_era5.py`
- `cdsapi` erreur auth : verifier `~/.cdsapirc`
- Port 5050 deja utilise : changer dans `src/interface/dashboard.py` ligne 5050


spacail distributioon : 
Processus ponctuel de Poisson 
Log-Gaussian Cox Process / GP spatial —
Régression Poisson zéro-inflatée