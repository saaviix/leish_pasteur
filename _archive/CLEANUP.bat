@echo off
chcp 65001 >nul
echo ============================================================
echo   NETTOYAGE DU PROJET LEISHMANIOSE
echo   Les articles sont CONSERVES (deja dans data/external/)
echo ============================================================
cd /d "%~dp0"

mkdir _archive 2>nul

echo [1/6] Deplacement des dossiers obsolètes...
move /Y data_MSANTE               _archive\data_MSANTE               >nul 2>&1
move /Y modelesself               _archive\modelesself               >nul 2>&1
move /Y modeles_mathematiques     _archive\modeles_mathematiques     >nul 2>&1
move /Y "mathematical _models"    _archive\mathematical_models       >nul 2>&1
move /Y facteurs_climatiques      _archive\facteurs_climatiques      >nul 2>&1
move /Y facteurs_environnementaux _archive\facteurs_environnementaux >nul 2>&1
move /Y geographie                _archive\geographie                >nul 2>&1
move /Y interface                 _archive\interface                 >nul 2>&1
move /Y phlebotomes               _archive\phlebotomes               >nul 2>&1
move /Y plan                      _archive\plan                      >nul 2>&1
echo [OK] articles/ conserve

echo [2/6] Deplacement du schema dans docs/...
move /Y schema docs\schema.dot >nul 2>&1

echo [3/6] Suppression des .gitkeep vides...
del /F data\raw\.gitkeep            >nul 2>&1
del /F data\processed\.gitkeep      >nul 2>&1
del /F data\external\.gitkeep       >nul 2>&1
del /F outputs\processed\.gitkeep   >nul 2>&1
del /F docs\.gitkeep                >nul 2>&1

echo [4/6] Verification de la structure finale...
echo.
tree /F /A | more

echo [5/6] Verification des articles...
if exist data\external\*.pdf (
    echo [OK] articles bien preserves dans data/external/
) else (
    echo [WARN] aucun PDF dans data/external/ - verifier
)

echo [6/6] Verification des fichiers utiles...
set utile=0
if exist README.md           set utile=1
if exist requirements.txt    set utile=1
if exist run_pipeline.py     set utile=1
if exist src\data_prep\config.py set utile=1
if exist src\models\bayesian_occupancy.py set utile=1
if exist src\interface\dashboard.py set utile=1
if %utile%==1 (
    echo [OK] fichiers principaux presents
) else (
    echo [ERREUR] fichiers manquants
)

echo.
echo ============================================================
echo   NETTOYAGE TERMINE
echo   Articles : CONSERVES dans data/external/
echo   Anciens dossiers : _archive/
echo   Pour supprimer _archive/ plus tard :
echo     rmdir /S /Q _archive
echo ============================================================
pause
