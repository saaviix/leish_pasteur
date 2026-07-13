# Scraper de presence de Phlebotomus sergenti par commune (Maroc)

## Installation
```bash
pip install -r requirements.txt --break-system-packages
```

## Utilisation
1. Mets tes PDF d'articles dans le dossier `articles/`
2. Mets tes URLs (PubMed, PMC, pages web, PDF directs) dans `urls.txt`, une par ligne
3. Lance :
```bash
python3 main.py
```
4. Resultat : `output_phlebotomes_sergenti.csv`

## Ce que fait le script
- Extrait le texte de chaque PDF et chaque URL (gere HTML et PDF telecharge)
- Cherche les mentions de "Phlebotomus sergenti" / "Ph. sergenti" / "P. sergenti"
- Pour chaque mention, regarde quelles communes marocaines (parmi les 1503 du
  referentiel `communes_maroc_final.csv`) sont citees a proximite (~400 caracteres
  autour de la mention), insensible aux accents/majuscules
- Recupere les annees (4 chiffres) mentionnees dans le meme voisinage
- Produit UNE LIGNE PAR COMMUNE (1503 lignes), qu'il y ait une mention ou non

## Garanties anti-hallucination
- Si aucune mention n'est trouvee pour une commune : la ligne reste avec
  `sergenti_mentionne = Non`, les champs dates/sources/extrait vides, et la
  remarque "Aucune mention trouvee dans les sources fournies".
- La colonne `extrait_verification` contient un COPIE LITTERALE (pas reformulee)
  du passage source, pour que tu puisses toi-meme verifier chaque ligne "Oui"
  avant de t'en servir dans ton memoire/rapport.
- Rien n'est invente : si un PDF ou une URL echoue a se telecharger/s'ouvrir,
  c'est signale dans les logs de la console (`[ERREUR PDF]` / `[ERREUR URL]`)
  et cette source est simplement ignoree (pas de contenu de remplacement).

## Limites a connaitre
- Le "voisinage" de 400 caracteres est un choix arbitraire : une mention et
  la commune concernee peuvent parfois etre plus eloignees dans le texte
  (par exemple si elles sont dans des phrases/paragraphes differents). Tu
  peux ajuster `WINDOW` dans `matching.py` si besoin.
- Si un PDF est une image scannee (pas du texte), `pdfplumber` ne pourra
  rien extraire ; il faudrait alors de l'OCR (non inclus ici).
- Certaines URLs (PubMed notamment) montrent parfois seulement l'abstract
  et pas l'article complet en libre acces -- le script ne recupere que ce
  qui est visible sur la page/le PDF fourni.
