"""
geo_matching.py
================
Reconciliation floue de noms de commune contre le referentiel
communes_maroc_final.csv -- utilise par clean_lct.py (cas LCT) et
fix_mun_pop.py (population).

Strategie : correspondance exacte sur la cle normalisee (config.norm_key)
d'abord, puis correspondance floue (difflib, stdlib -- pas de dependance
supplementaire) parmi les candidats les plus proches si aucune
correspondance exacte n'a ete trouvee. Chaque correspondance floue est
journalisee avec son score : aucune correction silencieuse, tout est
tracable dans les rapports de sortie.
"""

import difflib
import re
from collections import defaultdict

import pandas as pd

import config


def build_commune_index(communes: pd.DataFrame) -> pd.DataFrame:
    """communes_maroc_final.csv avec cles normalisees (commune, province) ajoutees."""
    out = communes.copy()
    out["commune_key"] = out["commune"].map(config.norm_key)
    out["prov_key"] = out["province"].map(config.norm_key)
    return out


def candidates_by_province(communes_idx: pd.DataFrame) -> dict:
    """{prov_key: (liste_cles_normalisees, liste_noms_originaux, liste_id)}."""
    out = {}
    for prov_key, g in communes_idx.groupby("prov_key"):
        out[prov_key] = (g["commune_key"].tolist(), g["commune"].tolist(), g["id"].tolist())
    return out


def fuzzy_match(name_key: str, candidate_keys: list, cutoff: float = 0.82):
    """Renvoie (index_dans_candidate_keys, score) ou (None, 0.0)."""
    if not name_key or not candidate_keys:
        return None, 0.0
    best = difflib.get_close_matches(name_key, candidate_keys, n=1, cutoff=cutoff)
    if not best:
        return None, 0.0
    idx = candidate_keys.index(best[0])
    score = difflib.SequenceMatcher(None, name_key, best[0]).ratio()
    return idx, score


# Trouve par audit (7261 cas LCT "Commune renseignee + province valide mais
# match echoue quand meme" sur 25003 lignes) : la source brute omet souvent
# le suffixe descriptif du referentiel ("Boumalne" pour "Boumalne Dades",
# 337 cas) -- le ratio difflib sur la chaine complete tombe sous le cutoff
# a cause de la longueur, pas d'une vraie difference orthographique. Reprend
# UNIQUEMENT si un SEUL candidat de la province commence par ce nom a une
# frontiere de mot (pas juste une sous-chaine) -- rejette explicitement les
# cas ambigus type "Skoura" (5 communes "Skoura X" dans des provinces
# differentes -- reste a bon droit non reconcilie).
#
# Bidirectionnel depuis l'audit du fichier de validation manuelle (session
# 2026-08-11) : le cas symetrique existe aussi -- la source AJOUTE un mot
# que le referentiel n'a pas ("Taounate Centre" pour "Taounate", "Ain Dfali
# Centre" pour "Ain Dfali") -- meme garde-fou (candidat UNIQUE) applique
# dans les deux sens.
def _word_prefix(long_key: str, short_key: str) -> bool:
    return bool(short_key) and long_key.startswith(short_key) and (
        len(long_key) == len(short_key) or long_key[len(short_key)] == " "
    )


def prefix_match(name_key: str, candidate_keys: list):
    if not name_key:
        return None, 0.0
    hits = [
        i for i, ck in enumerate(candidate_keys)
        if _word_prefix(ck, name_key) or _word_prefix(name_key, ck)
    ]
    if len(hits) == 1:
        return hits[0], 0.85  # score synthetique, distingue de "fuzzy" pur
    return None, 0.0


# Prefixes abreges reels trouves dans le brut (ex. "Z.Cheikh" x87 -> existe
# comme "Zaouiat Cheikh" au referentiel). Applique UNIQUEMENT comme variante
# supplementaire testee en plus du nom original, jamais a la place -- et
# seulement acceptee si le nom etendu obtient un match exact ou prefixe (pas
# une simple ressemblance floue, pour limiter le risque sur une expansion qui
# peut etre fausse).
#
# "A." est AMBIGU : peut vouloir dire "Ait" (tribu) OU "Ain" (source), deux
# prefixes de commune tres frequents au Maroc -- trouve par audit (86 cas
# "A.Aicha" restaient non reconcilies malgre l'expansion "Ait Aicha" testee,
# car la vraie commune est "Aïn Aïcha" ; meme chose pour "A.Mediouna",
# "A.Maatouf"). Les DEUX expansions sont essayees, et le resultat n'est
# accepte QUE si UNE SEULE aboutit a un match (0 ou 2+ resultats -> rejet,
# jamais de choix devine silencieusement entre les deux).
ABBREV_PREFIXES = {
    "a.": ["ait ", "ain "], "a ": ["ait ", "ain "], "a,": ["ait ", "ain "],
    "z.": ["zaouiat "], "z ": ["zaouiat "], "z,": ["zaouiat "],
    "od,": ["oulad "], "od.": ["oulad "], "od ": ["oulad "],
    # "S." et "B." trouves par audit du 2eme passage (session 2026-08-12) :
    # manquaient entierement, alors que "Sidi" et "Bni"/"Beni" sont parmi les
    # prefixes de commune les plus frequents au Maroc (ex. "s,redouane" ->
    # "Sidi Redouane", "b.bouayach" -> "Bni Bouayach"). "B." est ambigu
    # comme "A." (Bni OU Beni) -- memes garde-fous.
    "s.": ["sidi "], "s,": ["sidi "], "s ": ["sidi "],
    "b.": ["bni ", "beni "], "b,": ["bni ", "beni "], "b ": ["bni ", "beni "],
}


def expand_abbreviations(raw_name: str) -> list:
    """Renvoie TOUTES les expansions plausibles d'un prefixe abrege (une
    liste, pas un seul candidat -- voir ABBREV_PREFIXES pour pourquoi)."""
    low = str(raw_name).strip().lower()
    # "Z . Cheikh" / "Z .cheikh" : variantes avec espace(s) autour du point,
    # trouve par audit (session 2026-08-12) -- sans ca, "z ." ne matche ni le
    # prefixe "z." ni "z ", et l'expansion echoue silencieusement.
    low = re.sub(r"\s*([.,])\s*", r"\1", low, count=1)
    out = []
    for prefix, expansions in ABBREV_PREFIXES.items():
        if low.startswith(prefix):
            rest = low[len(prefix):].strip()
            out.extend(exp + rest for exp in expansions)
    return out


# Trouve par audit (session 2026-08-12) : certains noms bruts omettent
# ENTIEREMENT un prefixe tribal/toponymique courant, sans meme l'abreger
# ("Taguella" pour "Ait Taguella", meme province Azilal) -- distinct du cas
# abreviation ("A.Taguella" aurait ete attrape par ABBREV_PREFIXES). Teste
# UNIQUEMENT en restriction a la province declaree (teste aussi au niveau
# national par dry-run : seulement 4 cas de plus, dont un ("saiss" -> "Zaouiat
# Saiss", Fes vers Sidi Bennour) trop incertain -- mot generique, saut
# geographique important -- donc explicitement PAS active au niveau national,
# seulement en restriction province pour rester sur, meme garde-fou candidat
# UNIQUE que partout ailleurs).
DROPPED_PREFIXES = ["ait ", "ain ", "sidi ", "bni ", "beni ", "oulad ", "zaouiat ", "moulay ", "lalla "]


def restore_dropped_prefix(name_key: str, candidate_keys: list):
    if not name_key:
        return None, 0.0
    hits = set()
    for pfx in DROPPED_PREFIXES:
        cand_key = pfx + name_key
        if cand_key in candidate_keys:
            hits.add(candidate_keys.index(cand_key))
    if len(hits) == 1:
        return hits.pop(), 0.8
    return None, 0.0


def resolve_abbreviation(commune_name: str, keys: list, uniqueness: dict = None, cutoff: float = 0.82):
    """Essaie toutes les expansions plausibles (expand_abbreviations),
    priorite exact > prefixe > flou (le nom etendu peut lui-meme contenir une
    variante orthographique du referentiel, ex. "Od,Naceur" -> "oulad
    naceur" alors que le referentiel a "Oulad Nacer" -- ni exact ni prefixe,
    mais tres proche en flou). N'accepte que si UNE SEULE expansion, TOUS
    niveaux confondus par ordre de priorite, aboutit a un candidat unique
    (evite de deviner entre 'Ait X' et 'Ain X'). Si `uniqueness` est fourni
    (contexte inter-province), filtre en plus sur l'unicite nationale du nom.
    Renvoie (idx, score, method) ou (None, 0, None)."""
    exact_hits, prefix_hits, fuzzy_hits = set(), set(), set()
    for expanded in expand_abbreviations(commune_name):
        ekey = config.norm_key(expanded)
        if uniqueness is not None and uniqueness.get(ekey, 0) != 1:
            continue
        if ekey in keys:
            exact_hits.add(keys.index(ekey))
            continue
        idx, _ = prefix_match(ekey, keys)
        if idx is not None:
            prefix_hits.add(idx)
            continue
        idx, score = fuzzy_match(ekey, keys, cutoff=cutoff)
        if idx is not None:
            fuzzy_hits.add(idx)
    if len(exact_hits) == 1:
        return next(iter(exact_hits)), 0.9, "abbrev"
    if len(exact_hits) == 0 and len(prefix_hits) == 1:
        return next(iter(prefix_hits)), 0.85, "abbrev"
    if len(exact_hits) == 0 and len(prefix_hits) == 0 and len(fuzzy_hits) == 1:
        return next(iter(fuzzy_hits)), 0.8, "abbrev"
    return None, 0.0, None


# Corrections manuelles verifiees par l'utilisateur (session 2026-08-12), pas
# des suppositions de similarite textuelle : soit le nom brut correspond a une
# commune du referentiel mais avec un ecart trop grand pour un repli textuel
# sur (ex. "Afourer" -> "Afourar", "Ait Sedrat Jbel Oulya" -> "Ait Sedrate
# Jbel El Oulia", au-dela du seuil de securite meme en repli inter-province),
# soit c'est une variante colloquiale non couverte par les regles generiques
# ("Khmiss Dades" = contraction usuelle de "Souk Lakhmis Dades"). Chaque
# entree est un fait verifie independamment, jamais une supposition.
MANUAL_OVERRIDES = {
    "afourer": "Afourar",
    "ouaouizerth": "Ouaouizaght",
    "ouaouizert": "Ouaouizaght",
    "khmiss dades": "Souk Lakhmis Dades",
    "ait sedrat jbel oulya": "Ait Sedrate Jbel El Oulia",
    # Verifie (session 2026-08-12) : Tinghir n'a qu'UN SEUL "Kalaat"/"Kelaa" et
    # qu'UN SEUL "Souk Lakhmis"/"Khmiss" dans son referentiel actuel -- aucune
    # ambiguite possible. Applique aussi aux lignes declarees "Ouarzazate" :
    # Tinghir a ete cree en 2009 par scission de l'ancienne province
    # d'Ouarzazate, la source LCT declare parfois encore l'ancien decoupage.
    # "kelaa" RESTREINT a Tinghir/Ouarzazate uniquement : 5 lignes declarent
    # "kelaa" sous la province "El Kelaa des Sraghna" elle-meme -- ce sont
    # presque certainement une reference a cette ville-la, pas a Kalaat
    # M'Gouna -- appliquer sans restriction les aurait mal redirigees.
    "kelaa": {"target": "Kalaat M'Gouna", "provinces": {"tinghir", "ouarzazate"}},
    "kelaa centre": {"target": "Kalaat M'Gouna", "provinces": {"tinghir", "ouarzazate"}},
    "khmiss": "Souk Lakhmis Dades",
    "lakhmis": "Souk Lakhmis Dades",
    "lakhmiss": "Souk Lakhmis Dades",
    # Verifie : Al Haouz n'a qu'un seul "Fadma" (Sti Fadma, le village
    # touristique connu "Setti Fatma" dans la vallee de l'Ourika).
    "s,fadma": "Sti Fadma",
    # Verifie : Essaouira n'a qu'un seul "Hanchan/Hanchane" (El Hanchane).
    "hanchen": "El Hanchane",
    "hanchene": "El Hanchane",
    # "Gueliz" (session 2026-08-12, 4e tranche) : quartier reel et bien connu
    # de Marrakech, mais la source LCT le declare sous 4 provinces sans
    # rapport (Tinghir, Taza, Al Haouz, Al Hoceima) -- erreur de saisie
    # source, pas une ambiguite de nom. Confirme par l'utilisateur : cible
    # fixe "Marrakech" quelle que soit la province declaree, restreinte aux
    # 4 provinces effectivement rencontrees dans les donnees (pas d'autres
    # provinces non verifiees). Contrairement aux autres entrees de ce dict,
    # ceci CHANGE la province effective de la ligne (le join en aval se fait
    # sur commune_id, pas sur le texte Province d'origine).
    "gueliz": {"target": "Marrakech", "provinces": {"tinghir", "taza", "al haouz", "al hoceima"}},
}


def resolve_manual_overrides(communes_idx: pd.DataFrame) -> dict:
    """{cle_normalisee_brute: {commune_std, commune_id, provinces}} -- resout
    MANUAL_OVERRIDES contre le referentiel une seule fois. `provinces` est
    None (applique quelle que soit la province declaree) ou un set de
    prov_key auquel restreindre l'override (cas d'un nom ambigu selon la
    province declaree, ex. "kelaa"). Leve une erreur explicite si une cible a
    disparu du referentiel plutot que d'echouer silencieusement."""
    by_key = communes_idx.drop_duplicates("commune_key").set_index("commune_key")
    out = {}
    for raw_key, spec in MANUAL_OVERRIDES.items():
        if isinstance(spec, dict):
            target_name, provinces = spec["target"], spec["provinces"]
        else:
            target_name, provinces = spec, None
        target_key = config.norm_key(target_name)
        if target_key not in by_key.index:
            raise ValueError(f"MANUAL_OVERRIDES : cible '{target_name}' introuvable dans le referentiel actuel")
        row = by_key.loc[target_key]
        out[raw_key] = {"commune_std": row["commune"], "commune_id": row["id"], "provinces": provinces}
    return out


# Repli explicite demande par l'utilisateur (session 2026-08-12) pour des
# noms qu'il a identifies comme des LOCALITES reelles (pas des communes) dans
# une province connue -- pas moyen de les rattacher a la commune exacte qui
# les contient, donc repli sur le chef-lieu de la province (une vraie
# commune, avec de vraies coordonnees, mais PAS necessairement le bon
# emplacement precis -- une approximation assumee et tracee, pas une
# fabrication). Match_method distinct ("province_capital_fallback") pour
# rester audit-able separement des vraies identifications.
#
# NB S.Y.B.Z et S.H.M'ed sont declares "Taounate" dans la source LCT, pas
# "Azilal" -- l'utilisateur a groupe les 3 abreviations (S.A.B.B/S.Y.B.Z/
# S.H.M'ed) sous une seule instruction "Azilal" ; suivi tel quel ici, a
# reverifier si ce n'etait pas un raccourci pour "chacune sa province
# declaree" plutot qu'un fait verifie comme les autres entrees.
# Chefs-lieux de province verifies (existent comme commune du meme nom -- ou
# d'une variante proche -- dans le referentiel). Sert au repli dynamique
# SAME_PROVINCE_CAPITAL ci-dessous : au lieu d'une cible fixe, utilise LA
# PROVINCE DECLAREE DE LA LIGNE elle-meme -- gere nativement les noms
# declares sous plusieurs provinces differentes (ex. "ait idir" a la fois
# Tinghir et Ouarzazate : chacun tombe sur son propre chef-lieu). Seules les
# provinces effectivement rencontrees sont verifiees ici, pas les 76 -- pas
# de generalisation non verifiee.
PROVINCE_CAPITALS = {
    "taza": "Taza", "azilal": "Azilal", "chichaoua": "Chichaoua",
    "taroudant": "Taroudant", "sidi kacem": "Sidi Kacem", "tinghir": "Tinghir",
    "ouezzane": "Ouazzane", "agadir ida ou tanane": "Agadir",
    "ouarzazate": "Ouarzazate", "fquih ben salah": "Fkih Ben Salah",
    "settat": "Settat", "al haouz": "Tahannaout", "beni mellal": "Beni Mellal",
    # Ajoutes session 2026-08-12, 3e tranche -- tous verifies existants dans
    # le referentiel avant usage (meme nom que la province sauf El Kelaa,
    # variante orthographique verifiee).
    "el kelaa des sraghna": "El Kelâat Es-Sraghna", "boulemane": "Boulemane",
    "sefrou": "Sefrou", "nador": "Nador", "larache": "Larache",
    "moulay yacoub": "Moulay Yacoub", "tetouan": "Tétouan", "driouch": "Driouch",
    "al hoceima": "Al Hoceïma", "essaouira": "Essaouira",
    # Ajoutes session 2026-08-12, 4e tranche -- meme nom que la province,
    # verifie existant dans le referentiel (cf. verification programmatique
    # avant ajout, aucune supposition). "taounate" comblait un trou reel :
    # 4 entrees du repli dynamique la referencaient deja depuis la 3e tranche
    # mais echouaient silencieusement (PROVINCE_CAPITALS.get() renvoyait None)
    # faute de cette entree -- bug trouve par audit systematique, pas suppose.
    "taounate": "Taounate", "chefchaouen": "Chefchaouen", "errachidia": "Errachidia",
    "guercif": "Guercif", "khenifra": "Khénifra", "meknes": "Meknès",
    "safi": "Safi", "sidi slimane": "Sidi Slimane", "tiznit": "Tiznit",
    "guelmim": "Guelmim", "fes": "Fès",
    # Chef-lieu different du nom de la province (verifie existant, meme
    # logique que "al haouz"->"Tahannaout" ou "el kelaa..."->"El Kelaat..."
    # ci-dessus) :
    "chtouka ait baha": "Ait Baha", "rehamna": "Ben Guerir", "tanger assilah": "Tanger",
    # 5e tranche (session 2026-08-12) : completion des 76 provinces du
    # referentiel pour permettre un repli generique (voir
    # `catchall_province_capital` plus bas), plutot que d'ajouter les
    # provinces une a une a chaque nouvelle serie de noms. Meme nom que la
    # province, verifie existant (aucune supposition) :
    "aousserd": "Aousserd", "benslimane": "Benslimane", "berkane": "Berkane",
    "berrechid": "Berrechid", "boujdour": "Boujdour", "casablanca": "Casablanca",
    "el hajeb": "El Hajeb", "el jadida": "El Jadida", "figuig": "Figuig",
    "ifrane": "Ifrane", "jerada": "Jerada", "kenitra": "Kénitra",
    "khemisset": "Khémisset", "khouribga": "Khouribga", "laayoune": "Laâyoune",
    "marrakech": "Marrakech", "midelt": "Midelt", "mohammedia": "Mohammedia",
    "nouaceur": "Nouaceur", "rabat": "Rabat", "sale": "Salé",
    "sidi bennour": "Sidi Bennour", "sidi ifni": "Sidi Ifni", "tan tan": "Tan-Tan",
    "taourirt": "Taourirt", "tarfaya": "Tarfaya", "tata": "Tata",
    "youssoufia": "Youssoufia", "zagora": "Zagora",
    # Chef-lieu different du nom de la province (verifie existant) :
    "mediouna": "Ain Harrouda", "fahs anjra": "Anjra", "m diq fnideq": "M'Diq",
    "inezgane ait melloul": "Inezgane", "assa zag": "Assa", "es semara": "Smara",
    "oued ed dahab": "Dakhla", "oujda angad": "Oujda", "skhirate temara": "Témara",
    # "guigou" delibirement omise : le referentiel la liste comme province a
    # part entiere mais ce nom ne correspond a aucune des 76 provinces
    # officielles du Maroc -- tres probablement une erreur de donnees dans le
    # referentiel lui-meme. Non utilisee dans les donnees LCT actuelles (0
    # ligne) -- laissee non resolue plutot que de deviner un chef-lieu pour
    # une entite dont l'existence meme est douteuse.
}
SAME_PROVINCE_CAPITAL = "__same_province_capital__"

PROVINCE_CAPITAL_FALLBACK = {
    # Verifie (session 2026-08-16, recherche web) : "Bab Zitouna" est
    # litteralement un quartier de la ville de Taza elle-meme (pas une
    # localite distincte) -- ce repli n'est donc pas une approximation mais
    # une identification correcte, confirmee apres coup.
    "bab zitouna": "Taza",
    "babzitouna": "Taza",
    # "Ait Attab" (Azilal) : aucune coordonnee fiable trouvee malgre
    # recherche (session 2026-08-16) -- reste une approximation chef-lieu
    # assumee, pas une identification exacte comme Bab Zitouna ci-dessus.
    "ait attab": "Azilal",
    "sam": "Chichaoua",
    # 1 des 113 lignes "bab tete" est declaree Ouarzazate (pas Taza) --
    # restreint au cas majoritaire verifie, la ligne isolee reste non
    # reconciliee plutot que d'etre redirigee sans justification.
    "bab tete": {"target": "Taza", "provinces": {"taza"}},
    "babtete": {"target": "Taza", "provinces": {"taza"}},
    "chafarni": "Taroudant",
    # "ain dorrij" declaree a la fois Ouezzane (54) et Sidi Kacem (31) --
    # instruction explicite de l'utilisateur : les deux vers Ouazzane (le
    # chef-lieu), la province Sidi Kacem etant jugee une erreur de saisie.
    "ain dorrij": "Ouazzane",
    "ain dorij": "Ouazzane",
    "tanfarda": "Azilal",
    "s.a.b.b": "Azilal",
    "s.y.b.z": "Azilal",
    "s.h.m ed": "Azilal",
    "belksiri": "Sidi Kacem",
    "mcissi": "Tinghir",
    # Session 2026-08-16 : coordonnees reelles trouvees par recherche web et
    # ajoutees au referentiel (communes_maroc_final.csv, id 1504-1506) --
    # "Ouzoud" et "Tamraght" ne sont donc plus une approximation chef-lieu,
    # mais une identification a la localite exacte.
    "ouzoud": "Ouzoud",
    "tamraght": "Tamraght",
    # Deuxieme tranche (meme instruction utilisateur, etendue a tous les noms
    # restants sans candidat suffisamment sur) : repli sur LA PROVINCE
    # DECLAREE de chaque ligne (pas un nom fixe) -- gere "ait idir" declare a
    # la fois Tinghir (56 cas) et Ouarzazate (6 cas) correctement, chacun sur
    # son propre chef-lieu.
    # "iminoulaoun"/"iminoulaouen" : coordonnees reelles trouvees (session
    # 2026-08-16, centroide d'une boite englobante ~30km, moins precis que
    # Ouzoud/Tamraght mais nettement mieux que le chef-lieu de province) --
    # cible fixe desormais, plus un repli dynamique sur le chef-lieu.
    "iminoulaoun": "Imi N'Oulaoune",
    # "tizgui" (Azilal) : aucune coordonnee fiable trouvee (session
    # 2026-08-16) -- reste sur le repli chef-lieu de province.
    "tizgui": SAME_PROVINCE_CAPITAL,
    "iminifri": SAME_PROVINCE_CAPITAL,
    "ait hamd": SAME_PROVINCE_CAPITAL,
    "takhsayte": SAME_PROVINCE_CAPITAL,
    "sarghine": SAME_PROVINCE_CAPITAL,
    "bouzemlan": SAME_PROVINCE_CAPITAL,
    "bouzemlane": SAME_PROVINCE_CAPITAL,
    "bouzamlane": SAME_PROVINCE_CAPITAL,
    "ait idir": SAME_PROVINCE_CAPITAL,
    "lbrouj": SAME_PROVINCE_CAPITAL,
    "zegotta": SAME_PROVINCE_CAPITAL,
    "bab mrouj": SAME_PROVINCE_CAPITAL,
    "taguela": SAME_PROVINCE_CAPITAL,
    "khenichate": SAME_PROVINCE_CAPITAL,
    "bni kolla": SAME_PROVINCE_CAPITAL,
    "my brahim": SAME_PROVINCE_CAPITAL,
    "foum anser": SAME_PROVINCE_CAPITAL,
    "lkhmiss dades": SAME_PROVINCE_CAPITAL,
    # Troisieme tranche (session 2026-08-12) : 81 noms, chacun avec la
    # province EXACTE confirmee par l'utilisateur (verifiee coherente a 100%
    # avec la province deja declaree dans la source LCT pour ce nom -- 0
    # conflit trouve sur les 90 entrees soumises, 9 deja resolues par le fix
    # S./B. et donc omises ici). Restreint explicitement a cette/ces
    # province(s) -- PAS un repli ouvert -- car plusieurs de ces noms
    # (ex. "Menara") apparaissent aussi sous d'autres provinces NON
    # confirmees ailleurs dans le jeu de donnees, qui restent a bon droit
    # non reconciliees.
    'a s jbel sofla': {"dynamic": True, "provinces": {'tinghir'}},
    'a,chefk': {"dynamic": True, "provinces": {'al haouz'}},
    'a,mediona': {"dynamic": True, "provinces": {'taounate'}},
    'aderj': {"dynamic": True, "provinces": {'sefrou'}},
    'ain doreij': {"dynamic": True, "provinces": {'ouezzane'}},
    'ait guirte': {"dynamic": True, "provinces": {'azilal'}},
    'ait tazarine': {"dynamic": True, "provinces": {'tinghir'}},
    'ait yassine': {"dynamic": True, "provinces": {'tinghir'}},
    'al wahda': {"dynamic": True, "provinces": {'taza'}},
    'amajaw': {"dynamic": True, "provinces": {'driouch'}},
    'amsa': {"dynamic": True, "provinces": {'tetouan'}},
    'assg': {"dynamic": True, "provinces": {'tinghir'}},
    'bab wandar': {"dynamic": True, "provinces": {'taounate'}},
    'bani saden': {"dynamic": True, "provinces": {'sefrou'}},
    'beni hassen': {"dynamic": True, "provinces": {'azilal'}},
    'beni sadden': {"dynamic": True, "provinces": {'sefrou'}},
    'bin jradi': {"dynamic": True, "provinces": {'taza'}},
    'bouachiba': {"dynamic": True, "provinces": {'azilal'}},
    'bouazayer': {"dynamic": True, "provinces": {'azilal'}},
    'boujediane': {"dynamic": True, "provinces": {'larache'}},
    'bourmane': {"dynamic": True, "provinces": {'ouarzazate'}},
    'boutaghrar': {"dynamic": True, "provinces": {'fquih ben salah', 'tinghir'}},
    'boutghrar': {"dynamic": True, "provinces": {'tinghir'}},
    'bouzemlane,c': {"dynamic": True, "provinces": {'taza'}},
    'bouzmlne': {"dynamic": True, "provinces": {'taza'}},
    'chahid baha': {"dynamic": True, "provinces": {'sefrou'}},
    'chtioui': {"dynamic": True, "provinces": {'fquih ben salah'}},
    'daour jdid': {"dynamic": True, "provinces": {'fquih ben salah'}},
    'el kelaa': {"dynamic": True, "provinces": {'el kelaa des sraghna'}},
    'fariata': {"dynamic": True, "provinces": {'beni mellal'}},
    'ghar nhal': {"dynamic": True, "provinces": {'beni mellal'}},
    'ghdira hamra': {"dynamic": True, "provinces": {'beni mellal'}},
    'ghorm el alem': {"dynamic": True, "provinces": {'beni mellal'}},
    'hanchan': {"dynamic": True, "provinces": {'essaouira'}},
    'hsseya': {"dynamic": True, "provinces": {'tinghir'}},
    'iaazazen': {"dynamic": True, "provinces": {'nador'}},
    'ighilamgone': {"dynamic": True, "provinces": {'tinghir'}},
    'ighilnoumgoune': {"dynamic": True, "provinces": {'tinghir'}},
    # Cible fixe depuis session 2026-08-16 (coordonnees reelles ajoutees au
    # referentiel, cf. commentaire pres de "iminoulaoun" plus haut) -- avant,
    # repli dynamique sur le chef-lieu de la province declaree.
    'iminoulaouen': {"target": "Imi N'Oulaoune", "provinces": {'ouarzazate'}},
    'imzilne': {"dynamic": True, "provinces": {'tinghir'}},
    'kabdani': {"dynamic": True, "provinces": {'driouch'}},
    'kansara': {"dynamic": True, "provinces": {'moulay yacoub'}},
    'kouacem': {"dynamic": True, "provinces": {'fquih ben salah'}},
    'lakhemis': {"dynamic": True, "provinces": {'tinghir'}},
    'lakhmiss dades': {"dynamic": True, "provinces": {'ouarzazate'}},
    'lamzem': {"dynamic": True, "provinces": {'el kelaa des sraghna'}},
    'lekhmiss': {"dynamic": True, "provinces": {'tinghir'}},
    # "menara" confirme aussi sous Chichaoua (4e tranche) -- s'ajoute a Al
    # Haouz (3e tranche), toujours restreint (le nom apparait aussi sous 10
    # AUTRES provinces non confirmees dans les donnees, cf. verification).
    'menara': {"dynamic": True, "provinces": {'al haouz', 'chichaoua'}},
    'mesker': {"dynamic": True, "provinces": {'sidi kacem'}},
    'my bouchta': {"dynamic": True, "provinces": {'taounate'}},
    'mzem': {"dynamic": True, "provinces": {'el kelaa des sraghna'}},
    'neknafa': {"dynamic": True, "provinces": {'essaouira'}},
    'o akalay': {"dynamic": True, "provinces": {'agadir ida ou tanane'}},
    'ouad elbour': {"dynamic": True, "provinces": {'chichaoua'}},
    'ourbiaa': {"dynamic": True, "provinces": {'beni mellal'}},
    'q,militaire': {"dynamic": True, "provinces": {'taza'}},
    'ras alma': {"dynamic": True, "provinces": {'moulay yacoub'}},
    's k': {"dynamic": True, "provinces": {'sidi kacem'}},
    's y b z': {"dynamic": True, "provinces": {'taounate'}},
    's,a,b,b': {"dynamic": True, "provinces": {'azilal'}},
    'sahraoua': {"dynamic": True, "provinces": {'sidi kacem'}},
    'sk': {"dynamic": True, "provinces": {'sidi kacem'}},
    'skour': {"dynamic": True, "provinces": {'boulemane'}},
    'slilou': {"dynamic": True, "provinces": {'tinghir'}},
    'somaa': {"dynamic": True, "provinces": {'beni mellal'}},
    'tabrkhachte': {"dynamic": True, "provinces": {'tinghir'}},
    'taouloklot': {"dynamic": True, "provinces": {'chichaoua'}},
    'tassouit': {"dynamic": True, "provinces": {'tinghir'}},
    'tassouite': {"dynamic": True, "provinces": {'tinghir'}},
    'tdili': {"dynamic": True, "provinces": {'azilal'}},
    'tibihit': {"dynamic": True, "provinces": {'beni mellal'}},
    'tighrmatine': {"dynamic": True, "provinces": {'tinghir'}},
    'tikiouine': {"dynamic": True, "provinces": {'agadir ida ou tanane'}},
    'timolilte': {"dynamic": True, "provinces": {'azilal'}},
    'tirest': {"dynamic": True, "provinces": {'azilal'}},
    'tisi ousli': {"dynamic": True, "provinces": {'taza'}},
    'tislit': {"dynamic": True, "provinces": {'azilal'}},
    'tizentest': {"dynamic": True, "provinces": {'taroudant'}},
    'tizqui': {"dynamic": True, "provinces": {'azilal'}},
    'yasmine': {"dynamic": True, "provinces": {'sidi kacem'}},
    'zaouia': {"dynamic": True, "provinces": {'sidi kacem'}},
    # "kelaa" declare sous SA PROPRE province (El Kelaa des Sraghna) : distinct
    # de l'entree MANUAL_OVERRIDES["kelaa"] (Tinghir/Ouarzazate -> Kalaat
    # M'Gouna) -- celle-ci ne s'applique pas ici (restriction de province non
    # satisfaite), donc pas de conflit ; ce sont bien deux "Kelaa" differentes.
    'kelaa': {"dynamic": True, "provinces": {'el kelaa des sraghna'}},
    # Cas special : Commune totalement vide dans la source (pas un nom mal
    # ecrit -- rien a matcher). BUG CORRIGE (session 2026-08-12) : cru au
    # depart que `lct[col].astype(str)` transformait un NaN en la chaine
    # litterale "nan" (vrai avec un dtype object/float classique) -- faux ici
    # car la colonne est en dtype "string" (nullable) de pandas, ou
    # `.astype(str)` PRESERVE le NaN au lieu de le stringifier. `norm_key()`
    # renvoie donc "" (chaine vide, cf. `if pd.isna(s): return ""`) pour ces
    # lignes, pas "nan" -- la cle correcte est "", verifiee ne matcher QUE les
    # 83 lignes reellement vides (aucun faux positif sur un nom qui se
    # normaliserait par coincidence en chaine vide). L'entree "nan" ci-dessus
    # ne faisait donc RIEN silencieusement -- gardee commentee comme trace de
    # l'erreur plutot que supprimee.
    'nan': SAME_PROVINCE_CAPITAL,  # mort code : jamais atteint, cf. note ci-dessus
    '': SAME_PROVINCE_CAPITAL,
    # 4e tranche (session 2026-08-12) : 63 noms supplementaires, meme
    # protocole de verification que la 3e tranche -- province utilisateur
    # confirmee identique a la province deja declaree dans la source pour
    # CE nom (0 conflit sur 70 entrees soumises, sauf "Gueliz" qui est un
    # changement de province delibere -> traite via MANUAL_OVERRIDES, pas ici).
    'sebbab': {"dynamic": True, "provinces": {'taza'}},
    'dar kebdani': {"dynamic": True, "provinces": {'nador'}},
    'ajdir': {"dynamic": True, "provinces": {'taza'}},
    'm.b.mellal': {"dynamic": True, "provinces": {'beni mellal'}},
    'oulad harti': {"dynamic": True, "provinces": {'larache'}},
    'rhouazi': {"dynamic": True, "provinces": {'taounate'}},
    'ait atab': {"dynamic": True, "provinces": {'azilal'}},
    'a.taourirt': {"dynamic": True, "provinces": {'al hoceima'}},
    'ait brahim': {"dynamic": True, "provinces": {'tinghir'}},
    'a.y.ouali': {"dynamic": True, "provinces": {'al hoceima'}},
    'taghgigt': {"dynamic": True, "provinces": {'guelmim'}},
    'c boumalne': {"dynamic": True, "provinces": {'tinghir'}},
    'sekoura': {"dynamic": True, "provinces": {'sefrou'}},
    'al ouhda': {"dynamic": True, "provinces": {'taza'}},
    'ait tazarin': {"dynamic": True, "provinces": {'tinghir'}},
    'taouima': {"dynamic": True, "provinces": {'nador'}},
    'beni garfat': {"dynamic": True, "provinces": {'larache'}},
    'farkhana': {"dynamic": True, "provinces": {'nador'}},
    'bouzmlane': {"dynamic": True, "provinces": {'taza'}},
    'bibane': {"dynamic": True, "provinces": {'taounate'}},
    'bssabssa': {"dynamic": True, "provinces": {'taounate'}},
    'al quods': {"dynamic": True, "provinces": {'taza'}},
    'dhar': {"dynamic": True, "provinces": {'essaouira'}},
    'bkakcha': {"dynamic": True, "provinces": {'settat'}},
    'skhinate': {"dynamic": True, "provinces": {'fes'}},
    'skoura': {"dynamic": True, "provinces": {'taounate'}},
    'tigoudar': {"dynamic": True, "provinces": {'essaouira'}},
    'syba': {"dynamic": True, "provinces": {'al haouz'}},
    'ouled abdellah': {"dynamic": True, "provinces": {'fquih ben salah'}},
    'lahmadna': {"dynamic": True, "provinces": {'el kelaa des sraghna'}},
    'jyarine': {"dynamic": True, "provinces": {'taza'}},
    'ghorm el alam': {"dynamic": True, "provinces": {'beni mellal'}},
    'boutghrah': {"dynamic": True, "provinces": {'tinghir'}},
    'boumlne': {"dynamic": True, "provinces": {'ouarzazate'}},
    'binlwidane': {"dynamic": True, "provinces": {'azilal'}},
    'beni aros': {"dynamic": True, "provinces": {'larache'}},
    '15 eme gare': {"dynamic": True, "provinces": {'taza'}},
    'sarghin': {"dynamic": True, "provinces": {'tinghir'}},
    'tichibite': {"dynamic": True, "provinces": {'azilal'}},
    'agoudid': {"dynamic": True, "provinces": {'azilal'}},
    'ouled mbarek': {"dynamic": True, "provinces": {'fquih ben salah'}},
    'alouahda': {"dynamic": True, "provinces": {'taza'}},
    'ighilamgoune': {"dynamic": True, "provinces": {'tinghir'}},
    'alqods': {"dynamic": True, "provinces": {'agadir ida ou tanane'}},
    'ait h said': {"dynamic": True, "provinces": {'tinghir'}},
    'ait h ousaid': {"dynamic": True, "provinces": {'tinghir'}},
    'ait abdoune': {"dynamic": True, "provinces": {'tinghir'}},
    'ait hamou ousaid': {"dynamic": True, "provinces": {'tinghir'}},
    'ras lma': {"dynamic": True, "provinces": {'moulay yacoub'}},
    'boukerra': {"dynamic": True, "provinces": {'ouezzane'}},
    'tabrkhacht': {"dynamic": True, "provinces": {'tinghir'}},
    'tabrkhcht': {"dynamic": True, "provinces": {'tinghir'}},
    'bourached': {"dynamic": True, "provinces": {'taza'}},
    'kolla': {"dynamic": True, "provinces": {'larache'}},
    'iwariden': {"dynamic": True, "provinces": {'azilal'}},
    'hcia': {"dynamic": True, "provinces": {'tinghir'}},
    'hajra': {"dynamic": True, "provinces": {'beni mellal'}},
    'babmrouj': {"dynamic": True, "provinces": {'taza'}},
    'tazitount': {"dynamic": True, "provinces": {'chichaoua'}},
    'laamarcha': {"dynamic": True, "provinces": {'settat'}},
    'as jberl': {"dynamic": True, "provinces": {'tinghir'}},
    'a s s gharbia': {"dynamic": True, "provinces": {'tinghir'}},
    'b.s.jbel': {"dynamic": True, "provinces": {'nador'}},
}


def resolve_province_capital_fallback(communes_idx: pd.DataFrame) -> dict:
    """{cle_normalisee_brute: {commune_std, commune_id, provinces} |
    {dynamic: True, provinces}} -- resout PROVINCE_CAPITAL_FALLBACK contre le
    referentiel. Une cible dynamique (SAME_PROVINCE_CAPITAL, ou un dict
    {"dynamic": True, "provinces": {...}}) n'est PAS resolue ici (elle depend
    de la province de CHAQUE ligne, connue seulement au moment du match) --
    geree dans match_commune_in_province via PROVINCE_CAPITALS. `provinces`
    restreint l'application a un set de prov_key -- None = sans restriction
    (a n'utiliser QUE si le nom brut n'a jamais ete rencontre sous une autre
    province non verifiee dans le jeu de donnees complet)."""
    by_key = communes_idx.drop_duplicates("commune_key").set_index("commune_key")
    out = {}
    for raw_key, spec in PROVINCE_CAPITAL_FALLBACK.items():
        if spec == SAME_PROVINCE_CAPITAL:
            out[raw_key] = {"dynamic": True, "provinces": None}
            continue
        if isinstance(spec, dict) and spec.get("dynamic"):
            out[raw_key] = {"dynamic": True, "provinces": spec["provinces"]}
            continue
        if isinstance(spec, dict):
            target_name, provinces = spec["target"], spec["provinces"]
        else:
            target_name, provinces = spec, None
        target_key = config.norm_key(target_name)
        if target_key not in by_key.index:
            raise ValueError(f"PROVINCE_CAPITAL_FALLBACK : cible '{target_name}' introuvable dans le referentiel actuel")
        row = by_key.loc[target_key]
        out[raw_key] = {"commune_std": row["commune"], "commune_id": row["id"], "provinces": provinces, "dynamic": False}
    return out


def catchall_province_capital(prov_key: str, by_province: dict):
    """Dernier repli (session 2026-08-12, decision explicite de l'utilisateur
    apres avoir constate que ~1400 cas restaient non reconcilies malgre les 4
    vagues de correction manuelle precedentes) : pour TOUTE ligne encore non
    reconciliee apres les stages 1 (province declaree) et 2 (inter-province,
    nom unique au national), rattache au chef-lieu de la province declaree --
    sans restriction de nom, contrairement a PROVINCE_CAPITAL_FALLBACK qui
    exige un nom en liste blanche verifie individuellement. Differe des
    autres reglages de ce module : ici, precision individuelle deliberement
    sacrifiee pour couverture maximale, un choix du proprietaire du projet et
    pas une inference. Reste honnetement trace (match_method distinct,
    match_score bas) : filtrable pour tout usage voulant s'en passer, jamais
    confondu avec une identification exacte. Renvoie None si la province n'a
    pas de chef-lieu verifie dans PROVINCE_CAPITALS (n'invente jamais)."""
    capital_name = PROVINCE_CAPITALS.get(prov_key)
    if not capital_name:
        return None
    keys, names, ids = by_province.get(prov_key, ([], [], []))
    cap_key = config.norm_key(capital_name)
    if cap_key not in keys:
        return None
    i = keys.index(cap_key)
    return {"commune_std": names[i], "commune_id": ids[i],
            "match_method": "province_capital_catchall", "match_score": 0.3}


def match_commune_in_province(commune_name: str, prov_key: str, by_province: dict, cutoff: float = 0.82,
                               manual_overrides: dict = None, province_capital_fallback: dict = None):
    """Reconcilie un nom de commune brut contre le referentiel, restreint a
    la province deja identifiee (reduit drastiquement les faux positifs et
    le cout de calcul par rapport a une recherche floue sur les 1503 communes).

    Ordre d'essai : exact -> flou (ratio complet) -> prefixe de mot (candidat
    UNIQUE dans la province) -> abreviation courante (Ait/Zaouiat/Oulad)
    reessayee sur les 3 etapes precedentes.

    Renvoie un dict : {commune_std, commune_id, match_method, match_score}
    avec match_method in {"manual_override", "exact", "fuzzy",
    "prefix_in_province", "abbrev_in_province", "unmatched"}.
    """
    key = config.norm_key(commune_name)

    if manual_overrides and key in manual_overrides:
        ov = manual_overrides[key]
        if ov["provinces"] is None or prov_key in ov["provinces"]:
            return {"commune_std": ov["commune_std"], "commune_id": ov["commune_id"],
                    "match_method": "manual_override", "match_score": 1.0}
        # restriction de province non satisfaite -> continue vers les etapes
        # normales plutot que d'appliquer un override incertain hors contexte

    keys, names, ids = by_province.get(prov_key, ([], [], []))

    if key in keys:
        i = keys.index(key)
        return {"commune_std": names[i], "commune_id": ids[i], "match_method": "exact", "match_score": 1.0}

    idx, score = fuzzy_match(key, keys, cutoff=cutoff)
    if idx is not None:
        return {"commune_std": names[idx], "commune_id": ids[idx], "match_method": "fuzzy", "match_score": round(score, 3)}

    idx, score = prefix_match(key, keys)
    if idx is not None:
        return {"commune_std": names[idx], "commune_id": ids[idx], "match_method": "prefix_in_province", "match_score": score}

    idx, score, method = resolve_abbreviation(commune_name, keys)
    if idx is not None:
        return {"commune_std": names[idx], "commune_id": ids[idx], "match_method": "abbrev_in_province", "match_score": score}

    idx, score = restore_dropped_prefix(key, keys)
    if idx is not None:
        return {"commune_std": names[idx], "commune_id": ids[idx], "match_method": "restored_prefix_in_province", "match_score": score}

    if province_capital_fallback and key in province_capital_fallback:
        fb = province_capital_fallback[key]
        if fb.get("dynamic"):
            restrict = fb.get("provinces")
            capital_name = PROVINCE_CAPITALS.get(prov_key) if (restrict is None or prov_key in restrict) else None
            if capital_name:
                cap_key = config.norm_key(capital_name)
                if cap_key in keys:
                    i = keys.index(cap_key)
                    return {"commune_std": names[i], "commune_id": ids[i],
                            "match_method": "province_capital_fallback", "match_score": 0.5}
        elif fb["provinces"] is None or prov_key in fb["provinces"]:
            return {"commune_std": fb["commune_std"], "commune_id": fb["commune_id"],
                    "match_method": "province_capital_fallback", "match_score": 0.5}

    return {"commune_std": None, "commune_id": None, "match_method": "unmatched", "match_score": 0.0}


def build_global_uniqueness(communes_idx: pd.DataFrame) -> dict:
    """{commune_key: nombre_de_provinces_differentes_portant_ce_nom}. Un nom
    de commune "generique" (ex. "Ras El Ma", present dans 10 provinces) ne
    peut pas etre resolu sans ambiguite par le seul texte -- seuls les noms
    UNIQUES au niveau national sont de bons candidats pour un repli
    inter-province (stage 2)."""
    return communes_idx.groupby("commune_key")["province"].nunique().to_dict()


def match_commune_cross_province(commune_name: str, communes_idx: pd.DataFrame,
                                  uniqueness: dict, cutoff: float = 0.82):
    """Repli (stage 2) quand la correspondance restreinte a la province
    declaree a echoue : recherche floue sur les 1503 communes, mais
    n'accepte le resultat QUE s'il est unique au niveau national (une seule
    province porte ce nom de commune) -- sinon le risque de faux positif
    (mauvaise province) est trop eleve avec la seule similarite textuelle.

    Ordre d'essai : exact -> flou (ratio complet) -> prefixe de mot
    (bidirectionnel) -> abreviation courante -- les 2 derniers ajoutes suite a
    l'audit du fichier de validation manuelle (session 2026-08-11) qui a
    trouve des cas ou la province declaree dans les donnees LCT est FAUSSE
    ET le nom a besoin d'un prefixe/abreviation ("Z.Cheikh" declare a Beni
    Mellal alors que "Zaouiat Cheikh" est en fait dans la province de
    Khenifra -- le stage 1 (in-province) ne pouvait pas trouver ca puisqu'il
    cherche uniquement dans la province annoncee). Meme garde-fou partout :
    candidat UNIQUE au niveau national.

    Renvoie un dict avec en plus `province_conflict` = True si la province du
    match differe de la province annoncee dans les donnees LCT (a documenter,
    pas a corriger silencieusement)."""
    key = config.norm_key(commune_name)
    keys = communes_idx["commune_key"].tolist()
    names = communes_idx["commune"].tolist()
    ids = communes_idx["id"].tolist()
    provs = communes_idx["province"].tolist()

    def _accept(idx, method, score):
        return {"commune_std": names[idx], "commune_id": ids[idx], "province_match": provs[idx],
                "match_method": method, "match_score": score}

    if key in keys and uniqueness.get(key, 0) == 1:
        return _accept(keys.index(key), "exact_cross_province", 1.0)

    idx, score = fuzzy_match(key, keys, cutoff=cutoff)
    if idx is not None and score >= cutoff and uniqueness.get(keys[idx], 0) == 1:
        return _accept(idx, "fuzzy_cross_province", round(score, 3))

    idx, score = prefix_match(key, keys)
    if idx is not None and uniqueness.get(keys[idx], 0) == 1:
        return _accept(idx, "prefix_cross_province", score)

    idx, score, method = resolve_abbreviation(commune_name, keys, uniqueness=uniqueness)
    if idx is not None:
        return _accept(idx, "abbrev_cross_province", score)

    return {"commune_std": None, "commune_id": None, "province_match": None,
            "match_method": "unmatched", "match_score": 0.0}


def match_commune_global(name: str, communes_idx: pd.DataFrame, cutoff: float = 0.82):
    """Reconciliation sans province connue (ex. mun_pop.csv) : recherche
    floue sur l'ensemble des 1503 communes. A n'utiliser que pour de petits
    volumes de lignes non matchees (~centaines), pas pour tout un fichier."""
    key = config.norm_key(name)
    keys = communes_idx["commune_key"].tolist()
    names = communes_idx["commune"].tolist()
    ids = communes_idx["id"].tolist()

    if key in keys:
        i = keys.index(key)
        return {"commune_std": names[i], "commune_id": ids[i], "match_method": "exact", "match_score": 1.0}

    idx, score = fuzzy_match(key, keys, cutoff=cutoff)
    if idx is not None:
        return {"commune_std": names[idx], "commune_id": ids[idx], "match_method": "fuzzy", "match_score": round(score, 3)}

    return {"commune_std": None, "commune_id": None, "match_method": "unmatched", "match_score": 0.0}
