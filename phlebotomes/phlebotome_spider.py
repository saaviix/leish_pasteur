import scrapy
import pdfplumber
import pandas as pd
from pathlib import Path

class SandflySpider(scrapy.Spider):
    name = 'sandfly'
    
    start_urls = [
        "https://openmd.com/search?q=phlebotomus+sergenti",
        "https://pubmed.ncbi.nlm.nih.gov/?term=sergenti+morocco",
        "https://www.cochranelibrary.com/browse-by-topic",
        "https://www.sciencedirect.com/search?qs=phlebotomus+sergenti",
    ]

    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'USER_AGENT': 'Mozilla/5.0 (compatible; SandflyResearchBot/1.0)',
    }

    def __init__(self):
        self.results = []
        
        
        self.REQUETES_RECHERCHE = [
            "Phlebotomus sergenti Morocco", "Phlebotomus sergenti vector Leishmania tropica",
            "Phlebotomus sergenti distribution Morocco", "sand fly Phlebotomus sergenti Morocco",
            "cutaneous leishmaniasis vector Morocco", "Leishmania tropica vector Morocco",
            "Phlebotomus sergenti prevalence Morocco", "Phlebotomus sergenti in North Africa",
            "anthropophilic sand fly Morocco", "Phlebotomus sergenti leishmaniasis transmission",

            "Phlebotomus sergenti Maroc", "Phlebotomus sergenti vecteur Leishmania tropica",
            "Phlébotome sergenti Maroc", "leishmaniose cutanée tropica vecteur Maroc",
            "leishmaniose cutanée Maroc Phlebotomus", "répartition Phlebotomus sergenti Maroc",
            "vecteur de leishmaniose cutanée au Maroc", "Phlébotomes Maroc sergenti",
            "présence Phlebotomus sergenti région Maroc", "leishmaniose cutanée tropica Phlébotomus sergenti",

            "Phlebotomus sergenti Marokko", "Phlebotomus sergenti Vektor Leishmania tropica",
            "Sandfliege Phlebotomus sergenti Marokko", "kutane Leishmaniose Vektor Marokko",
            "Leishmania tropica Überträger Marokko", "Phlebotomus sergenti Verbreitung Marokko",
            "Sandmücke Phlebotomus sergenti", "kutane Leishmaniose Marokko",

            "Phlebotomus sergenti habitat Morocco", "Phlebotomus sergenti altitude distribution",
            "vector of anthroponotic cutaneous leishmaniasis Morocco", "Phlébotomus sergenti biotope Maroc",
            "Phlebotomus sergenti climate requirements", "sergenti sandfly Morocco",
            "Phlebotomus sergenti Morocco", "sergenti Leishmania tropica Morocco",
            "cutaneous leishmaniasis tropica vector Morocco",
        ]
        
        self.CATEGORIES = { ... }  

        self.leish_df = self.load_leish_csv()

    def load_leish_csv(self):
        try:
            path = Path("leish_LCT.csv")
            df = pd.read_csv(path)
            print(f"✅ {len(df)} lignes chargées")
            return df
        except Exception as e:
            print(f"❌ Erreur CSV: {e}")
            return pd.DataFrame()

    def parse(self, response):
        text = ' '.join(response.css('*:not(script):not(style)::text').getall())
        self.analyze_document(text, 'web', response.url, response.css('title::text').get(default="No Title"))

    def analyze_document(self, text, source_type, source, title=""):
        if not text or len(text) < 50 or self.leish_df.empty:
            return
        text_lower = text.lower()

        for _, row in self.leish_df.iterrows():
            localite = str(row.get('localite', row.get('commune', ''))).strip()
            province = str(row.get('province', '')).strip()
            region   = str(row.get('region', '')).strip()
            secteur  = str(row.get('secteur', '')).strip()

            terms = [t.lower() for t in [localite, province, region, secteur] if len(t) > 2]

            if any(term in text_lower for term in terms):
                sergenti = any(k in text_lower for k in ["sergenti", "phlebotomus sergenti", "phlébotome sergenti"])
                status = self.get_status(text_lower, terms)

                self.results.append({
                    'localite': localite,
                    'province': province,
                    'region': region,
                    'secteur': secteur,
                    'source_type': source_type,
                    'source': source,
                    'title': str(title)[:200],
                    'sergenti_mentioned': sergenti,
                    'presence_status': status,
                
                    'text_snippet': text[:700].replace('\n', ' ')
                })

     def create_csv(self):

    def get_status(self, text_lower, location_terms):
        
        mentioned = any(term in text_lower for term in location_terms)
        if not mentioned:
            return "Non mentionné (aucune info)"

        if any(w in text_lower for w in ["présent","present","found","détécté","signalé","confirmed","existe"]):
            return "Présence confirmée"
        if any(w in text_lower for w in ["absent","absente","not found","aucun","jamais signalé"]):
            return "Absence mentionnée"
        return "Mentionné (statut incertain)"

    def closed(self, reason):
        articles_dir = Path("articles")
        if articles_dir.exists():
            for pdf_path in articles_dir.glob("**/*.pdf"):
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                        self.analyze_document(text, 'local_pdf', str(pdf_path), pdf_path.name)
                except:
                    pass

        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv('leish_LCT_enriched.csv', index=False, encoding='utf-8')
            print(f"✅ Terminé ! {len(df)} lignes dans leish_LCT_enriched.csv")
            print("   Source de chaque info bien indiquée.")
        else:
            print("Aucun résultat.")