"""
SCRIPT D'ENRICHISSEMENT - Culture Burkinabè
Téléchargement de PDFs et scraping de sites web
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
from typing import List, Dict
import hashlib

# Pour les PDFs
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    print("⚠️ PyPDF2 non installé. Installer avec: pip install PyPDF2")
    PDF_SUPPORT = False


class DataEnricher:
    """Enrichissement de la base de données avec PDFs et sites web"""
    
    def __init__(self, output_file: str = "data/raw/enriched_articles.json"):
        self.output_file = output_file
        self.articles = []
        self.pdf_dir = "data/raw/pdfs"
        
        # Créer le dossier pour les PDFs
        os.makedirs(self.pdf_dir, exist_ok=True)
    
    def download_pdf(self, url: str) -> str:
        """Télécharger un PDF depuis une URL"""
        try:
            print(f"📥 Téléchargement: {url[:60]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Générer un nom de fichier unique
                filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
                filepath = os.path.join(self.pdf_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"   ✅ PDF téléchargé: {filename}")
                return filepath
            else:
                print(f"   ❌ Erreur {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)[:100]}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extraire le texte d'un PDF"""
        if not PDF_SUPPORT:
            return ""
        
        try:
            print(f"   📄 Extraction du texte...")
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            # Nettoyer le texte
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            print(f"   ✅ {len(text)} caractères extraits")
            return text
            
        except Exception as e:
            print(f"   ❌ Erreur extraction: {str(e)[:100]}")
            return ""
    
    def scrape_website(self, url: str) -> Dict:
        """Scraper un site web"""
        try:
            print(f"🌐 Scraping: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"   ❌ Erreur {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraire le titre
            title = ""
            if soup.find('h1'):
                title = soup.find('h1').get_text().strip()
            elif soup.find('title'):
                title = soup.find('title').get_text().strip()
            else:
                title = "Article sans titre"
            
            # Extraire le contenu principal
            content = ""
            
            # Essayer différents sélecteurs
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', class_=re.compile('content|main|article|post', re.I))
            )
            
            if main_content:
                # Supprimer scripts et styles
                for script in main_content(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extraire le texte
                paragraphs = main_content.find_all(['p', 'h2', 'h3', 'li'])
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            else:
                # Fallback: prendre tous les paragraphes
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs[:20]])
            
            # Nettoyer
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            if len(content) < 100:
                print(f"   ⚠️ Contenu trop court ({len(content)} chars)")
                return None
            
            print(f"   ✅ {len(content)} caractères extraits")
            
            return {
                'title': title[:200],
                'content': content,
                'url': url,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'category': 'Culture'
            }
            
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)[:100]}")
            return None
    
    def process_pdf_url(self, url: str) -> Dict:
        """Traiter une URL de PDF"""
        # Télécharger
        pdf_path = self.download_pdf(url)
        
        if not pdf_path:
            return None
        
        # Extraire le texte
        content = self.extract_text_from_pdf(pdf_path)
        
        if not content or len(content) < 100:
            print(f"   ⚠️ Contenu insuffisant")
            return None
        
        # Créer un titre depuis l'URL
        title = url.split('/')[-1].replace('.pdf', '').replace('-', ' ').replace('_', ' ')
        title = title[:100] if len(title) > 100 else title
        
        return {
            'title': title,
            'content': content[:10000],  # Limiter à 10k caractères
            'url': url,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': 'Culture - PDF'
        }
    
    def process_sources(self, sources: List[str]):
        """Traiter toutes les sources (PDFs et sites web)"""
        print("="*60)
        print("🚀 ENRICHISSEMENT DE LA BASE DE DONNÉES")
        print("="*60)
        
        results = {
            'success': 0,
            'failed': 0,
            'total': len(sources)
        }
        
        for i, url in enumerate(sources, 1):
            print(f"\n[{i}/{len(sources)}] Traitement de: {url[:60]}...")
            
            try:
                # Détecter si c'est un PDF
                is_pdf = url.lower().endswith('.pdf') or '/pdf/' in url.lower()
                
                if is_pdf:
                    article = self.process_pdf_url(url)
                else:
                    article = self.scrape_website(url)
                
                if article:
                    self.articles.append(article)
                    results['success'] += 1
                    print(f"   ✅ Article ajouté: {article['title'][:60]}")
                else:
                    results['failed'] += 1
                
                # Pause pour éviter le rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Erreur globale: {str(e)[:100]}")
                results['failed'] += 1
        
        # Résumé
        print("\n" + "="*60)
        print("📊 RÉSUMÉ")
        print("="*60)
        print(f"✅ Succès: {results['success']}/{results['total']}")
        print(f"❌ Échecs: {results['failed']}/{results['total']}")
        print(f"📚 Total articles: {len(self.articles)}")
        
        return results
    
    def save_articles(self):
        """Sauvegarder les articles extraits"""
        if not self.articles:
            print("⚠️ Aucun article à sauvegarder")
            return
        
        # Charger les articles existants si le fichier existe
        existing_articles = []
        existing_file = "data/raw/culture_articles.json"
        
        if os.path.exists(existing_file):
            print(f"\n📂 Chargement des articles existants: {existing_file}")
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
            print(f"   {len(existing_articles)} articles existants")
        
        # Fusionner
        all_articles = existing_articles + self.articles
        
        # Sauvegarder
        print(f"\n💾 Sauvegarde de {len(all_articles)} articles...")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Fichier sauvegardé: {self.output_file}")
        print(f"   Articles existants: {len(existing_articles)}")
        print(f"   Nouveaux articles: {len(self.articles)}")
        print(f"   Total: {len(all_articles)}")
        
        # Créer un fichier sources.txt
        sources_file = "data/raw/sources_enriched.txt"
        with open(sources_file, 'w', encoding='utf-8') as f:
            for article in all_articles:
                f.write(f"{article['url']}\n")
        
        print(f"✅ Sources sauvegardées: {sources_file}")


def main():
    """Fonction principale"""
    
    # Liste des sources à traiter
    sources = [
        # PDFs
        "https://www.ziglobitha.org/wp-content/uploads/2024/09/24-Art.-SAWADOGO-Awa-2e-Jumelle-pp.353-368.pdf",
        "https://www.maisondesculturesdumonde.org/media/mcm/188444-dossier_pe_dagogique_burkina_faso.pdf",
        "https://www.centraider.org/wp-content/uploads/sites/3/2019/03/culture-developpement.pdf",
        "https://paloc.fr/sites/paloc/files/atoms/files/2021/09/patrimoineimmateriel-burkina.pdf",
        "https://www.arcjournals.org/pdfs/ijsell/v11-i5/4.pdf",
        "https://horizon.documentation.ird.fr/exl-doc/pleins_textes/pleins_textes_6/b_fdi_43-44/010005872.pdf",
        "https://www.revue-akofena.com/wp-content/uploads/2023/02/12-T07v03-36-Mamadou-KABRE_141-154.pdf",
        "https://revues.acaref.net/wp-content/uploads/sites/3/2022/11/Jean-Paul-OUEDRAOGO.pdf",
        "https://www.ziglobitha.org/wp-content/uploads/2024/06/17-Art.-Sy-COULIBALY-pp.237-246.pdf",
        "https://dicames.online/jspui/bitstream/20.500.12177/3930/1/CheickNare.pdf",
        "https://coginta.org/wp-content/uploads/2024/03/2023-COGINTA-Etude-MARC-Burkina-Faso.pdf",
        
        # Sites web
        "https://discover-burkinafaso.com/culture-langues-religions/",
        "https://www.ontb.bf/evenements/fespaco",
        "https://www.ontb.bf/evenements/siao",
        "https://www.ontb.bf/evenements/kunde",
        "https://www.ontb.bf/evenements/jazz-a-ouaga",
        "https://www.ontb.bf/evenements/snc",
        "https://www.ontb.bf/evenements/sitho",
        "https://www.ontb.bf/evenements/nak",
        "https://www.ontb.bf/evenements/filo",
        "https://www.ontb.bf/evenements/fitd",
        "https://www.ontb.bf/evenements/boloarts",
        "https://www.ontb.bf/evenements/rendez-vous-chez-nous"
    ]
    
    print("\n🇧🇫 ENRICHISSEMENT BASE DE DONNÉES - Culture Burkinabè 🇧🇫\n")
    print(f"📋 {len(sources)} sources à traiter:")
    print(f"   - PDFs: {sum(1 for s in sources if s.lower().endswith('.pdf'))}")
    print(f"   - Sites web: {sum(1 for s in sources if not s.lower().endswith('.pdf'))}")
    
    # Créer l'enrichisseur
    enricher = DataEnricher()
    
    # Traiter les sources
    results = enricher.process_sources(sources)
    
    # Sauvegarder
    if enricher.articles:
        enricher.save_articles()
        
        print("\n" + "="*60)
        print("🎉 ENRICHISSEMENT TERMINÉ")
        print("="*60)
        print("\n📝 PROCHAINES ÉTAPES:")
        print("1. python src/data_preprocessing.py")
        print("2. python -c \"from src.rag_pipeline import CultureRAGPipeline; rag = CultureRAGPipeline('data/processed/corpus_cleaned.json'); rag.index_corpus()\"")
        print("3. streamlit run frontend/app.py")
    else:
        print("\n⚠️ Aucun article extrait. Vérifiez les URLs.")


if __name__ == "__main__":
    main()