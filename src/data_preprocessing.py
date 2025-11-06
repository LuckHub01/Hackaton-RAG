"""
DATA PREPROCESSING - Culture Burkinabè
Nettoyage et préparation des données pour le RAG
"""

import json
import re
from datetime import datetime
from typing import List, Dict
import unicodedata
from collections import Counter

class DataPreprocessor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.stats = {
            'total_articles': 0,
            'valid_articles': 0,
            'duplicates_removed': 0,
            'empty_content_removed': 0,
            'avg_content_length': 0
        }
    
    def clean_text(self, text: str) -> str:
        """Nettoyage approfondi du texte"""
        if not text:
            return ""
        
        # Normalisation Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Suppression des balises HTML résiduelles
        text = re.sub(r'<[^>]+>', '', text)
        
        # Suppression des caractères spéciaux multiples
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = re.sub(r'\n+', '\n', text)  # Retours à la ligne multiples
        
        # Suppression des URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Nettoyage des patterns spécifiques LeFaso.net
        text = re.sub(r'Lefaso\.net$', '', text)
        text = re.sub(r'Newsletter LeFaso\.net', '', text)
        text = re.sub(r'Lire aussi\s*:', '', text)
        
        # Suppression des mentions d'auteurs à la fin
        text = re.sub(r'\b[A-Z][a-zéèêà]+\s+[A-Z][a-zéèêà]+(\s+\([^)]+\))?\s*$', '', text)
        
        return text.strip()
    
    def extract_date(self, date_str: str) -> str:
        """Extraction et normalisation de la date"""
        if not date_str:
            return ""
        
        # Pattern: "Publié le mardi 10 octobre 2023 à 10h30min"
        match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day, month_fr, year = match.groups()
            
            # Mapping mois français
            mois = {
                'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
                'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
                'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
            }
            
            month_num = mois.get(month_fr.lower(), '01')
            return f"{year}-{month_num}-{day.zfill(2)}"
        
        return date_str
    
    def extract_metadata(self, article: Dict) -> Dict:
        """Extraction des métadonnées enrichies"""
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        full_text = f"{title} {content}"
        
        # Détection de catégories culturelles
        categories = []
        keywords = {
            'musique': ['musique', 'concert', 'artiste', 'chanson', 'album', 'festival'],
            'cinéma': ['film', 'cinéma', 'projection', 'réalisateur', 'acteur'],
            'théâtre': ['théâtre', 'pièce', 'comédien', 'spectacle'],
            'arts_visuels': ['peinture', 'exposition', 'sculpture', 'photographie'],
            'littérature': ['livre', 'écrivain', 'poésie', 'auteur', 'littérature'],
            'patrimoine': ['patrimoine', 'tradition', 'coutume', 'musée'],
            'mode': ['mode', 'styliste', 'fashion']
        }
        
        for category, words in keywords.items():
            if any(word in full_text for word in words):
                categories.append(category)
        
        # Extraction des noms d'artistes (mots en majuscules)
        artists = re.findall(r'\b[A-Z][a-zéèêà]+(?:\s+[A-Z][a-zéèêà]+)*\b', article.get('content', ''))
        artists = [a for a in artists if len(a) > 3 and a not in ['Burkina', 'Faso', 'Ouagadougou']]
        
        # Extraction des événements
        events = re.findall(r'(?:festival|concert|exposition|REMA|FESPACO|semaine)[^.]*', full_text, re.IGNORECASE)
        
        return {
            'categories': list(set(categories)),
            'artists_mentioned': list(set(artists[:5])),  # Top 5
            'events': events[:3],  # Top 3
            'word_count': len(content.split())
        }
    
    def is_valid_article(self, article: Dict) -> bool:
        """Validation de l'article"""
        # Vérifier les champs obligatoires
        if not article.get('url') or not article.get('title'):
            return False
        
        # Contenu minimum (au moins 100 caractères)
        content = article.get('content', '')
        if len(content) < 100:
            self.stats['empty_content_removed'] += 1
            return False
        
        # Vérifier que c'est bien du contenu culturel
        cultural_keywords = ['culture', 'musique', 'art', 'festival', 'cinéma', 'théâtre', 
                           'artiste', 'concert', 'exposition', 'film', 'peinture']
        text = f"{article.get('title', '')} {content}".lower()
        
        if not any(keyword in text for keyword in cultural_keywords):
            return False
        
        return True
    
    def remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Suppression des doublons"""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            # Vérifier URL et titre
            if url not in seen_urls not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_articles.append(article)
            else:
                self.stats['duplicates_removed'] += 1
        
        return unique_articles
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Découpage du texte en chunks avec overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:  # Minimum 50 mots par chunk
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def process(self):
        """Pipeline complet de preprocessing"""
        print("🔄 Chargement des données brutes...")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            raw_articles = json.load(f)
        
        self.stats['total_articles'] = len(raw_articles)
        print(f"📊 {self.stats['total_articles']} articles chargés")
        
        # Étape 1: Nettoyage
        print("\n🧹 Nettoyage des données...")
        cleaned_articles = []
        
        for article in raw_articles:
            if self.is_valid_article(article):
                cleaned = {
                    'id': article.get('url', '').split('article')[-1],
                    'url': article.get('url', ''),
                    'title': self.clean_text(article.get('title', '')),
                    'content': self.clean_text(article.get('content', '')),
                    'date': self.extract_date(article.get('date', '')),
                    'category': article.get('category', 'Culture')
                }
                
                # Ajout des métadonnées
                metadata = self.extract_metadata(cleaned)
                cleaned['metadata'] = metadata
                
                cleaned_articles.append(cleaned)
        
        self.stats['valid_articles'] = len(cleaned_articles)
        
        # Étape 2: Suppression des doublons
        print("🔍 Suppression des doublons...")
        unique_articles = self.remove_duplicates(cleaned_articles)
        
        # Étape 3: Chunking pour le RAG
        print("✂️ Découpage en chunks...")
        processed_data = []
        
        for article in unique_articles:
            # Créer des chunks du contenu
            chunks = self.chunk_text(article['content'])
            
            for idx, chunk in enumerate(chunks):
                processed_data.append({
                    'id': f"{article['id']}_chunk_{idx}",
                    'article_id': article['id'],
                    'url': article['url'],
                    'title': article['title'],
                    'content': chunk,
                    'date': article['date'],
                    'category': article['category'],
                    'metadata': article['metadata'],
                    'chunk_index': idx,
                    'total_chunks': len(chunks)
                })
        
        # Statistiques
        content_lengths = [len(a['content'].split()) for a in unique_articles]
        self.stats['avg_content_length'] = sum(content_lengths) / len(content_lengths)
        
        # Sauvegarde
        print(f"\n💾 Sauvegarde de {len(processed_data)} chunks...")
        
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_articles': len(unique_articles),
                'total_chunks': len(processed_data),
                'source': 'LeFaso.net - Culture',
                'preprocessing_stats': self.stats
            },
            'corpus': processed_data
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Rapport final
        print("\n" + "="*50)
        print("📈 RAPPORT DE PREPROCESSING")
        print("="*50)
        print(f"✅ Articles initiaux: {self.stats['total_articles']}")
        print(f"✅ Articles valides: {self.stats['valid_articles']}")
        print(f"❌ Doublons supprimés: {self.stats['duplicates_removed']}")
        print(f"❌ Contenus vides: {self.stats['empty_content_removed']}")
        print(f"📝 Longueur moyenne: {self.stats['avg_content_length']:.0f} mots")
        print(f"🎯 Chunks finaux: {len(processed_data)}")
        print(f"💾 Fichier sauvegardé: {self.output_file}")
        print("="*50)
        
        return output_data


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/raw/enriched_articles.json"
    OUTPUT_FILE = "data/processed/corpus_cleaned.json"
    
    # Exécution
    preprocessor = DataPreprocessor(INPUT_FILE, OUTPUT_FILE)
    result = preprocessor.process()
    
    print("\n✅ Preprocessing terminé avec succès!")