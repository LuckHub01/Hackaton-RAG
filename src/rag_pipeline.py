
"""
PIPELINE RAG COMPLET - Culture Burkinabè
Version finale avec HuggingFace Router API
"""

import json
import time
from typing import List, Dict
import os

# Open Source Libraries
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()


class CultureRAGPipeline:
    """Pipeline RAG pour questions-réponses sur la culture burkinabè"""
    
    def __init__(
        self,
        corpus_file: str,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        vector_db_path: str = "data/vectors/chroma_db",
        top_k: int = 5
    ):
        """
        Initialisation du pipeline RAG
        """
        self.corpus_file = corpus_file
        self.top_k = top_k
        
        print("🚀 Initialisation du pipeline RAG...")
        
        # 1. Chargement du corpus
        print("📚 Chargement du corpus...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.corpus = data['corpus']
        
        print(f"✅ {len(self.corpus)} chunks chargés")
        
        # 2. Modèle d'embeddings
        print(f"🧠 Chargement du modèle d'embeddings: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✅ Dimension des vecteurs: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # 3. Base vectorielle ChromaDB
        print("💾 Initialisation de ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Créer ou récupérer la collection
        self.collection_name = "culture_burkina"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' chargée ({self.collection.count()} documents)")
        except:
            self.collection = None
            print("⚠️ Collection non trouvée, il faut l'indexer")
    
    def index_corpus(self):
        """Indexation du corpus dans ChromaDB"""
        print("\n🔄 INDEXATION DU CORPUS")
        print("="*50)
        
        try:
            self.chroma_client.delete_collection(self.collection_name)
            print("🗑️ Ancienne collection supprimée")
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Articles culture burkinabè - LeFaso.net"}
        )
        
        texts = []
        metadatas = []
        ids = []
        
        for doc in self.corpus:
            full_text = f"Titre: {doc['title']}\n\nContenu: {doc['content']}"
            texts.append(full_text)
            
            metadatas.append({
                'article_id': doc['article_id'],
                'url': doc['url'],
                'title': doc['title'],
                'date': doc['date'],
                'category': doc['category'],
                'chunk_index': str(doc['chunk_index']),
                'content': doc['content']
            })
            
            ids.append(doc['id'])
        
        print(f"🔢 Génération des embeddings pour {len(texts)} documents...")
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings.tolist())
            print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} traité")
        
        print("💾 Ajout des documents à ChromaDB...")
        self.collection.add(
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexation terminée: {self.collection.count()} documents")
        print("="*50)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Récupération des documents pertinents"""
        if top_k is None:
            top_k = self.top_k
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['metadatas'][0][i]['content'],
                'title': results['metadatas'][0][i]['title'],
                'url': results['metadatas'][0][i]['url'],
                'date': results['metadatas'][0][i]['date'],
                'distance': results['distances'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            }
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def generate_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Génération du prompt pour le LLM"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Titre: {doc['title']}\n"
                f"Date: {doc['date']}\n"
                f"Contenu: {doc['content'][:500]}...\n"  # Limiter la longueur
                f"Source: {doc['url']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""Tu es un assistant expert sur la culture burkinabè. Réponds à la question en te basant UNIQUEMENT sur les documents fournis ci-dessous.

DOCUMENTS DE RÉFÉRENCE:
{context}

---

QUESTION: {query}

INSTRUCTIONS:
1. Réponds en français de manière claire et précise
2. Utilise UNIQUEMENT les informations des documents ci-dessus
3. Si l'information n'est pas dans les documents, dis "Je n'ai pas trouvé cette information dans ma base de données"
4. Cite les sources en mentionnant les titres des articles
5. Structure ta réponse avec des paragraphes si nécessaire

RÉPONSE:"""
        
        return prompt
    
    def generate_answer_huggingface(self, prompt: str) -> str:
        """
        Génération avec HuggingFace Router API
        Utilise Mistral-7B via le router HuggingFace
        """
        try:
            import requests
            
            # Récupérer le token (supporte HF_TOKEN ou HUGGINGFACE_TOKEN)
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
            
            if not hf_token:
                return "❌ Token HuggingFace manquant. Ajouter HF_TOKEN ou HUGGINGFACE_TOKEN dans .env"
            
            # URL de l'API Router HuggingFace
            API_URL = "https://router.huggingface.co/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }
            
            # Payload au format OpenAI (chat completions)
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant expert sur la culture burkinabè. Réponds en français."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            print("   🔄 Appel de l'API HuggingFace Router...")
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extraire le message de la réponse
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]["content"]
                    print("   ✅ Réponse générée avec succès")
                    return message
                else:
                    return "❌ Format de réponse inattendu"
                    
            elif response.status_code == 401:
                return "❌ Token HuggingFace invalide. Vérifiez votre token dans .env"
            elif response.status_code == 429:
                return "❌ Limite de requêtes atteinte. Réessayez dans quelques minutes."
            elif response.status_code == 503:
                return "⏳ Modèle en cours de chargement. Réessayez dans 30 secondes."
            else:
                return f"❌ Erreur API: {response.status_code} - {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return "⏱️ Timeout: La requête a pris trop de temps. Réessayez."
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
    def generate_answer_local(self, prompt: str) -> str:
        """
        Génération de réponse avec LLM local (Ollama)
        Optionnel - nécessite Ollama installé
        """
        try:
            import requests
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "❌ Ollama non disponible. Installez Ollama ou utilisez HuggingFace."
        
        except Exception as e:
            return f"❌ Erreur Ollama: {str(e)}\nInstallez Ollama: https://ollama.ai/"
    
    def answer_question(
        self,
        query: str,
        use_local_llm: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Pipeline complet: Question → Réponse
        
        Args:
            query: Question de l'utilisateur
            use_local_llm: True=Ollama local, False=HuggingFace API
            verbose: Afficher les étapes
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n❓ Question: {query}")
            print("="*50)
        
        # Étape 1: Retrieval
        if verbose:
            print("🔍 Recherche de documents pertinents...")
        
        retrieved_docs = self.retrieve(query)
        
        if verbose:
            print(f"✅ {len(retrieved_docs)} documents trouvés")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"  {i}. {doc['title'][:60]}... (score: {doc['similarity_score']:.3f})")
        
        # Étape 2: Génération du prompt
        prompt = self.generate_prompt(query, retrieved_docs)
        
        # Étape 3: Génération de la réponse
        if verbose:
            print("\n🤖 Génération de la réponse...")
        
        if use_local_llm:
            answer = self.generate_answer_local(prompt)
        else:
            answer = self.generate_answer_huggingface(prompt)
        
        # Calcul du temps
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n⏱️ Temps de réponse: {elapsed_time:.2f}s")
            print("="*50)
        
        # Résultat complet
        result = {
            'question': query,
            'answer': answer,
            'sources': [
                {
                    'title': doc['title'],
                    'url': doc['url'],
                    'date': doc['date'],
                    'content': doc['content'],
                    'relevance_score': doc['similarity_score']
                }
                for doc in retrieved_docs
            ],
            'response_time': elapsed_time,
            'num_docs_retrieved': len(retrieved_docs)
        }
        
        return result


def test_rag():
    """Test rapide du pipeline"""
    
    print("\n" + "🇧🇫"*30)
    print("\nTEST DU PIPELINE RAG - Culture Burkinabè")
    print("\n" + "🇧🇫"*30 + "\n")
    
    # Initialisation
    rag = CultureRAGPipeline(
        corpus_file="data/processed/corpus_cleaned.json",
        top_k=5
    )
    
    # Indexer si nécessaire
    if rag.collection is None or rag.collection.count() == 0:
        print("⚠️ Collection non indexée. Indexation en cours...")
        rag.index_corpus()
    
    # Questions test
    questions = [
        "Quels sont les principaux festivals culturels au Burkina Faso?",
        "Parle-moi de la musique burkinabè",
        "Qui est Alif Naaba?",
        "Qu'est-ce que le BBDA?"
    ]
    
    print("\n" + "="*60)
    print("TESTS DES QUESTIONS")
    print("="*60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n\n{'='*60}")
        print(f"QUESTION {i}/{len(questions)}")
        print('='*60)
        
        result = rag.answer_question(q, use_local_llm=False)
        
        print(f"\n💬 RÉPONSE:")
        print("-"*60)
        print(result['answer'])
        print("-"*60)
        
        print(f"\n📚 Sources ({len(result['sources'])}):")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. {source['title']}")
            print(f"     Pertinence: {source['relevance_score']:.2f}")
            print(f"     {source['url']}")
        
        print(f"\n⏱️ Temps: {result['response_time']:.2f}s")
        
        time.sleep(2)  # Pause entre les questions
    
    print("\n\n" + "="*60)
    print("✅ TEST TERMINÉ")
    print("="*60)


if __name__ == "__main__":
    test_rag()