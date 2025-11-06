"""
API FASTAPI - Culture Burkinabè RAG
Backend REST API pour le système RAG
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from rag_pipeline import CultureRAGPipeline

# Initialisation de l'app
app = FastAPI(
    title="Culture Burkinabè RAG API",
    description="API pour questions-réponses sur la culture burkinabè 🇧🇫",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS pour permettre les requêtes frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du pipeline RAG (une seule fois au démarrage)
rag_pipeline: Optional[CultureRAGPipeline] = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage du serveur"""
    global rag_pipeline
    print("🚀 Démarrage de l'API...")
    
    rag_pipeline = CultureRAGPipeline(
        corpus_file="data/processed/corpus_cleaned.json",
        top_k=5
    )
    
    # Vérifier si le corpus est indexé
    if rag_pipeline.collection is None or rag_pipeline.collection.count() == 0:
        print("⚠️ Corpus non indexé. Indexation en cours...")
        rag_pipeline.index_corpus()
    
    print("✅ API prête!")


# Modèles Pydantic pour validation
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_local_llm: Optional[bool] = False

class Source(BaseModel):
    title: str
    url: str
    date: str
    relevance_score: float

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    response_time: float
    num_docs_retrieved: int


# Routes de l'API
@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🇧🇫 Culture Burkinabè RAG API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask - Poser une question",
            "health": "/health - Vérifier le statut",
            "stats": "/stats - Statistiques du corpus",
            "docs": "/docs - Documentation Swagger"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline non initialisé")
    
    return {
        "status": "healthy",
        "corpus_size": rag_pipeline.collection.count() if rag_pipeline.collection else 0,
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "vector_db": "ChromaDB"
    }

@app.get("/stats")
async def get_stats():
    """Statistiques du corpus"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline non initialisé")
    
    return {
        "total_documents": rag_pipeline.collection.count(),
        "total_articles": len(set([doc['article_id'] for doc in rag_pipeline.corpus])),
        "collection_name": rag_pipeline.collection_name,
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "vector_database": "ChromaDB (Open Source)"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Poser une question sur la culture burkinabè
    
    Args:
        question: La question à poser
        top_k: Nombre de documents à récupérer (défaut: 5)
        use_local_llm: Utiliser Ollama local (True) ou HuggingFace API (False)
    
    Returns:
        Réponse avec sources et métadonnées
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline non initialisé")
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question vide")
    
    try:
        # Traitement de la question
        result = rag_pipeline.answer_question(
            query=request.question,
            use_local_llm=request.use_local_llm,
            verbose=False
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/retrieve")
async def retrieve_documents(request: QuestionRequest):
    """
    Récupérer uniquement les documents pertinents (sans génération)
    
    Args:
        question: La requête de recherche
        top_k: Nombre de documents à récupérer
    
    Returns:
        Liste des documents pertinents
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline non initialisé")
    
    try:
        docs = rag_pipeline.retrieve(
            query=request.question,
            top_k=request.top_k
        )
        
        return {
            "query": request.question,
            "documents": docs,
            "count": len(docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# Lancement du serveur
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )