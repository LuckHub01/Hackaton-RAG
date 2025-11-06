"""
INTERFACE STREAMLIT - Culture Burkinabè RAG
Interface utilisateur interactive et moderne
"""

import streamlit as st
import sys
import time
from typing import Dict
import plotly.graph_objects as go
import plotly.express as px

# Import du pipeline RAG
sys.path.append('.')
from src.rag_pipeline import CultureRAGPipeline

# Configuration de la page
st.set_page_config(
    page_title="Culture Burkinabè 🇧🇫",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #009E49 0%, #EF2B2D 50%, #FCD116 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #009E49;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #EF2B2D;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #FCD116;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# @st.cache_resource
# def load_rag_pipeline():
#     """Chargement du pipeline RAG (mis en cache)"""
#     with st.spinner("🔄 Chargement du système RAG..."):
#         rag = CultureRAGPipeline(
#             corpus_file="data/processed/corpus_cleaned.json",
#             top_k=5
#         )
        
#         # Vérifier l'indexation
#         if rag.collection is None or rag.collection.count() == 0:
#             st.warning("⚠️ Indexation du corpus en cours...")
#             rag.index_corpus()
        
#         return rag

@st.cache_resource
def load_rag_pipeline():
    """Chargement du pipeline RAG (mis en cache)"""
    with st.spinner("🔄 Chargement du système RAG..."):
        rag = CultureRAGPipeline(
            corpus_file="data/processed/corpus_cleaned.json",
            top_k=5
        )
        
        # Vérifier la collection et indexer si nécessaire
        try:
            if rag.collection is None or rag.collection.count() == 0:
                st.info("⚙️ Indexation du corpus en cours...")
                rag.index_corpus()
        except Exception as e:
            st.warning(f"⚠️ Problème lors de la vérification de la collection: {str(e)}")
            st.info("🔄 Création et indexation de la collection...")
            rag.index_corpus()
        
        return rag



def display_sources(sources: list):
    """Affichage élégant des sources"""
    st.markdown("### 📚 Sources consultées")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"📄 Source {i}: {source['title'][:60]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Titre:** {source['title']}")
                st.markdown(f"**Date:** {source['date']}")
                st.markdown(f"**Lien:** [{source['url']}]({source['url']})")
            
            with col2:
                # Score de pertinence avec gauge
                score_percent = source['relevance_score'] * 100
                st.metric("Pertinence", f"{score_percent:.1f}%")


def display_metrics(result: Dict):
    """Affichage des métriques de performance"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⏱️ Temps de réponse", f"{result['response_time']:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📄 Documents consultés", result['num_docs_retrieved'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_relevance = sum(s['relevance_score'] for s in result['sources']) / len(result['sources'])
        st.metric("🎯 Pertinence moyenne", f"{avg_relevance*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Application principale"""
    
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🇧🇫 Assistant Culture Burkinabè</h1>
        <p>Posez vos questions sur la culture, la musique, le cinéma et les arts du Burkina Faso</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/31/Flag_of_Burkina_Faso.svg", width=250)
        
        st.markdown("### ⚙️ Configuration")
        
        # Options
        use_local = st.checkbox(
            "🖥️ Utiliser LLM local (Ollama)",
            value=False,
            help="Si coché, utilise Ollama (nécessite installation). Sinon, utilise HuggingFace API"
        )
        
        top_k = st.slider(
            "📄 Nombre de documents",
            min_value=3,
            max_value=10,
            value=5,
            help="Nombre de documents à consulter pour répondre"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Statistiques du système")
        
        # Chargement du pipeline
        try:
            rag = load_rag_pipeline()
            
            total_docs = rag.collection.count()
            total_articles = len(set([doc['article_id'] for doc in rag.corpus]))
            
            st.metric("📚 Articles", total_articles)
            st.metric("📄 Chunks indexés", total_docs)
            st.success("✅ Système opérationnel")
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            rag = None
        
        st.markdown("---")
        st.markdown("### 🔧 Technologies")
        st.markdown("""
        - **Embeddings:** sentence-transformers
        - **Vector DB:** ChromaDB
        - **LLM:** Mistral-7B
        - **Backend:** FastAPI
        - **Frontend:** Streamlit
        - **Licence:** 100% Open Source ✅
        """)
    
    # Zone principale
    if rag is None:
        st.error("❌ Impossible de charger le système RAG. Vérifiez que le corpus est bien présent.")
        return
    
    # Exemples de questions
    st.markdown("### 💡 Exemples de questions")
    
    col1, col2, col3 = st.columns(3)
    
    example_questions = [
        "Quels sont les principaux festivals culturels au Burkina Faso?",
        "Parle-moi de la musique burkinabè et ses artistes",
        "Qu'est-ce que le BBDA et quel est son rôle?",
        "Raconte-moi l'histoire des REMA",
        "Qui sont les artistes engagés pour la paix?",
        "Quelle est l'importance de la culture dans la cohésion sociale?"
    ]
    
    for i, col in enumerate([col1, col2, col3]):
        with col:
            for j in range(2):
                idx = i * 2 + j
                if st.button(f"💬 {example_questions[idx][:40]}...", key=f"ex_{idx}"):
                    st.session_state['question'] = example_questions[idx]
    
    st.markdown("---")
    
    # Zone de question
    st.markdown("### ❓ Posez votre question")
    
    question = st.text_area(
        "",
        value=st.session_state.get('question', ''),
        placeholder="Exemple: Quels sont les événements culturels majeurs au Burkina Faso?",
        height=100,
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Effacer", use_container_width=True)
    
    if clear_button:
        st.session_state['question'] = ''
        st.rerun()
    
    # Traitement de la question
    if search_button and question.strip():
        with st.spinner("🤔 Recherche en cours..."):
            try:
                # Mise à jour du top_k si modifié
                rag.top_k = top_k
                
                # Obtenir la réponse
                result = rag.answer_question(
                    query=question,
                    use_local_llm=use_local,
                    verbose=False
                )
                
                # Affichage de la réponse
                st.markdown("---")
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 💬 Réponse")
                st.markdown(result['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Métriques
                display_metrics(result)
                
                st.markdown("---")
                
                # Sources
                display_sources(result['sources'])
                
                # Graphique de pertinence
                st.markdown("---")
                st.markdown("### 📊 Analyse de pertinence")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Source {i+1}" for i in range(len(result['sources']))],
                        y=[s['relevance_score'] * 100 for s in result['sources']],
                        marker_color=['#009E49', '#EF2B2D', '#FCD116', '#4CAF50', '#FF9800'][:len(result['sources'])],
                        text=[f"{s['relevance_score']*100:.1f}%" for s in result['sources']],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Score de pertinence par source",
                    xaxis_title="Sources",
                    yaxis_title="Pertinence (%)",
                    yaxis_range=[0, 100],
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")
    
    elif search_button:
        st.warning("⚠️ Veuillez entrer une question")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🇧🇫 <strong>Culture Burkinabè RAG System</strong></p>
        <p>Données issues de LeFaso.net • 100% Open Source • Hackathon MTDPCE 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()