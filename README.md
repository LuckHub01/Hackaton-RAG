# 🇧🇫 Assistant IA Culture Burkinabè

**Système RAG (Retrieval-Augmented Generation) 100% Open Source pour répondre aux questions sur la culture burkinabè**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-100%25-green)](https://opensource.org/)

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Justification du sujet](#justification-du-sujet)
3. [Architecture technique](#architecture-technique)
4. [Technologies open source](#technologies-open-source)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [Évaluation](#évaluation)
8. [Résultats](#résultats)
9. [Structure du projet](#structure-du-projet)
10. [Contributeurs](#contributeurs)

---

## 🎯 Vue d'ensemble

Ce projet est un **assistant IA contextuel** capable de répondre à des questions sur la **culture burkinabè** en s'appuyant exclusivement sur un corpus d'articles collectés depuis LeFaso.net. Le système utilise une architecture RAG (Retrieval-Augmented Generation) entièrement composée de technologies open source.

### Caractéristiques principales

✅ **100% Open Source** - Toutes les technologies utilisées sont libres et gratuites  
✅ **Données locales** - Plus de 500 articles sur la culture burkinabè  
✅ **Réponses sourcées** - Chaque réponse cite ses sources  
✅ **Interface moderne** - Application web interactive avec Streamlit  
✅ **API REST** - Backend FastAPI pour intégrations  
✅ **Évaluation rigoureuse** - 20 questions test avec métriques détaillées  

---

## 💡 Justification du sujet

### Pourquoi la Culture Burkinabè?

1. **Richesse culturelle** : Le Burkina Faso possède une scène culturelle dynamique (musique, cinéma, festivals)
2. **Documentation accessible** : LeFaso.net offre une excellente couverture de l'actualité culturelle
3. **Pertinence locale** : Un système qui valorise et rend accessible le patrimoine culturel national
4. **Impact social** : Promouvoir la culture contribue à la cohésion sociale et à l'identité nationale

### Cas d'usage

- **Étudiants** : Recherche d'informations sur la culture burkinabè
- **Journalistes** : Documentation rapide sur événements culturels
- **Touristes** : Découverte de la scène culturelle locale
- **Chercheurs** : Analyse de l'évolution culturelle du pays

---

## 🏗️ Architecture technique

### Pipeline RAG

```
Question utilisateur
    ↓
[1. EMBEDDINGS]
Conversion de la question en vecteur numérique
    ↓
[2. RECHERCHE VECTORIELLE]
Recherche des 5 documents les plus similaires
    ↓
[3. CONTEXTE]
Construction du contexte avec les documents
    ↓
[4. GÉNÉRATION LLM]
Génération de la réponse basée sur le contexte
    ↓
Réponse + Sources citées
```

### Composants détaillés

#### 1️⃣ **Modèle d'Embeddings**
- **Modèle** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dimension** : 384 dimensions
- **Avantages** :
  - Support multilingue (français + 50 langues)
  - Léger (120 MB)
  - Excellent pour la similarité sémantique
- **Licence** : Apache 2.0
- **Lien** : [HuggingFace](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

#### 2️⃣ **Base de Données Vectorielle**
- **Solution** : ChromaDB
- **Type** : Base vectorielle locale avec persistance
- **Avantages** :
  - Installation simple (pure Python)
  - Recherche par similarité rapide
  - Pas de dépendance cloud
  - Métadonnées riches
- **Licence** : Apache 2.0
- **Lien** : [ChromaDB](https://www.trychroma.com/)

#### 3️⃣ **Grand Modèle de Langage (LLM)**

**Option 1 (Recommandée) : Mistral-7B via Ollama**
- **Modèle** : Mistral-7B-Instruct
- **Déploiement** : Local via Ollama
- **Avantages** :
  - Excellent en français
  - Totalement gratuit
  - Contrôle complet
  - Pas de limite d'API
- **Licence** : Apache 2.0
- **Installation** : `curl -fsSL https://ollama.ai/install.sh | sh && ollama pull mistral`

**Option 2 (Backup) : HuggingFace Inference API - Ce qu'on a utilisé** 
- **Modèle** : Mistral-7B-Instruct-v0.2
- **Déploiement** : API gratuite HuggingFace
- **Limite** : 1000 requêtes/jour
- **Avantages** :
  - Pas d'installation locale
  - Gratuit
- **Configuration** : Token HF requis

---

## 🛠️ Technologies open source

| Composant | Technologie | Licence | Usage |
|-----------|-------------|---------|-------|
| **Embeddings** | sentence-transformers | Apache 2.0 | Vectorisation du texte |
| **Vector DB** | ChromaDB | Apache 2.0 | Stockage et recherche vectorielle |
| **LLM** | Mistral-7B (Ollama) | Apache 2.0 | Génération de texte |
| **Backend API** | FastAPI | MIT | API REST |
| **Frontend** | Streamlit | Apache 2.0 | Interface web |
| **Web Scraping** | BeautifulSoup4 | MIT | Collecte de données |
| **Data Processing** | NumPy, Pandas | BSD | Traitement données |

### Pourquoi ces choix?

✅ **Licences libres** : MIT et Apache 2.0 autorisent usage commercial et modification  
✅ **Communauté active** : Support et documentation excellents  
✅ **Performance** : Technologies éprouvées et optimisées  
✅ **Pérennité** : Pas de dépendance à des services payants  

---

## 📥 Installation

### Prérequis

- Python 3.8 ou supérieur
- 8 GB RAM minimum
- 10 GB espace disque

### Étape 1 : Cloner le repository

```bash
git clone https://github.com/luckhub01/hackaton_rag.git
cd culture-burkina-rag
```

### Étape 2 : Créer un environnement virtuel

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 : Configuration (optionnel)

Si vous utilisez HuggingFace API :

```bash
cp .env.example .env
# Éditer .env et ajouter votre token HuggingFace
```

Contenu de `.env` :
```
HUGGINGFACE_TOKEN=votre_token_ici
```

### Étape 5 : Installer Ollama (optionnel, recommandé)

Pour utiliser le LLM local :

```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger Mistral
ollama pull mistral
```

### Étape 6 : Préparer les données

```bash
# Placer vos données scrappées dans data/raw/
mkdir -p data/raw data/processed data/vectors evaluation

# Copier culture_articles.json dans data/raw/

# Lancer le preprocessing
python src/data_preprocessing.py
```

### Étape 7 : Indexer le corpus

```bash
python -c "from rag_pipeline import CultureRAGPipeline; rag = CultureRAGPipeline('data/processed/corpus_cleaned.json'); rag.index_corpus()"
```

---

## 🚀 Utilisation

### Option 1 : Interface Streamlit (Recommandé)

```bash
streamlit run frontend/app.py
```

Ouvrir http://localhost:8501 dans votre navigateur

### Option 2 : API FastAPI

```bash
# Lancer le serveur
python src/api.py

# API disponible sur http://localhost:8000
# Documentation : http://localhost:8000/docs
```

Exemple de requête :

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les principaux festivals culturels?",
    "top_k": 5,
    "use_local_llm": false
  }'
```

### Option 3 : Python direct

```python
from rag_pipeline import CultureRAGPipeline

# Initialiser
rag = CultureRAGPipeline("data/processed/corpus_cleaned.json")

# Poser une question
result = rag.answer_question(
    "Qui est Alif Naaba?",
    use_local_llm=False
)

print(result['answer'])
```

---

## 📊 Évaluation

### Lancer l'évaluation

```bash
python evaluation/evaluate.py
```

### Métriques calculées

1. **Précision Retrieval** : % de documents pertinents récupérés
2. **Pertinence Réponse** : Score /5 basé sur :
   - Contenu attendu (2 pts)
   - Réponse directe (1 pt)
   - Structure claire (1 pt)
   - Citations sources (1 pt)
3. **Temps de Réponse** : Moyenne en secondes

### Dataset de test

20 questions couvrant :
- Festivals culturels (REMA, FESPACO)
- Artistes burkinabè (Alif Naaba, Floby, etc.)
- Institutions (BBDA)
- Événements culturels
- Rôle de la culture

---

## 🎯 Résultats

### Métriques obtenues

| Métrique | Score | Objectif |
|----------|-------|----------|
| **Précision Retrieval** | 87.3% | > 80% ✅ |
| **Pertinence Réponse** | 4.2/5 | > 4/5 ✅ |
| **Temps Réponse** | 2.8s | < 5s ✅ |

### Distribution des scores

- **Excellent (4-5/5)** : 16 questions (80%)
- **Bon (3-4/5)** : 3 questions (15%)
- **Moyen (2-3/5)** : 1 question (5%)
- **Faible (<2/5)** : 0 question (0%)

### Exemples de questions réussies

✅ "Quels sont les principaux festivals culturels au Burkina Faso?"  
✅ "Qui est Alif Naaba et quel est son rôle?"  
✅ "Qu'est-ce que le BBDA?"  
✅ "Comment la culture contribue-t-elle à la paix?"  

---

## 📁 Structure du projet

```
culture-burkina-rag/
├── data/
│   ├── raw/                          # Données brutes
│   │   ├── culture_articles.json    # Articles scrappés
│   │   └── sources.txt              # Liste des URLs
│   ├── processed/                    # Données nettoyées
│   │   └── corpus_cleaned.json      # Corpus préprocessé
│   └── vectors/                      # Base vectorielle
│       └── chroma_db/               # ChromaDB
├── src/
│   ├── data_preprocessing.py        # Nettoyage données
│   ├── rag_pipeline.py              # Pipeline RAG complet
│   └── api.py                       # API FastAPI
├── frontend/
│   └── app.py                       # Interface Streamlit
├── evaluation/
│   ├── test_questions.json          # 20 questions test
│   ├── evaluate.py                  # Script d'évaluation
│   ├── results.json                 # Résultats JSON
│   └── RAPPORT_EVALUATION.md        # Rapport détaillé
├── requirements.txt                  # Dépendances
├── README.md                        # Ce fichier
├── LICENSE                          # Licence MIT
└── .env.example                     # Configuration exemple
```

---

## 👥 Contributeurs

**Équipe** : Les Potiers du code

- **Membre 1** : PARE Ina
- **Membre 2** : ILBOUDO Kieffer


**Hackathon** : MTDPCE - Sélection Équipes Hackathon SN 2025  
**Date** : Novembre 2025  
**Organisateur** : Ministère de la Transition Digitale, des Postes et des Communications Électroniques  

---

## 📄 Licence



**Toutes les technologies utilisées** sont également sous licences open source (MIT ou Apache 2.0).

---

## 🙏 Remerciements

- **LeFaso.net** pour la richesse de leur contenu sur la culture burkinabè
- **Communauté open source** pour les outils exceptionnels
- **MTDPCE** pour l'organisation du hackathon

---

## 📞 Contact

Pour toute question ou suggestion :

- **Email** : [inaparehub@gmail.com]
- **GitHub** : [hackaton_rag]


---

**🇧🇫 Faso na yikri ! (Burkina Faso, terre d'intégrité)**