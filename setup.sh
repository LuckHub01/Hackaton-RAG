#!/bin/bash

# ============================================
# SCRIPT D'INSTALLATION AUTOMATIQUE
# Culture Burkinabè RAG System
# ============================================

set -e  # Arrêter en cas d'erreur

echo "🇧🇫 =========================================="
echo "   Culture Burkinabè RAG - Installation"
echo "=========================================="

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# ============================================
# ÉTAPE 1: Vérifier Python
# ============================================
echo ""
print_info "Vérification de Python..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION détecté"

# ============================================
# ÉTAPE 2: Créer l'environnement virtuel
# ============================================
echo ""
print_info "Création de l'environnement virtuel..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Environnement virtuel créé"
else
    print_warning "Environnement virtuel existe déjà"
fi

# Activer l'environnement virtuel
source venv/bin/activate
print_success "Environnement virtuel activé"

# ============================================
# ÉTAPE 3: Installer les dépendances
# ============================================
echo ""
print_info "Installation des dépendances..."

pip install --upgrade pip
pip install -r requirements.txt

print_success "Toutes les dépendances installées"

# ============================================
# ÉTAPE 4: Créer la structure des dossiers
# ============================================
echo ""
print_info "Création de la structure des dossiers..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/vectors
mkdir -p evaluation
mkdir -p src
mkdir -p frontend

print_success "Structure des dossiers créée"

# ============================================
# ÉTAPE 5: Configuration
# ============================================
echo ""
print_info "Configuration..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Fichier .env créé (à configurer)"
else
    print_warning "Fichier .env existe déjà"
fi

# ============================================
# ÉTAPE 6: Vérifier les données
# ============================================
echo ""
print_info "Vérification des données..."

if [ -f "data/raw/culture_articles.json" ]; then
    print_success "Données brutes trouvées"
    
    # Lancer le preprocessing
    print_info "Lancement du preprocessing..."
    python src/data_preprocessing.py
    print_success "Preprocessing terminé"
else
    print_warning "Fichier data/raw/culture_articles.json non trouvé"
    print_info "Veuillez placer vos données scrappées dans ce fichier"
fi

# ============================================
# ÉTAPE 7: Indexation du corpus
# ============================================
echo ""
print_info "Indexation du corpus..."

if [ -f "data/processed/corpus_cleaned.json" ]; then
    python -c "
from src.rag_pipeline import CultureRAGPipeline
rag = CultureRAGPipeline('data/processed/corpus_cleaned.json')
if rag.collection is None or rag.collection.count() == 0:
    print('Indexation en cours...')
    rag.index_corpus()
    print('Indexation terminée!')
else:
    print('Corpus déjà indexé!')
"
    print_success "Indexation terminée"
else
    print_warning "Corpus nettoyé non trouvé. Lancez d'abord le preprocessing."
fi

# ============================================
# ÉTAPE 8: Installation Ollama (optionnel)
# ============================================
echo ""
read -p "Voulez-vous installer Ollama pour le LLM local? (recommandé) [y/N]: " install_ollama

if [[ $install_ollama =~ ^[Yy]$ ]]; then
    print_info "Installation d'Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_warning "Ollama est déjà installé"
    else
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installé"
    fi
    
    print_info "Téléchargement du modèle Mistral..."
    ollama pull mistral
    print_success "Modèle Mistral téléchargé"
else
    print_info "Installation d'Ollama ignorée"
    print_warning "Vous devrez utiliser HuggingFace API (configurer HUGGINGFACE_TOKEN dans .env)"
fi

# ============================================
# ÉTAPE 9: Création des questions test
# ============================================
echo ""
print_info "Création des questions test..."

python -c "
from evaluation.evaluate import create_test_questions
import json

questions = create_test_questions()
with open('evaluation/test_questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f'{len(questions)} questions test créées')
"

print_success "Questions test créées"

# ============================================
# RÉSUMÉ DE L'INSTALLATION
# ============================================
echo ""
echo "=========================================="
echo "📊 RÉSUMÉ DE L'INSTALLATION"
echo "=========================================="
print_success "Environnement virtuel: venv/"
print_success "Dépendances installées"

if [ -f "data/processed/corpus_cleaned.json" ]; then
    print_success "Corpus nettoyé et indexé"
else
    print_warning "Corpus non préparé - placez vos données dans data/raw/"
fi

if command -v ollama &> /dev/null; then
    print_success "Ollama installé et prêt"
else
    print_warning "Ollama non installé - configurez HuggingFace dans .env"
fi

echo ""
echo "=========================================="
echo "🚀 PROCHAINES ÉTAPES"
echo "=========================================="
echo ""
echo "1. Activer l'environnement virtuel:"
echo "   source venv/bin/activate"
echo ""
echo "2. Si nécessaire, configurer .env:"
echo "   nano .env"
echo ""
echo "3. Lancer l'interface Streamlit:"
echo "   streamlit run frontend/app.py"
echo ""
echo "4. OU lancer l'API:"
echo "   python src/api.py"
echo ""
echo "5. Pour évaluer le système:"
echo "   python evaluation/evaluate.py"
echo ""
echo "=========================================="
print_success "Installation terminée!"
echo "=========================================="