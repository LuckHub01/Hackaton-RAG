@echo off
REM ============================================
REM SCRIPT D'INSTALLATION WINDOWS
REM Culture Burkinabe RAG System
REM ============================================

echo.
echo ==========================================
echo    Culture Burkinabe RAG - Installation
echo             WINDOWS VERSION
echo ==========================================
echo.

REM Verification Python
echo [ETAPE 1] Verification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    echo Telecharger Python sur: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo [OK] Python detecte
echo.

REM Creation environnement virtuel
echo [ETAPE 2] Creation de l'environnement virtuel...
if not exist "venv" (
    python -m venv venv
    echo [OK] Environnement virtuel cree
) else (
    echo [INFO] Environnement virtuel existe deja
)
echo.

REM Activation environnement virtuel
echo [ETAPE 3] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel active
echo.

REM Installation dependances
echo [ETAPE 4] Installation des dependances...
echo Cela peut prendre quelques minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [OK] Toutes les dependances installees
echo.

REM Creation structure dossiers
echo [ETAPE 5] Creation de la structure des dossiers...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\vectors" mkdir data\vectors
if not exist "evaluation" mkdir evaluation
if not exist "src" mkdir src
if not exist "frontend" mkdir frontend
echo [OK] Structure des dossiers creee
echo.

REM Configuration
echo [ETAPE 6] Configuration...
if not exist ".env" (
    copy .env.example .env >nul
    echo [OK] Fichier .env cree (a configurer)
) else (
    echo [INFO] Fichier .env existe deja
)
echo.

REM Verification donnees
echo [ETAPE 7] Verification des donnees...
if exist "data\raw\culture_articles.json" (
    echo [OK] Donnees brutes trouvees
    echo.
    echo [INFO] Lancement du preprocessing...
    python src\data_preprocessing.py
    echo [OK] Preprocessing termine
) else (
    echo [ATTENTION] Fichier data\raw\culture_articles.json non trouve
    echo Veuillez placer vos donnees scrappees dans ce fichier
)
echo.

REM Indexation
echo [ETAPE 8] Indexation du corpus...
if exist "data\processed\corpus_cleaned.json" (
    python -c "from src.rag_pipeline import CultureRAGPipeline; rag = CultureRAGPipeline('data/processed/corpus_cleaned.json'); rag.index_corpus() if rag.collection is None or rag.collection.count() == 0 else print('Corpus deja indexe')"
    echo [OK] Indexation terminee
) else (
    echo [ATTENTION] Corpus nettoye non trouve
)
echo.

REM Creation questions test
echo [ETAPE 9] Creation des questions test...
python -c "from evaluation.evaluate import create_test_questions; import json; questions = create_test_questions(); open('evaluation/test_questions.json', 'w', encoding='utf-8').write(json.dumps(questions, ensure_ascii=False, indent=2)); print(f'{len(questions)} questions test creees')"
echo [OK] Questions test creees
echo.

REM Resume
echo ==========================================
echo         RESUME DE L'INSTALLATION
echo ==========================================
echo [OK] Environnement virtuel: venv\
echo [OK] Dependances installees

if exist "data\processed\corpus_cleaned.json" (
    echo [OK] Corpus nettoye et indexe
) else (
    echo [ATTENTION] Corpus non prepare
)

echo.
echo ==========================================
echo         PROCHAINES ETAPES
echo ==========================================
echo.
echo 1. Activer l'environnement virtuel:
echo    venv\Scripts\activate
echo.
echo 2. Si necessaire, configurer .env avec notepad:
echo    notepad .env
echo.
echo 3. Lancer l'interface Streamlit:
echo    streamlit run frontend\app.py
echo.
echo 4. OU lancer l'API:
echo    python src\api.py
echo.
echo 5. Pour evaluer le systeme:
echo    python evaluation\evaluate.py
echo.
echo ==========================================
echo     Installation terminee avec succes!
echo ==========================================
echo.
pause