"""
SYSTÈME D'ÉVALUATION - Culture Burkinabè RAG
Évaluation automatique avec 20 questions test
"""

import json
import time
from typing import List, Dict
import numpy as np
from datetime import datetime
import sys

sys.path.append('.')
from src.rag_pipeline import CultureRAGPipeline


class RAGEvaluator:
    """Évaluateur pour le système RAG"""
    
    def __init__(self, rag_pipeline: CultureRAGPipeline, test_file: str):
        """
        Args:
            rag_pipeline: Pipeline RAG à évaluer
            test_file: Fichier JSON avec questions test
        """
        self.rag = rag_pipeline
        self.test_file = test_file
        self.results = []
    
    def calculate_retrieval_precision(self, retrieved_docs: List[Dict], expected_keywords: List[str]) -> float:
        """
        Calcul de la précision du retrieval
        
        Args:
            retrieved_docs: Documents récupérés
            expected_keywords: Mots-clés attendus dans les documents pertinents
        
        Returns:
            Score de précision entre 0 et 1
        """
        if not expected_keywords or not retrieved_docs:
            return 0.0
        
        # Compter combien de documents contiennent au moins un mot-clé
        relevant_docs = 0
        
        for doc in retrieved_docs:
            content = f"{doc['title']} {doc['content']}".lower()
            if any(keyword.lower() in content for keyword in expected_keywords):
                relevant_docs += 1
        
        return relevant_docs / len(retrieved_docs)
    
    def calculate_answer_relevance(self, answer: str, expected_answer: str, question: str) -> float:
        """
        Évaluation de la pertinence de la réponse (score /5)
        
        Critères:
        - Contient les informations attendues (2 pts)
        - Répond directement à la question (1 pt)
        - Structure claire (1 pt)
        - Sources mentionnées (1 pt)
        """
        score = 0.0
        answer_lower = answer.lower()
        expected_lower = expected_answer.lower()
        
        # 1. Contenu attendu (2 pts)
        expected_words = set(expected_lower.split())
        answer_words = set(answer_lower.split())
        
        # Calculer l'intersection
        common_words = expected_words & answer_words
        content_score = min(2.0, (len(common_words) / max(len(expected_words), 1)) * 2)
        score += content_score
        
        # 2. Répond à la question (1 pt)
        question_words = set(question.lower().split())
        question_overlap = len(question_words & answer_words)
        if question_overlap >= 2:
            score += 1.0
        elif question_overlap >= 1:
            score += 0.5
        
        # 3. Structure (1 pt)
        if len(answer) > 50 and not answer.startswith("Je n'ai pas"):
            score += 0.5
        if any(char in answer for char in ['.', ',', ';']):  # Ponctuation
            score += 0.5
        
        # 4. Sources mentionnées (1 pt)
        source_indicators = ['selon', 'article', 'source', 'd\'après', 'lefaso', 'titre']
        if any(indicator in answer_lower for indicator in source_indicators):
            score += 1.0
        
        return min(5.0, score)
    
    def evaluate_single_question(self, test_case: Dict, use_local_llm: bool = False) -> Dict:
        """
        Évaluation d'une seule question
        
        Args:
            test_case: Dictionnaire avec question, expected_answer, keywords
            use_local_llm: Utiliser LLM local ou HuggingFace
        
        Returns:
            Résultats de l'évaluation
        """
        question = test_case['question']
        expected_answer = test_case['expected_answer']
        keywords = test_case.get('keywords', [])
        
        print(f"\n❓ Question: {question}")
        
        # Obtenir la réponse
        start_time = time.time()
        result = self.rag.answer_question(
            query=question,
            use_local_llm=use_local_llm,
            verbose=False
        )
        response_time = time.time() - start_time
        
        # Calculer les métriques
        retrieval_precision = self.calculate_retrieval_precision(
            result['sources'],
            keywords
        )
        
        answer_relevance = self.calculate_answer_relevance(
            result['answer'],
            expected_answer,
            question
        )
        
        evaluation = {
            'question': question,
            'answer': result['answer'],
            'expected_answer': expected_answer,
            'retrieval_precision': retrieval_precision,
            'answer_relevance': answer_relevance,
            'response_time': response_time,
            'num_sources': len(result['sources']),
            'sources': result['sources']
        }
        
        print(f"  ✅ Précision Retrieval: {retrieval_precision*100:.1f}%")
        print(f"  ✅ Pertinence Réponse: {answer_relevance:.1f}/5")
        print(f"  ⏱️ Temps: {response_time:.2f}s")
        
        return evaluation
    
    def run_full_evaluation(self, use_local_llm: bool = False) -> Dict:
        """
        Évaluation complète avec toutes les questions test
        
        Returns:
            Résultats agrégés + détails
        """
        print("="*60)
        print("🔬 ÉVALUATION DU SYSTÈME RAG")
        print("="*60)
        
        # Charger les questions test
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        print(f"\n📋 {len(test_cases)} questions test chargées\n")
        
        # Évaluer chaque question
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            result = self.evaluate_single_question(test_case, use_local_llm)
            results.append(result)
            time.sleep(0.5)  # Pause pour éviter rate limiting
        
        # Calcul des métriques agrégées
        avg_retrieval_precision = np.mean([r['retrieval_precision'] for r in results])
        avg_answer_relevance = np.mean([r['answer_relevance'] for r in results])
        avg_response_time = np.mean([r['response_time'] for r in results])
        
        # Distribution des scores
        relevance_scores = [r['answer_relevance'] for r in results]
        score_distribution = {
            'excellent (4-5)': sum(1 for s in relevance_scores if s >= 4),
            'bon (3-4)': sum(1 for s in relevance_scores if 3 <= s < 4),
            'moyen (2-3)': sum(1 for s in relevance_scores if 2 <= s < 3),
            'faible (<2)': sum(1 for s in relevance_scores if s < 2)
        }
        
        # Résultats finaux
        final_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_questions': len(test_cases),
                'llm_used': 'Local (Ollama)' if use_local_llm else 'HuggingFace API'
            },
            'aggregate_metrics': {
                'avg_retrieval_precision': round(avg_retrieval_precision, 3),
                'avg_retrieval_precision_percent': round(avg_retrieval_precision * 100, 1),
                'avg_answer_relevance': round(avg_answer_relevance, 2),
                'avg_response_time': round(avg_response_time, 2),
                'score_distribution': score_distribution
            },
            'detailed_results': results
        }
        
        # Affichage du rapport
        print("\n" + "="*60)
        print("📊 RÉSULTATS D'ÉVALUATION")
        print("="*60)
        print(f"\n🎯 Précision Retrieval (moyenne): {avg_retrieval_precision*100:.1f}%")
        print(f"💬 Pertinence Réponse (moyenne): {avg_answer_relevance:.2f}/5")
        print(f"⏱️ Temps Réponse (moyenne): {avg_response_time:.2f}s")
        
        print(f"\n📈 Distribution des scores:")
        for category, count in score_distribution.items():
            print(f"  - {category}: {count} questions ({count/len(test_cases)*100:.1f}%)")
        
        print("\n" + "="*60)
        
        return final_results
    
    def save_results(self, results: Dict, output_file: str):
        """Sauvegarde des résultats en JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
    
    def generate_report(self, results: Dict, output_file: str):
        """Génération d'un rapport markdown"""
        report = f"""# Rapport d'Évaluation - Culture Burkinabè RAG

**Date:** {results['metadata']['evaluation_date']}  
**LLM utilisé:** {results['metadata']['llm_used']}  
**Nombre de questions test:** {results['metadata']['total_questions']}

---

## 📊 Métriques Globales

| Métrique | Valeur |
|----------|--------|
| **Précision Retrieval** | {results['aggregate_metrics']['avg_retrieval_precision_percent']}% |
| **Pertinence Réponse** | {results['aggregate_metrics']['avg_answer_relevance']}/5 |
| **Temps de Réponse** | {results['aggregate_metrics']['avg_response_time']}s |

## 📈 Distribution des Scores de Pertinence

"""
        for category, count in results['aggregate_metrics']['score_distribution'].items():
            percent = count / results['metadata']['total_questions'] * 100
            report += f"- **{category}**: {count} questions ({percent:.1f}%)\n"
        
        report += "\n---\n\n## 📝 Résultats Détaillés\n\n"
        
        for i, result in enumerate(results['detailed_results'], 1):
            report += f"""### Question {i}

**Question:** {result['question']}

**Réponse générée:**  
{result['answer']}

**Métriques:**
- Précision Retrieval: {result['retrieval_precision']*100:.1f}%
- Pertinence: {result['answer_relevance']:.1f}/5
- Temps: {result['response_time']:.2f}s
- Sources: {result['num_sources']}

---

"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Rapport markdown généré: {output_file}")


def create_test_questions():
    """Création des 20 questions test avec réponses attendues"""
    
    test_questions = [
        {
            "question": "Quels sont les principaux festivals culturels au Burkina Faso?",
            "expected_answer": "Les principaux festivals incluent les REMA (Rencontres Musicales Africaines), le FESPACO (festival de cinéma), et divers festivals de musique et d'arts.",
            "keywords": ["REMA", "festival", "FESPACO", "musique", "culture"]
        },
        {
            "question": "Qui est Alif Naaba?",
            "expected_answer": "Alif Naaba est un artiste burkinabè, initiateur du projet 'Nos voix pour la paix' qui rassemble plusieurs artistes pour promouvoir la cohésion sociale.",
            "keywords": ["Alif Naaba", "artiste", "paix", "musique"]
        },
        {
            "question": "Qu'est-ce que le BBDA?",
            "expected_answer": "Le BBDA (Bureau Burkinabè du Droit d'Auteur) est une institution qui gère les droits collectifs des artistes au Burkina Faso.",
            "keywords": ["BBDA", "droits d'auteur", "artistes", "gestion"]
        },
        {
            "question": "Que sont les REMA?",
            "expected_answer": "Les REMA (Rencontres Musicales Africaines) sont un festival majeur de musique qui se tient à Ouagadougou et rassemble des artistes africains.",
            "keywords": ["REMA", "musique", "festival", "Ouagadougou", "africain"]
        },
        {
            "question": "Quel est le rôle de la culture dans la cohésion sociale au Burkina Faso?",
            "expected_answer": "La culture, notamment la musique et les festivals, joue un rôle de ciment social et contribue à promouvoir la paix et la cohésion entre les différentes communautés.",
            "keywords": ["culture", "cohésion", "paix", "social", "musique"]
        },
        {
            "question": "Qui a visité le BBDA récemment?",
            "expected_answer": "Le Premier ministre a effectué une visite historique au BBDA, une première pour cette institution.",
            "keywords": ["Premier ministre", "BBDA", "visite", "historique"]
        },
        {
            "question": "Quels artistes ont participé au projet 'Nos voix pour la paix'?",
            "expected_answer": "9 artistes ont participé: Alif Naaba, Floby, Amzy, Kawayoto, Fleur, Flora Paré, ATT et Sissao, Sydyr.",
            "keywords": ["artistes", "Nos voix pour la paix", "Floby", "Alif Naaba"]
        },
        {
            "question": "Quelle est la tournée prévue pour 'Nos voix pour la paix'?",
            "expected_answer": "Une tournée est prévue dans 8 villes: Ouagadougou, Kaya, Tenkodogo, Pô, Gaoua, Koudougou, Ouahigouya et Bobo Dioulasso.",
            "keywords": ["tournée", "villes", "Ouagadougou", "Bobo"]
        },
        {
            "question": "Quel message le ministre de la culture a-t-il transmis aux artistes?",
            "expected_answer": "Le ministre encourage les artistes à rechercher l'excellence, à créer des œuvres originales et de qualité, et à conquérir le monde au-delà des frontières du Burkina Faso.",
            "keywords": ["ministre", "excellence", "artistes", "qualité"]
        },
        {
            "question": "Comment les artistes peuvent-ils contribuer à la sécurité nationale?",
            "expected_answer": "Les artistes peuvent galvaniser les combattants et le peuple par leurs créations et leur engagement dans le combat pour la souveraineté.",
            "keywords": ["artistes", "sécurité", "engagement", "souveraineté"]
        },
        {
            "question": "Quel est le rôle de l'Union européenne dans les projets culturels?",
            "expected_answer": "L'Union européenne soutient des projets culturels comme 'Nos voix pour la paix' pour contribuer au retour de la paix et au renforcement de la cohésion sociale.",
            "keywords": ["Union européenne", "soutien", "paix", "culture"]
        },
        {
            "question": "Quelles langues sont utilisées dans le titre 'Nos voix pour la paix'?",
            "expected_answer": "Plusieurs langues nationales sont chantées dans ce titre afin que le message soit entendu par la majorité de la population.",
            "keywords": ["langues", "nationales", "message", "population"]
        },
        {
            "question": "Quand se tient la 6e édition des REMA?",
            "expected_answer": "La 6e édition des REMA se tient du 19 au 21 octobre 2023.",
            "keywords": ["REMA", "octobre 2023", "édition"]
        },
        {
            "question": "Quelle est l'importance du BBDA pour les artistes?",
            "expected_answer": "Le BBDA permet aux artistes de vivre dignement de leur art en gérant le recouvrement de leurs droits d'auteur.",
            "keywords": ["BBDA", "artistes", "droits", "recouvrement"]
        },
        {
            "question": "Quels thèmes sont abordés durant la tournée 'Nos voix pour la paix'?",
            "expected_answer": "Les thèmes incluent la paix, la solidarité, la démocratie, la protection de l'environnement, la justice et l'égalité.",
            "keywords": ["paix", "solidarité", "démocratie", "justice", "égalité"]
        },
        {
            "question": "Quelle est la vision du ministre pour les artistes burkinabè?",
            "expected_answer": "Le ministre souhaite que les artistes burkinabè franchissent les frontières et que leurs œuvres soient reconnues mondialement.",
            "keywords": ["ministre", "artistes", "frontières", "mondial"]
        },
        {
            "question": "Comment la musique peut-elle changer les comportements?",
            "expected_answer": "Selon Alif Naaba, la musique a la capacité d'apporter un changement de comportement et peut contribuer à apaiser les situations difficiles.",
            "keywords": ["musique", "comportement", "changement", "Alif Naaba"]
        },
        {
            "question": "Qu'est-ce qui accompagne la tournée musicale?",
            "expected_answer": "Un tournoi de Maracana accompagne la tournée, débutant le jour de l'ouverture des REMA.",
            "keywords": ["tournoi", "Maracana", "REMA", "sport"]
        },
        {
            "question": "Quelle est l'importance de l'art pour la paix selon les participants?",
            "expected_answer": "L'art, particulièrement la culture, est considéré comme le ciment qui contribue à apporter la paix, toutes les guerres se terminant sur la table de la négociation.",
            "keywords": ["art", "paix", "culture", "négociation"]
        },
        {
            "question": "Quel est le message porté par 'Nos voix pour la paix'?",
            "expected_answer": "Le message vise à promouvoir la paix, la cohésion sociale et l'espoir au Burkina Faso à travers la musique.",
            "keywords": ["message", "paix", "cohésion", "espoir", "musique"]
        }
    ]
    
    return test_questions


if __name__ == "__main__":
    # Créer les questions test
    print("📝 Création des questions test...")
    test_questions = create_test_questions()
    
    # Sauvegarder
    with open("evaluation/test_questions.json", 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {len(test_questions)} questions test créées")
    
    # Initialiser le RAG
    print("\n🚀 Initialisation du pipeline RAG...")
    rag = CultureRAGPipeline(
        corpus_file="data/processed/corpus_cleaned.json"
    )
    
    # Créer l'évaluateur
    evaluator = RAGEvaluator(rag, "evaluation/test_questions.json")
    
    # Lancer l'évaluation
    results = evaluator.run_full_evaluation(use_local_llm=False)
    
    # Sauvegarder les résultats
    evaluator.save_results(results, "evaluation/results.json")
    evaluator.generate_report(results, "evaluation/RAPPORT_EVALUATION.md")
    
    print("\n✅ Évaluation terminée!")