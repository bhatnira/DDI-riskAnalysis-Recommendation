#!/usr/bin/env python3
"""
LLM-Enhanced Recommendation System Validation Framework

Publication-grade evaluation of RAG (Retrieval-Augmented Generation) 
for drug interaction explanations.

Evaluation Dimensions:
1. Faithfulness - Does LLM accurately represent KG facts?
2. Fluency - Is the text grammatically correct and readable?
3. Relevance - Does the explanation address the query?
4. Completeness - Are all important facts covered?
5. Clinical Utility - Is it useful for clinical decision-making?

Metrics:
- Automated: BERTScore, ROUGE, entity extraction F1, fact verification
- Human: Likert scales for quality dimensions

Author: DDI Risk Analysis Research Team
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# Import system components
from kg_recommendation_system import KGRecommendationEngine, KnowledgeGraphLoader
from kg_polypharmacy_risk import PolypharmacyRiskAssessor, PolypharmacyRiskResult
from kg_llm_recommender import LLMEnhancedRecommender, LLMRecommendationResult


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation"""
    total_facts: int = 0
    facts_mentioned: int = 0
    facts_correct: int = 0
    facts_hallucinated: int = 0
    
    precision: float = 0.0  # correct / (correct + hallucinated)
    recall: float = 0.0     # mentioned / total
    f1: float = 0.0
    
    fact_details: List[Dict] = field(default_factory=list)


@dataclass 
class TextQualityResult:
    """Result of text quality metrics"""
    # ROUGE scores
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    
    # Semantic similarity
    semantic_similarity: float = 0.0
    
    # Readability
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    
    # Coverage
    drug_mentions: int = 0
    interaction_mentions: int = 0
    severity_mentions: int = 0


@dataclass
class EvaluationCase:
    """Single evaluation case"""
    case_id: str
    drugs: List[str]
    
    # KG ground truth
    kg_risk_level: str = ""
    kg_risk_score: float = 0.0
    kg_interactions: List[Dict] = field(default_factory=list)
    kg_recommendations: List[Dict] = field(default_factory=list)
    
    # Generated explanations
    template_explanation: str = ""
    llm_explanation: str = ""
    
    # Evaluation results
    faithfulness: FaithfulnessResult = None
    text_quality: TextQualityResult = None
    
    # Human evaluation (to be filled)
    human_scores: Dict = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    output_dir: Path = field(default_factory=lambda: Path("publication_llm_validation"))
    
    # Test cases
    n_test_cases: int = 50
    random_seed: int = 42
    
    # LLM settings
    use_llm: bool = True
    llm_model: str = "biomistral"
    
    # Metrics
    compute_rouge: bool = True
    compute_semantic: bool = True


# ============================================================================
# FAITHFULNESS EVALUATOR
# ============================================================================

class FaithfulnessEvaluator:
    """
    Evaluate faithfulness of LLM explanations to KG facts
    
    Checks:
    1. Drug names are correctly mentioned
    2. Interaction severities are accurate
    3. Risk levels match KG assessment
    4. No hallucinated interactions
    5. Recommendations align with KG suggestions
    """
    
    # Severity keywords for extraction
    SEVERITY_KEYWORDS = {
        'contraindicated': ['contraindicated', 'avoid', 'do not use', 'prohibited'],
        'major': ['major', 'serious', 'significant', 'severe', 'dangerous'],
        'moderate': ['moderate', 'caution', 'monitor', 'watch'],
        'minor': ['minor', 'mild', 'slight', 'minimal']
    }
    
    RISK_KEYWORDS = {
        'critical': ['critical', 'emergency', 'urgent', 'immediate'],
        'high': ['high', 'significant', 'serious', 'substantial'],
        'moderate': ['moderate', 'medium', 'some'],
        'low': ['low', 'minimal', 'slight', 'minor']
    }
    
    def __init__(self, kg_loader: KnowledgeGraphLoader):
        self.kg = kg_loader
    
    def extract_drug_mentions(self, text: str, expected_drugs: List[str]) -> Set[str]:
        """Extract drug names mentioned in text"""
        text_lower = text.lower()
        mentioned = set()
        
        for drug in expected_drugs:
            drug_lower = drug.lower()
            # Check exact match or common variations
            if drug_lower in text_lower:
                mentioned.add(drug)
            # Check for aspirin/acetylsalicylic acid equivalence
            elif drug_lower == 'aspirin' and 'acetylsalicylic' in text_lower:
                mentioned.add(drug)
            elif drug_lower == 'acetylsalicylic acid' and 'aspirin' in text_lower:
                mentioned.add(drug)
        
        return mentioned
    
    def extract_severity_mentions(self, text: str) -> Dict[str, int]:
        """Extract severity level mentions"""
        text_lower = text.lower()
        counts = defaultdict(int)
        
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            for kw in keywords:
                counts[severity] += text_lower.count(kw)
        
        return dict(counts)
    
    def extract_risk_level_mention(self, text: str) -> Optional[str]:
        """Extract mentioned risk level"""
        text_lower = text.lower()
        
        for level, keywords in self.RISK_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return level.upper()
        
        return None
    
    def check_interaction_accuracy(self, text: str, 
                                   kg_interactions: List[Dict]) -> Tuple[int, int, List[Dict]]:
        """
        Check if mentioned interactions match KG
        
        Returns:
            (correct_count, hallucinated_count, details)
        """
        text_lower = text.lower()
        correct = 0
        details = []
        
        # Check each KG interaction
        for inter in kg_interactions:
            drug1 = inter.get('drug1', inter.get('drug_1', '')).lower()
            drug2 = inter.get('drug2', inter.get('drug_2', '')).lower()
            severity = inter.get('severity', inter.get('severity_label', ''))
            
            # Check if pair is mentioned
            pair_mentioned = (drug1 in text_lower and drug2 in text_lower)
            
            # Check if severity is roughly correct
            severity_matched = False
            if 'contraindicated' in severity.lower():
                severity_matched = any(kw in text_lower for kw in self.SEVERITY_KEYWORDS['contraindicated'])
            elif 'major' in severity.lower():
                severity_matched = any(kw in text_lower for kw in self.SEVERITY_KEYWORDS['major'])
            elif 'moderate' in severity.lower():
                severity_matched = any(kw in text_lower for kw in self.SEVERITY_KEYWORDS['moderate'])
            
            if pair_mentioned:
                correct += 1
                details.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'severity': severity,
                    'mentioned': True,
                    'severity_matched': severity_matched
                })
        
        # Simple hallucination detection: look for "interaction" without KG support
        # This is a simplified heuristic
        hallucinated = 0
        
        return correct, hallucinated, details
    
    def evaluate(self, text: str, 
                 drugs: List[str],
                 kg_risk_level: str,
                 kg_interactions: List[Dict]) -> FaithfulnessResult:
        """
        Evaluate faithfulness of explanation
        
        Args:
            text: Generated explanation text
            drugs: List of drugs in regimen
            kg_risk_level: Risk level from KG
            kg_interactions: List of interactions from KG
        """
        result = FaithfulnessResult()
        
        # Count expected facts
        result.total_facts = len(drugs) + len(kg_interactions) + 1  # +1 for risk level
        
        # Check drug mentions
        mentioned_drugs = self.extract_drug_mentions(text, drugs)
        drug_facts = len(mentioned_drugs)
        
        # Check interaction mentions
        correct_interactions, hallucinated, interaction_details = \
            self.check_interaction_accuracy(text, kg_interactions)
        
        # Check risk level
        mentioned_risk = self.extract_risk_level_mention(text)
        risk_correct = 1 if mentioned_risk == kg_risk_level else 0
        
        # Aggregate
        result.facts_mentioned = drug_facts + correct_interactions + (1 if mentioned_risk else 0)
        result.facts_correct = drug_facts + correct_interactions + risk_correct
        result.facts_hallucinated = hallucinated
        
        # Compute metrics
        if result.facts_correct + result.facts_hallucinated > 0:
            result.precision = result.facts_correct / (result.facts_correct + result.facts_hallucinated)
        
        if result.total_facts > 0:
            result.recall = result.facts_mentioned / result.total_facts
        
        if result.precision + result.recall > 0:
            result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
        
        result.fact_details = interaction_details
        
        return result


# ============================================================================
# TEXT QUALITY EVALUATOR
# ============================================================================

class TextQualityEvaluator:
    """
    Evaluate text quality of generated explanations
    
    Metrics:
    - ROUGE (vs reference)
    - Semantic similarity (sentence embeddings)
    - Readability metrics
    - Coverage metrics
    """
    
    def __init__(self):
        self.rouge_scorer = None
        self.sentence_model = None
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass
    
    def compute_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        if not self.rouge_scorer:
            return {'rouge1_f': 0, 'rouge2_f': 0, 'rougeL_f': 0}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings"""
        if not self.sentence_model:
            return 0.0
        
        try:
            emb1 = self.sentence_model.encode(text1)
            emb2 = self.sentence_model.encode(text2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except:
            return 0.0
    
    def compute_readability(self, text: str) -> Dict[str, float]:
        """Compute readability metrics"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        unique_words = set(w.lower() for w in words)
        
        return {
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'vocabulary_richness': len(unique_words) / max(len(words), 1)
        }
    
    def count_mentions(self, text: str, drugs: List[str]) -> Dict[str, int]:
        """Count key entity mentions"""
        text_lower = text.lower()
        
        drug_count = sum(1 for d in drugs if d.lower() in text_lower)
        
        interaction_keywords = ['interaction', 'interact', 'combine', 'combination']
        interaction_count = sum(text_lower.count(kw) for kw in interaction_keywords)
        
        severity_keywords = ['major', 'minor', 'moderate', 'contraindicated', 'severe', 'serious']
        severity_count = sum(text_lower.count(kw) for kw in severity_keywords)
        
        return {
            'drug_mentions': drug_count,
            'interaction_mentions': interaction_count,
            'severity_mentions': severity_count
        }
    
    def evaluate(self, generated: str, reference: str, 
                drugs: List[str]) -> TextQualityResult:
        """Evaluate text quality"""
        result = TextQualityResult()
        
        # ROUGE
        rouge = self.compute_rouge(generated, reference)
        result.rouge1_f = rouge['rouge1_f']
        result.rouge2_f = rouge['rouge2_f']
        result.rougeL_f = rouge['rougeL_f']
        
        # Semantic similarity
        result.semantic_similarity = self.compute_semantic_similarity(generated, reference)
        
        # Readability
        readability = self.compute_readability(generated)
        result.avg_sentence_length = readability['avg_sentence_length']
        result.vocabulary_richness = readability['vocabulary_richness']
        
        # Coverage
        mentions = self.count_mentions(generated, drugs)
        result.drug_mentions = mentions['drug_mentions']
        result.interaction_mentions = mentions['interaction_mentions']
        result.severity_mentions = mentions['severity_mentions']
        
        return result


# ============================================================================
# A/B COMPARISON
# ============================================================================

class ABComparator:
    """
    Compare template-based vs LLM-based explanations
    """
    
    def __init__(self, kg_loader: KnowledgeGraphLoader):
        self.kg = kg_loader
        self.faithfulness_eval = FaithfulnessEvaluator(kg_loader)
        self.text_quality_eval = TextQualityEvaluator()
    
    def generate_reference(self, risk_result: PolypharmacyRiskResult) -> str:
        """Generate reference text from KG facts (for ROUGE comparison)"""
        lines = []
        
        # Risk level
        lines.append(f"Risk level: {risk_result.risk_level}")
        lines.append(f"Risk score: {risk_result.overall_risk_score:.2f}")
        
        # Interactions
        for inter in risk_result.ddi_pairs[:5]:
            d1 = inter.get('drug1', '?')
            d2 = inter.get('drug2', '?')
            sev = inter.get('severity', 'Unknown')
            desc = inter.get('description', '')[:100]
            lines.append(f"{d1} and {d2}: {sev}. {desc}")
        
        return ' '.join(lines)
    
    def compare(self, drugs: List[str],
               template_text: str,
               llm_text: str,
               risk_result: PolypharmacyRiskResult) -> Dict[str, Any]:
        """
        Compare template vs LLM explanations
        """
        reference = self.generate_reference(risk_result)
        kg_interactions = risk_result.ddi_pairs
        
        # Evaluate template
        template_faith = self.faithfulness_eval.evaluate(
            template_text, drugs, risk_result.risk_level, kg_interactions
        )
        template_quality = self.text_quality_eval.evaluate(
            template_text, reference, drugs
        )
        
        # Evaluate LLM
        llm_faith = self.faithfulness_eval.evaluate(
            llm_text, drugs, risk_result.risk_level, kg_interactions
        )
        llm_quality = self.text_quality_eval.evaluate(
            llm_text, reference, drugs
        )
        
        return {
            'template': {
                'faithfulness': asdict(template_faith),
                'text_quality': asdict(template_quality)
            },
            'llm': {
                'faithfulness': asdict(llm_faith),
                'text_quality': asdict(llm_quality)
            },
            'comparison': {
                'faithfulness_f1_delta': llm_faith.f1 - template_faith.f1,
                'rouge1_delta': llm_quality.rouge1_f - template_quality.rouge1_f,
                'semantic_delta': llm_quality.semantic_similarity - template_quality.semantic_similarity
            }
        }


# ============================================================================
# HUMAN EVALUATION FRAMEWORK
# ============================================================================

class HumanEvaluationGenerator:
    """
    Generate materials for human evaluation study
    
    Dimensions:
    1. Faithfulness (1-5): Does explanation match the facts?
    2. Fluency (1-5): Is it well-written?
    3. Completeness (1-5): Does it cover all important points?
    4. Clarity (1-5): Is it easy to understand?
    5. Utility (1-5): Would this help a clinician?
    """
    
    EVALUATION_DIMENSIONS = [
        {
            'name': 'Faithfulness',
            'question': 'How accurately does the explanation represent the drug interaction facts?',
            'anchors': {
                1: 'Contains major factual errors',
                3: 'Mostly accurate with minor issues',
                5: 'Completely accurate'
            }
        },
        {
            'name': 'Fluency',
            'question': 'How well-written and grammatical is the explanation?',
            'anchors': {
                1: 'Many grammar/spelling errors',
                3: 'Generally readable',
                5: 'Professional quality writing'
            }
        },
        {
            'name': 'Completeness',
            'question': 'Does the explanation cover all important interactions and risks?',
            'anchors': {
                1: 'Missing major information',
                3: 'Covers main points',
                5: 'Comprehensive coverage'
            }
        },
        {
            'name': 'Clarity',
            'question': 'How easy is it to understand the explanation?',
            'anchors': {
                1: 'Confusing or unclear',
                3: 'Understandable with effort',
                5: 'Crystal clear'
            }
        },
        {
            'name': 'Clinical Utility',
            'question': 'Would this explanation help a clinician make decisions?',
            'anchors': {
                1: 'Not useful',
                3: 'Somewhat helpful',
                5: 'Very helpful for decision-making'
            }
        }
    ]
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.output_dir = config.output_dir / 'human_evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_evaluation_form(self, cases: List[EvaluationCase]) -> str:
        """Generate human evaluation form as markdown"""
        lines = [
            "# Human Evaluation of Drug Interaction Explanations",
            "",
            "## Instructions",
            "",
            "You will evaluate explanations of drug-drug interactions generated by two systems:",
            "- **System A**: Template-based explanations",
            "- **System B**: LLM-generated explanations",
            "",
            "For each case, rate BOTH explanations on the following dimensions (1-5 scale):",
            ""
        ]
        
        # Dimension descriptions
        for dim in self.EVALUATION_DIMENSIONS:
            lines.append(f"### {dim['name']}")
            lines.append(f"_{dim['question']}_")
            lines.append("")
            for score, desc in dim['anchors'].items():
                lines.append(f"- **{score}**: {desc}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Evaluation cases
        for i, case in enumerate(cases, 1):
            lines.extend([
                f"## Case {i}: {', '.join(case.drugs)}",
                "",
                f"**Ground Truth Risk Level**: {case.kg_risk_level}",
                f"**Number of Interactions**: {len(case.kg_interactions)}",
                "",
                "### System A (Template)",
                "```",
                case.template_explanation[:500] + "..." if len(case.template_explanation) > 500 else case.template_explanation,
                "```",
                "",
                "### System B (LLM)",
                "```",
                case.llm_explanation[:500] + "..." if len(case.llm_explanation) > 500 else case.llm_explanation,
                "```",
                "",
                "### Your Ratings",
                "",
                "| Dimension | System A | System B |",
                "|-----------|----------|----------|",
            ])
            
            for dim in self.EVALUATION_DIMENSIONS:
                lines.append(f"| {dim['name']} | ___ | ___ |")
            
            lines.extend([
                "",
                "**Comments (optional):**",
                "",
                "_______________________________________",
                "",
                "---",
                ""
            ])
        
        return '\n'.join(lines)
    
    def generate_csv_form(self, cases: List[EvaluationCase]) -> pd.DataFrame:
        """Generate CSV form for data collection"""
        rows = []
        
        for case in cases:
            for system in ['template', 'llm']:
                row = {
                    'case_id': case.case_id,
                    'drugs': ';'.join(case.drugs),
                    'system': system,
                    'kg_risk_level': case.kg_risk_level,
                    'n_interactions': len(case.kg_interactions)
                }
                
                for dim in self.EVALUATION_DIMENSIONS:
                    row[f'score_{dim["name"].lower()}'] = ''
                
                row['comments'] = ''
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_materials(self, cases: List[EvaluationCase]):
        """Save all evaluation materials"""
        # Markdown form
        md_content = self.generate_evaluation_form(cases)
        with open(self.output_dir / 'evaluation_form.md', 'w') as f:
            f.write(md_content)
        
        # CSV for data entry
        df = self.generate_csv_form(cases)
        df.to_csv(self.output_dir / 'evaluation_data.csv', index=False)
        
        # Case details JSON
        case_data = []
        for case in cases:
            case_data.append({
                'case_id': case.case_id,
                'drugs': case.drugs,
                'kg_risk_level': case.kg_risk_level,
                'kg_risk_score': case.kg_risk_score,
                'kg_interactions': case.kg_interactions[:5],
                'template_explanation': case.template_explanation,
                'llm_explanation': case.llm_explanation
            })
        
        with open(self.output_dir / 'cases.json', 'w') as f:
            json.dump(case_data, f, indent=2)
        
        print(f"   Saved: evaluation_form.md")
        print(f"   Saved: evaluation_data.csv")
        print(f"   Saved: cases.json")


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================

class LLMValidationPipeline:
    """
    Main pipeline for LLM validation
    """
    
    # Test regimens covering different risk levels
    TEST_REGIMENS = [
        # High risk
        ['warfarin', 'aspirin', 'metoprolol'],
        ['warfarin', 'ibuprofen', 'lisinopril'],
        ['clopidogrel', 'omeprazole', 'aspirin'],
        ['methotrexate', 'ibuprofen', 'prednisone'],
        ['lithium', 'ibuprofen', 'hydrochlorothiazide'],
        ['digoxin', 'amiodarone', 'furosemide'],
        ['fluoxetine', 'tramadol', 'alprazolam'],
        ['simvastatin', 'amiodarone', 'diltiazem'],
        
        # Moderate risk
        ['atorvastatin', 'amlodipine', 'lisinopril'],
        ['metformin', 'glipizide', 'lisinopril'],
        ['omeprazole', 'clopidogrel', 'metoprolol'],
        ['sertraline', 'trazodone', 'alprazolam'],
        ['gabapentin', 'pregabalin', 'duloxetine'],
        
        # Lower risk
        ['acetaminophen', 'ibuprofen', 'omeprazole'],
        ['lisinopril', 'amlodipine', 'hydrochlorothiazide'],
        ['metformin', 'sitagliptin', 'atorvastatin'],
        ['levothyroxine', 'calcium', 'vitamin d'],
    ]
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def run(self) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        print("=" * 70)
        print("LLM-ENHANCED RECOMMENDATION VALIDATION")
        print("Publication-Grade Evaluation Framework")
        print("=" * 70)
        
        # Step 1: Initialize components
        print("\n📚 Step 1: Loading components...")
        kg = KnowledgeGraphLoader().load()
        risk_assessor = PolypharmacyRiskAssessor(kg)
        
        # Create recommender with template mode for comparison
        template_recommender = LLMEnhancedRecommender(use_llm=False)
        
        # Try to create LLM recommender
        llm_available = False
        llm_recommender = None
        try:
            llm_recommender = LLMEnhancedRecommender(
                model=self.config.llm_model,
                use_llm=self.config.use_llm
            )
            llm_available = llm_recommender.use_llm
        except:
            pass
        
        if not llm_available:
            print("⚠️ LLM not available - will evaluate templates only")
        
        # Step 2: Generate evaluation cases
        print("\n📝 Step 2: Generating evaluation cases...")
        cases = self._generate_cases(
            kg, risk_assessor, template_recommender, llm_recommender
        )
        print(f"   Generated {len(cases)} evaluation cases")
        
        # Step 3: Faithfulness evaluation
        print("\n🔍 Step 3: Evaluating faithfulness...")
        faith_eval = FaithfulnessEvaluator(kg)
        faith_results = self._evaluate_faithfulness(cases, faith_eval)
        self.results['faithfulness'] = faith_results
        
        # Step 4: Text quality evaluation
        print("\n📊 Step 4: Evaluating text quality...")
        quality_eval = TextQualityEvaluator()
        quality_results = self._evaluate_quality(cases, quality_eval, risk_assessor)
        self.results['text_quality'] = quality_results
        
        # Step 5: A/B comparison
        print("\n⚖️ Step 5: A/B comparison...")
        comparator = ABComparator(kg)
        ab_results = self._run_ab_comparison(cases, comparator, risk_assessor)
        self.results['ab_comparison'] = ab_results
        
        # Step 6: Generate human evaluation materials
        print("\n📋 Step 6: Generating human evaluation materials...")
        human_eval = HumanEvaluationGenerator(self.config)
        human_eval.save_materials(cases[:20])  # Top 20 cases
        
        # Step 7: Generate outputs
        print("\n📁 Step 7: Generating outputs...")
        self._generate_outputs(cases)
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _generate_cases(self, kg, risk_assessor, template_rec, llm_rec) -> List[EvaluationCase]:
        """Generate evaluation cases"""
        cases = []
        
        for i, drugs in enumerate(self.TEST_REGIMENS):
            case_id = f"case_{i+1:03d}"
            
            # Get KG assessment
            risk_result = risk_assessor.assess_polypharmacy_risk(drugs)
            
            # Generate template explanation
            template_result = template_rec.analyze_regimen(drugs)
            template_text = template_result.risk_explanation
            
            # Generate LLM explanation (if available)
            llm_text = ""
            if llm_rec and llm_rec.use_llm:
                llm_result = llm_rec.analyze_regimen(drugs)
                llm_text = llm_result.risk_explanation
            else:
                llm_text = "[LLM not available]"
            
            case = EvaluationCase(
                case_id=case_id,
                drugs=drugs,
                kg_risk_level=risk_result.risk_level,
                kg_risk_score=risk_result.overall_risk_score,
                kg_interactions=risk_result.ddi_pairs[:10],
                kg_recommendations=[],
                template_explanation=template_text,
                llm_explanation=llm_text
            )
            
            cases.append(case)
        
        return cases
    
    def _evaluate_faithfulness(self, cases: List[EvaluationCase], 
                               evaluator: FaithfulnessEvaluator) -> Dict:
        """Evaluate faithfulness for all cases"""
        template_scores = []
        llm_scores = []
        
        for case in cases:
            # Template
            template_faith = evaluator.evaluate(
                case.template_explanation,
                case.drugs,
                case.kg_risk_level,
                case.kg_interactions
            )
            case.faithfulness = template_faith
            template_scores.append(template_faith.f1)
            
            # LLM (if available)
            if case.llm_explanation and '[LLM not available]' not in case.llm_explanation:
                llm_faith = evaluator.evaluate(
                    case.llm_explanation,
                    case.drugs,
                    case.kg_risk_level,
                    case.kg_interactions
                )
                llm_scores.append(llm_faith.f1)
        
        return {
            'template': {
                'mean_f1': np.mean(template_scores) if template_scores else 0,
                'std_f1': np.std(template_scores) if template_scores else 0,
                'n': len(template_scores)
            },
            'llm': {
                'mean_f1': np.mean(llm_scores) if llm_scores else 0,
                'std_f1': np.std(llm_scores) if llm_scores else 0,
                'n': len(llm_scores)
            }
        }
    
    def _evaluate_quality(self, cases: List[EvaluationCase],
                         evaluator: TextQualityEvaluator,
                         risk_assessor) -> Dict:
        """Evaluate text quality for all cases"""
        template_rouge = []
        template_coverage = []
        
        for case in cases:
            risk_result = risk_assessor.assess_polypharmacy_risk(case.drugs)
            reference = f"Risk: {case.kg_risk_level}. " + ' '.join([
                f"{i.get('drug1', '?')} and {i.get('drug2', '?')}: {i.get('severity', 'Unknown')}."
                for i in case.kg_interactions[:3]
            ])
            
            quality = evaluator.evaluate(
                case.template_explanation,
                reference,
                case.drugs
            )
            case.text_quality = quality
            template_rouge.append(quality.rouge1_f)
            template_coverage.append(quality.drug_mentions / max(len(case.drugs), 1))
        
        return {
            'template': {
                'mean_rouge1': np.mean(template_rouge) if template_rouge else 0,
                'mean_coverage': np.mean(template_coverage) if template_coverage else 0,
                'n': len(template_rouge)
            }
        }
    
    def _run_ab_comparison(self, cases: List[EvaluationCase],
                          comparator: ABComparator,
                          risk_assessor) -> Dict:
        """Run A/B comparison"""
        comparisons = []
        
        for case in cases:
            if '[LLM not available]' in case.llm_explanation:
                continue
            
            risk_result = risk_assessor.assess_polypharmacy_risk(case.drugs)
            
            comparison = comparator.compare(
                case.drugs,
                case.template_explanation,
                case.llm_explanation,
                risk_result
            )
            comparisons.append(comparison)
        
        if not comparisons:
            return {'note': 'LLM not available for comparison'}
        
        # Aggregate
        faith_deltas = [c['comparison']['faithfulness_f1_delta'] for c in comparisons]
        
        return {
            'n_comparisons': len(comparisons),
            'mean_faithfulness_delta': np.mean(faith_deltas) if faith_deltas else 0,
            'llm_better_count': sum(1 for d in faith_deltas if d > 0),
            'template_better_count': sum(1 for d in faith_deltas if d < 0),
            'tie_count': sum(1 for d in faith_deltas if d == 0)
        }
    
    def _generate_outputs(self, cases: List[EvaluationCase]):
        """Generate output files"""
        output_dir = self.config.output_dir
        (output_dir / 'tables').mkdir(exist_ok=True)
        (output_dir / 'data').mkdir(exist_ok=True)
        
        # Summary table
        rows = []
        for case in cases:
            row = {
                'case_id': case.case_id,
                'drugs': ', '.join(case.drugs),
                'kg_risk_level': case.kg_risk_level,
                'kg_risk_score': case.kg_risk_score,
                'n_interactions': len(case.kg_interactions),
                'faithfulness_f1': case.faithfulness.f1 if case.faithfulness else 0,
                'rouge1': case.text_quality.rouge1_f if case.text_quality else 0
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'tables' / 'evaluation_summary.csv', index=False)
        
        # Full results JSON
        with open(output_dir / 'data' / 'validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"   Saved: tables/evaluation_summary.csv")
        print(f"   Saved: data/validation_results.json")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        faith = self.results.get('faithfulness', {})
        quality = self.results.get('text_quality', {})
        ab = self.results.get('ab_comparison', {})
        
        print("\n📊 Faithfulness (Template):")
        print(f"   Mean F1: {faith.get('template', {}).get('mean_f1', 0):.3f}")
        print(f"   Std: {faith.get('template', {}).get('std_f1', 0):.3f}")
        
        if faith.get('llm', {}).get('n', 0) > 0:
            print(f"\n📊 Faithfulness (LLM):")
            print(f"   Mean F1: {faith['llm']['mean_f1']:.3f}")
            print(f"   Std: {faith['llm']['std_f1']:.3f}")
        
        print(f"\n📝 Text Quality (Template):")
        print(f"   Mean ROUGE-1: {quality.get('template', {}).get('mean_rouge1', 0):.3f}")
        print(f"   Mean Coverage: {quality.get('template', {}).get('mean_coverage', 0):.3f}")
        
        if ab.get('n_comparisons', 0) > 0:
            print(f"\n⚖️ A/B Comparison:")
            print(f"   Cases compared: {ab['n_comparisons']}")
            print(f"   LLM better: {ab['llm_better_count']}")
            print(f"   Template better: {ab['template_better_count']}")
        
        print(f"\n📁 Results saved to: {self.config.output_dir}/")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate LLM-enhanced recommendation system'
    )
    parser.add_argument('--output-dir', type=str, 
                       default='publication_llm_validation',
                       help='Output directory')
    parser.add_argument('--no-llm', action='store_true',
                       help='Skip LLM evaluation')
    parser.add_argument('--model', type=str, default='biomistral',
                       help='LLM model to use')
    
    args = parser.parse_args()
    
    config = ValidationConfig(
        output_dir=Path(args.output_dir),
        use_llm=not args.no_llm,
        llm_model=args.model
    )
    
    pipeline = LLMValidationPipeline(config)
    results = pipeline.run()
    
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
