#!/usr/bin/env python3
"""
Robust Semantic Severity Recalibration

Uses sentence embeddings for semantic similarity matching instead of
fixed keyword lists. This approach:
1. Generalizes to unseen terminology
2. Captures meaning rather than exact words
3. Handles paraphrasing and synonyms
4. Is extensible without manual marker curation

Dependencies: sentence-transformers, numpy, pandas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SemanticRecalibrationConfig:
    """Configuration for semantic recalibration"""
    # Model for semantic embeddings
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, good quality
    
    # Similarity thresholds
    contraindicated_threshold: float = 0.65
    major_threshold: float = 0.55
    moderate_threshold: float = 0.45
    
    # Component weights
    semantic_weight: float = 0.45
    confidence_weight: float = 0.25
    drug_class_weight: float = 0.30
    
    # Confidence thresholds for zero-shot adjustment
    contraindicated_min_confidence: float = 0.65
    major_min_confidence: float = 0.50


# ============================================================================
# SEMANTIC PROTOTYPES - Representative descriptions for each severity
# ============================================================================

SEVERITY_PROTOTYPES = {
    'contraindicated': [
        # Life-threatening cardiac
        "This combination causes fatal cardiac arrhythmias including torsades de pointes",
        "Co-administration leads to QT prolongation and sudden cardiac death",
        "The combination results in potentially lethal ventricular arrhythmias",
        
        # Serotonin syndrome
        "Combined use causes life-threatening serotonin syndrome",
        "This interaction produces potentially fatal serotonergic toxicity",
        "The drugs together cause dangerous hyperserotonergic state",
        
        # Other fatal
        "This combination is absolutely contraindicated due to fatality risk",
        "Never use together - deaths have been reported",
        "Co-administration causes potentially fatal reactions",
        "The combination leads to neuroleptic malignant syndrome",
        "This interaction causes severe malignant hyperthermia",
    ],
    
    'major': [
        # Bleeding
        "This combination significantly increases the risk of serious bleeding",
        "Co-administration causes major hemorrhagic complications",
        "The interaction substantially enhances anticoagulant effect leading to bleeding",
        "Combined use results in gastrointestinal bleeding requiring hospitalization",
        
        # Metabolic serious
        "This combination causes dangerous hyperkalemia",
        "The interaction leads to severe hypoglycemia",
        "Co-administration results in rhabdomyolysis and muscle breakdown",
        
        # Organ toxicity
        "The combination causes acute renal failure",
        "This interaction produces severe hepatotoxicity",
        "Co-administration leads to agranulocytosis and bone marrow suppression",
        
        # Other serious
        "The interaction requires immediate medical attention",
        "Combined use may require hospitalization",
        "This combination causes serious adverse events requiring intervention",
    ],
    
    'moderate': [
        # Pharmacokinetic
        "This combination may increase serum concentration of the drug",
        "Co-administration can decrease the metabolism of the affected drug",
        "The interaction may enhance the therapeutic effect requiring monitoring",
        "Combined use may reduce drug clearance",
        
        # Blood pressure
        "This combination may cause hypertension requiring monitoring",
        "Co-administration can lead to hypotension",
        "The interaction may affect blood pressure control",
        
        # Monitoring needed
        "Use together with caution and monitor for adverse effects",
        "Dose adjustment may be needed when using these drugs together",
        "Consider monitoring for increased or decreased drug effects",
        "The interaction has moderate clinical significance",
    ],
    
    'minor': [
        # Minimal effects
        "This interaction is unlikely to be clinically significant",
        "The combination has minimal impact on drug effects",
        "Co-administration may cause minor changes in drug levels",
        "The interaction has theoretical significance but rarely causes problems",
        
        # Low concern
        "This is a minor interaction with negligible clinical effect",
        "The combination may slightly alter therapeutic efficacy",
        "Drug interaction is of low clinical significance",
        "Slight changes in drug levels are not expected to be significant",
    ]
}


# ============================================================================
# DRUG CLASS DEFINITIONS (same as before, for completeness)
# ============================================================================

HIGH_RISK_DRUG_CLASSES = {
    'anticoagulants': [
        'warfarin', 'heparin', 'enoxaparin', 'rivaroxaban', 'apixaban',
        'dabigatran', 'edoxaban', 'fondaparinux', 'argatroban', 'bivalirudin'
    ],
    'antiplatelets': [
        'aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel', 'dipyridamole',
        'ticlopidine', 'cangrelor', 'vorapaxar'
    ],
    'qt_prolonging': [
        'amiodarone', 'sotalol', 'dofetilide', 'dronedarone', 'quinidine',
        'procainamide', 'ibutilide', 'azithromycin', 'erythromycin',
        'haloperidol', 'methadone', 'ondansetron'
    ],
    'maois': [
        'phenelzine', 'tranylcypromine', 'isocarboxazid', 'selegiline',
        'rasagiline', 'safinamide', 'moclobemide'
    ],
    'strong_cyp3a4_inhibitors': [
        'ketoconazole', 'itraconazole', 'clarithromycin', 'ritonavir',
        'nelfinavir', 'indinavir', 'nefazodone', 'cobicistat'
    ],
    'serotonergics': [
        'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram',
        'venlafaxine', 'duloxetine', 'tramadol', 'fentanyl', 'triptans'
    ]
}


class SemanticSeverityRecalibrator:
    """
    Robust semantic-based severity recalibration using sentence embeddings.
    
    Instead of fixed keyword matching, this approach:
    1. Embeds prototype descriptions for each severity level
    2. Embeds the interaction description
    3. Computes cosine similarity to find best matching severity
    4. Combines with confidence and drug class signals
    """
    
    SEVERITY_NUMERIC = {
        'Contraindicated interaction': 4,
        'Major interaction': 3,
        'Moderate interaction': 2,
        'Minor interaction': 1
    }
    
    NUMERIC_SEVERITY = {
        4: 'Contraindicated interaction',
        3: 'Major interaction',
        2: 'Moderate interaction',
        1: 'Minor interaction'
    }
    
    def __init__(self, config: Optional[SemanticRecalibrationConfig] = None):
        self.config = config or SemanticRecalibrationConfig()
        self.model = None
        self.prototype_embeddings = {}
        self.stats = {}
        
    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.config.embedding_model}")
                self.model = SentenceTransformer(self.config.embedding_model)
                self._compute_prototype_embeddings()
            except ImportError:
                print("WARNING: sentence-transformers not available, falling back to keyword matching")
                self.model = "fallback"
    
    def _compute_prototype_embeddings(self):
        """Pre-compute embeddings for all severity prototypes"""
        print("Computing prototype embeddings...")
        for severity, prototypes in SEVERITY_PROTOTYPES.items():
            embeddings = self.model.encode(prototypes, convert_to_numpy=True)
            # Store mean embedding as class centroid
            self.prototype_embeddings[severity] = {
                'centroid': np.mean(embeddings, axis=0),
                'all': embeddings
            }
        print(f"  ✓ Computed embeddings for {len(SEVERITY_PROTOTYPES)} severity classes")
    
    def _compute_semantic_score(self, description: str) -> Dict[str, float]:
        """
        Compute semantic similarity to each severity class.
        
        Returns dict with:
        - best_severity: highest matching severity
        - score: numeric score (1-4)
        - similarities: dict of similarity to each class
        """
        if pd.isna(description) or description.strip() == '':
            return {'score': 2.0, 'best_severity': 'moderate', 'similarities': {}}
        
        self._load_model()
        
        if self.model == "fallback":
            # Fallback to keyword-based scoring
            return self._keyword_fallback(description)
        
        # Embed the description
        desc_embedding = self.model.encode([description], convert_to_numpy=True)[0]
        
        # Compute cosine similarity to each class centroid
        similarities = {}
        for severity, data in self.prototype_embeddings.items():
            centroid = data['centroid']
            # Cosine similarity
            sim = np.dot(desc_embedding, centroid) / (
                np.linalg.norm(desc_embedding) * np.linalg.norm(centroid)
            )
            similarities[severity] = float(sim)
        
        # Find best matching severity based on thresholds
        if similarities['contraindicated'] >= self.config.contraindicated_threshold:
            score = 4.0
            best = 'contraindicated'
        elif similarities['major'] >= self.config.major_threshold:
            score = 3.2
            best = 'major'
        elif similarities['moderate'] >= self.config.moderate_threshold:
            score = 2.0
            best = 'moderate'
        else:
            score = 1.5
            best = 'minor'
        
        return {
            'score': score,
            'best_severity': best,
            'similarities': similarities
        }
    
    def _keyword_fallback(self, description: str) -> Dict[str, float]:
        """Fallback keyword-based scoring when embeddings unavailable"""
        desc_lower = description.lower()
        
        contra_keywords = ['fatal', 'death', 'torsades', 'serotonin syndrome', 
                          'cardiac arrest', 'contraindicated', 'lethal']
        major_keywords = ['bleeding', 'hemorrhage', 'hyperkalemia', 'rhabdomyolysis',
                         'renal failure', 'serious', 'hospitalization']
        moderate_keywords = ['monitor', 'caution', 'may increase', 'may decrease',
                            'concentration', 'metabolism']
        
        if any(kw in desc_lower for kw in contra_keywords):
            return {'score': 4.0, 'best_severity': 'contraindicated', 'similarities': {}}
        elif any(kw in desc_lower for kw in major_keywords):
            return {'score': 3.2, 'best_severity': 'major', 'similarities': {}}
        elif any(kw in desc_lower for kw in moderate_keywords):
            return {'score': 2.0, 'best_severity': 'moderate', 'similarities': {}}
        else:
            return {'score': 1.6, 'best_severity': 'moderate', 'similarities': {}}
    
    def _get_drug_class_score(self, drug1: str, drug2: str) -> float:
        """Compute drug class risk score"""
        d1_lower = drug1.lower() if pd.notna(drug1) else ''
        d2_lower = drug2.lower() if pd.notna(drug2) else ''
        
        d1_classes = set()
        d2_classes = set()
        
        for cls, drugs in HIGH_RISK_DRUG_CLASSES.items():
            if any(d in d1_lower for d in drugs):
                d1_classes.add(cls)
            if any(d in d2_lower for d in drugs):
                d2_classes.add(cls)
        
        overlap = d1_classes & d2_classes
        
        # Check dangerous combinations
        if 'maois' in d1_classes and 'serotonergics' in d2_classes:
            return 4.0
        if 'maois' in d2_classes and 'serotonergics' in d1_classes:
            return 4.0
        if 'maois' in overlap:
            return 4.0
        if 'anticoagulants' in overlap or 'qt_prolonging' in overlap:
            return 3.5
        if d1_classes and d2_classes:
            return 3.0
        if d1_classes or d2_classes:
            return 2.5
        return 2.0
    
    def _adjust_confidence_score(self, original_severity: str, confidence: float) -> float:
        """Adjust zero-shot score based on confidence"""
        original_numeric = self.SEVERITY_NUMERIC.get(original_severity, 2)
        
        if original_numeric == 4 and confidence < self.config.contraindicated_min_confidence:
            return 3.0  # Downgrade low-confidence contraindicated
        elif original_numeric >= 3 and confidence < self.config.major_min_confidence:
            return 2.5  # Partial downgrade
        return float(original_numeric)
    
    def recalibrate_single(self, row: pd.Series) -> Dict:
        """Recalibrate a single interaction"""
        drug1 = row.get('drug_name_1', '')
        drug2 = row.get('drug_name_2', '')
        description = row.get('interaction_description', '')
        original_severity = row.get('severity_label', 'Moderate interaction')
        original_confidence = row.get('severity_confidence', 0.5)
        
        # Component 1: Semantic similarity
        semantic_result = self._compute_semantic_score(description)
        s_semantic = semantic_result['score']
        
        # Component 2: Confidence adjustment
        s_confidence = self._adjust_confidence_score(original_severity, original_confidence)
        
        # Component 3: Drug class risk
        s_drug_class = self._get_drug_class_score(drug1, drug2)
        
        # Weighted combination
        s_final = (
            self.config.semantic_weight * s_semantic +
            self.config.confidence_weight * s_confidence +
            self.config.drug_class_weight * s_drug_class
        )
        
        # Map to severity
        if s_final >= 3.2:
            new_severity = 'Contraindicated interaction'
        elif s_final >= 2.5:
            new_severity = 'Major interaction'
        elif s_final >= 2.0:
            new_severity = 'Moderate interaction'
        else:
            new_severity = 'Minor interaction'
        
        return {
            'severity': new_severity,
            'confidence': min(0.95, original_confidence + 0.1),
            'scores': {
                'semantic': s_semantic,
                'confidence': s_confidence,
                'drug_class': s_drug_class,
                'final': s_final
            },
            'semantic_match': semantic_result['best_severity']
        }
    
    def recalibrate_dataset(self, df: pd.DataFrame, 
                           show_progress: bool = True,
                           batch_size: int = 1000) -> pd.DataFrame:
        """
        Recalibrate entire dataset using semantic similarity.
        """
        print("\n" + "="*70)
        print("SEMANTIC SEVERITY RECALIBRATION")
        print("="*70)
        print(f"Processing {len(df):,} interactions...")
        print(f"\nConfiguration:")
        print(f"   Embedding model: {self.config.embedding_model}")
        print(f"   Semantic weight: {self.config.semantic_weight}")
        print(f"   Confidence weight: {self.config.confidence_weight}")
        print(f"   Drug class weight: {self.config.drug_class_weight}")
        
        # Pre-load model
        self._load_model()
        
        # Process in batches for efficiency
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            if show_progress and (i // batch_size) % 10 == 0:
                print(f"   Progress: {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)")
            
            for _, row in batch.iterrows():
                result = self.recalibrate_single(row)
                results.append(result)
        
        # Add results to dataframe
        df_recal = df.copy()
        df_recal['severity_recalibrated'] = [r['severity'] for r in results]
        df_recal['recal_confidence'] = [r['confidence'] for r in results]
        df_recal['recal_method'] = 'semantic_hybrid'
        
        # Compute statistics
        self._compute_stats(df, df_recal)
        
        return df_recal
    
    def _compute_stats(self, df_original: pd.DataFrame, df_recal: pd.DataFrame):
        """Compute recalibration statistics"""
        print("\n" + "-"*50)
        print("RECALIBRATION RESULTS")
        print("-"*50)
        
        orig_dist = df_original['severity_label'].value_counts(normalize=True)
        recal_dist = df_recal['severity_recalibrated'].value_counts(normalize=True)
        
        target_dist = {
            'Contraindicated interaction': 0.05,
            'Major interaction': 0.25,
            'Moderate interaction': 0.60,
            'Minor interaction': 0.10
        }
        
        print("\n   Distribution Comparison:")
        print(f"   {'Severity':<30} {'Original':>10} {'Recalibrated':>12} {'Target':>10}")
        print("   " + "-"*65)
        
        for sev in ['Contraindicated interaction', 'Major interaction', 
                    'Moderate interaction', 'Minor interaction']:
            orig = orig_dist.get(sev, 0)
            recal = recal_dist.get(sev, 0)
            target = target_dist[sev]
            print(f"   {sev:<30} {orig:>9.1%} {recal:>11.1%} {target:>9.1%}")
        
        changes = (df_original['severity_label'] != df_recal['severity_recalibrated']).sum()
        print(f"\n   Total changes: {changes:,} ({changes/len(df_original)*100:.1f}%)")
        
        self.stats = {
            'original_distribution': orig_dist.to_dict(),
            'recalibrated_distribution': recal_dist.to_dict(),
            'target_distribution': target_dist,
            'total_changes': int(changes),
            'change_rate': float(changes / len(df_original))
        }


def main():
    """Run semantic recalibration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic severity recalibration')
    parser.add_argument('--data', type=str,
                       default='data/ddi_cardio_or_antithrombotic_labeled (1).csv')
    parser.add_argument('--output', type=str,
                       default='data/ddi_semantic_recalibrated.csv')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence transformer model')
    args = parser.parse_args()
    
    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} interactions")
    
    config = SemanticRecalibrationConfig(embedding_model=args.model)
    recalibrator = SemanticSeverityRecalibrator(config)
    
    df_recal = recalibrator.recalibrate_dataset(df)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_recal.to_csv(output_path, index=False)
    print(f"\n📄 Saved to: {output_path}")
    
    # Save stats
    stats_path = output_path.with_suffix('.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'config': {
                'model': config.embedding_model,
                'semantic_weight': config.semantic_weight,
                'confidence_weight': config.confidence_weight,
                'drug_class_weight': config.drug_class_weight
            },
            'stats': recalibrator.stats
        }, f, indent=2)
    print(f"📊 Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
