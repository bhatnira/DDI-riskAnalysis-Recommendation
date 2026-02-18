#!/usr/bin/env python3
"""
Knowledge Graph-Based Drug Recommendation System

Publication-grade recommendation engine for high-risk drug interactions:
1. Risk Decomposition - Identify which drug pairs contribute most risk
2. Alternative Drug Finder - Find therapeutically equivalent safer alternatives
3. Multi-Objective Optimization - Balance efficacy, safety, and interaction risk

Validation Framework:
- Known substitution pairs from clinical guidelines
- FAERS outcome comparison
- Ranking metrics (Precision@K, NDCG, MRR)

Author: DDI Risk Analysis Research Team
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, NamedTuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from kg_polypharmacy_risk import (
    KGConfig, KnowledgeGraphLoader, PolypharmacyRiskAssessor,
    DrugProfile, DDIRecord, PolypharmacyRiskResult
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DrugAlternative:
    """A candidate alternative drug"""
    drug_id: str
    drug_name: str
    
    # Similarity metrics
    protein_similarity: float = 0.0  # Jaccard similarity of protein targets
    disease_similarity: float = 0.0  # Jaccard similarity of disease indications
    atc_match_level: int = 0         # 0=none, 3=pharmacological, 4=chemical, 5=exact
    pathway_similarity: float = 0.0  # Shared metabolic pathways
    
    # Safety metrics
    ddi_risk_with_regimen: float = 0.0  # DDI risk with other drugs in regimen
    side_effect_burden: float = 0.0      # Total side effect count
    contraindicated_count: int = 0       # Count of contraindicated DDIs
    
    # Ranking
    therapeutic_score: float = 0.0       # Combined similarity
    safety_score: float = 0.0            # Combined safety
    recommendation_score: float = 0.0    # Final multi-objective score
    
    # Details
    shared_proteins: List[str] = field(default_factory=list)
    shared_diseases: List[str] = field(default_factory=list)
    new_ddis: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskContributor:
    """A drug pair contributing to polypharmacy risk"""
    drug1: str
    drug2: str
    ddi_severity: str
    ddi_score: float
    shared_side_effects: int
    shared_proteins: int
    risk_contribution: float  # Percentage of total risk
    
    # Substitutability
    drug1_alternatives: int = 0
    drug2_alternatives: int = 0
    suggested_replacement: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RecommendationResult:
    """Complete recommendation result for a drug regimen"""
    original_drugs: List[str]
    original_risk_score: float
    original_risk_level: str
    
    # Risk decomposition
    risk_contributors: List[RiskContributor] = field(default_factory=list)
    highest_risk_pair: Tuple[str, str] = ("", "")
    
    # Recommendations
    recommended_substitutions: List[Dict] = field(default_factory=list)
    optimized_regimen: List[str] = field(default_factory=list)
    optimized_risk_score: float = 0.0
    risk_reduction: float = 0.0
    
    # Validation metrics (when gold standard available)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['risk_contributors'] = [rc for rc in self.risk_contributors]
        return d


# ============================================================================
# THERAPEUTIC SIMILARITY ENGINE
# ============================================================================

class TherapeuticSimilarityEngine:
    """
    Find therapeutically similar drugs using KG relationships
    
    Similarity based on:
    1. Protein targets (mechanism of action)
    2. Disease indications (therapeutic use)
    3. ATC classification (pharmacological class)
    4. Metabolic pathways
    """
    
    def __init__(self, kg_loader: KnowledgeGraphLoader):
        self.kg = kg_loader
        self.atc_index: Dict[str, Set[str]] = defaultdict(set)
        self.drug_atc_cache: Dict[str, List[str]] = {}  # Cache drug -> ATC codes
        self._build_atc_index()
    
    def _build_atc_index(self):
        """Build ATC code to drug mapping"""
        drugs_csv = self.kg.config.kg_dir / "drugs.csv"
        if not drugs_csv.exists():
            return
        
        df = pd.read_csv(drugs_csv, dtype=str)  # Read all as strings
        for _, row in df.iterrows():
            drug_id = row['drugbank_id']
            atc_codes = row.get('atc_codes', '')
            
            if pd.notna(atc_codes) and atc_codes:
                atc_list = [a.strip() for a in str(atc_codes).split('|')]
                self.drug_atc_cache[drug_id] = atc_list
                
                for atc in atc_list:
                    if len(atc) >= 1:
                        self.atc_index[atc[:1]].add(drug_id)  # Level 1 (anatomical)
                    if len(atc) >= 3:
                        self.atc_index[atc[:3]].add(drug_id)  # Level 2 (therapeutic)
                    if len(atc) >= 4:
                        self.atc_index[atc[:4]].add(drug_id)  # Level 3 (pharmacological)
                    if len(atc) >= 5:
                        self.atc_index[atc[:5]].add(drug_id)  # Level 4 (chemical)
                    self.atc_index[atc].add(drug_id)           # Level 5 (exact)
        
        print(f"   Built ATC index: {len(self.atc_index)} levels, {len(self.drug_atc_cache)} drugs with ATC")
    
    def get_drug_atc(self, drug_id: str) -> List[str]:
        """Get ATC codes for a drug (cached)"""
        return self.drug_atc_cache.get(drug_id, [])
    
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def compute_protein_similarity(self, drug1_id: str, drug2_id: str) -> Tuple[float, List[str]]:
        """Compute protein target similarity"""
        p1 = self.kg.drug_proteins.get(drug1_id, set())
        p2 = self.kg.drug_proteins.get(drug2_id, set())
        
        similarity = self.jaccard_similarity(p1, p2)
        shared = list(p1 & p2)
        
        return similarity, shared
    
    def compute_disease_similarity(self, drug1_id: str, drug2_id: str) -> Tuple[float, List[str]]:
        """Compute disease indication similarity"""
        d1 = self.kg.drug_diseases.get(drug1_id, set())
        d2 = self.kg.drug_diseases.get(drug2_id, set())
        
        similarity = self.jaccard_similarity(d1, d2)
        shared = list(d1 & d2)
        
        return similarity, shared
    
    def compute_atc_match_level(self, drug1_id: str, drug2_id: str) -> int:
        """Compute ATC classification match level (0-5)"""
        atc1_list = self.get_drug_atc(drug1_id)
        atc2_list = self.get_drug_atc(drug2_id)
        
        if not atc1_list or not atc2_list:
            return 0
        
        max_level = 0
        for atc1 in atc1_list:
            for atc2 in atc2_list:
                if atc1 == atc2:
                    max_level = max(max_level, 5)
                elif len(atc1) >= 5 and len(atc2) >= 5 and atc1[:5] == atc2[:5]:
                    max_level = max(max_level, 5)
                elif len(atc1) >= 4 and len(atc2) >= 4 and atc1[:4] == atc2[:4]:
                    max_level = max(max_level, 4)
                elif len(atc1) >= 3 and len(atc2) >= 3 and atc1[:3] == atc2[:3]:
                    max_level = max(max_level, 3)
        
        return max_level
    
    def find_similar_drugs(self, drug_id: str, 
                          min_protein_sim: float = 0.1,
                          min_disease_sim: float = 0.1,
                          min_atc_level: int = 3,
                          top_k: int = 20) -> List[DrugAlternative]:
        """
        Find therapeutically similar drugs
        
        Args:
            drug_id: Source drug DrugBank ID
            min_protein_sim: Minimum protein target similarity
            min_disease_sim: Minimum disease indication similarity  
            min_atc_level: Minimum ATC match level (3=pharmacological, 4=chemical)
            top_k: Number of alternatives to return
        
        Returns:
            List of DrugAlternative candidates sorted by similarity
        """
        candidates = []
        
        source_proteins = self.kg.drug_proteins.get(drug_id, set())
        source_diseases = self.kg.drug_diseases.get(drug_id, set())
        source_atc = self.get_drug_atc(drug_id)
        
        # Get candidate drugs from ATC
        candidate_ids = set()
        for atc in source_atc:
            if len(atc) >= 3:
                candidate_ids |= self.atc_index.get(atc[:3], set())
            if len(atc) >= 4:
                candidate_ids |= self.atc_index.get(atc[:4], set())
        
        # Also get candidates from shared proteins
        for protein in source_proteins:
            candidate_ids |= self.kg.protein_drugs.get(protein, set())
        
        # And from shared diseases
        for disease in source_diseases:
            candidate_ids |= self.kg.disease_drugs.get(disease, set())
        
        # Remove source drug
        candidate_ids.discard(drug_id)
        
        for cand_id in candidate_ids:
            if cand_id not in self.kg.drugs:
                continue
            
            # Compute similarities
            protein_sim, shared_proteins = self.compute_protein_similarity(drug_id, cand_id)
            disease_sim, shared_diseases = self.compute_disease_similarity(drug_id, cand_id)
            atc_level = self.compute_atc_match_level(drug_id, cand_id)
            
            # Clinical validity filter: require meaningful therapeutic similarity
            # ATC level 5 = exact match (always valid)
            # ATC level 4 = chemical subgroup (valid if disease similarity)
            # ATC level 3 = pharmacological subgroup (valid if strong disease similarity)
            is_clinically_valid = (
                atc_level >= 5 or
                (atc_level >= 4 and disease_sim >= 0.25) or
                (atc_level >= 3 and disease_sim >= 0.35) or
                (disease_sim >= 0.5)  # Same indications even without ATC match
            )
            
            # Original filter for backward compatibility
            passes_threshold = (protein_sim >= min_protein_sim or 
                               disease_sim >= min_disease_sim or 
                               atc_level >= min_atc_level)
            
            if not (is_clinically_valid and passes_threshold):
                continue
            
            # Pathway similarity
            p1 = self.kg.drug_pathways.get(drug_id, set())
            p2 = self.kg.drug_pathways.get(cand_id, set())
            pathway_sim = self.jaccard_similarity(p1, p2)
            
            # Side effect burden
            se_count = len(self.kg.drug_side_effects.get(cand_id, set()))
            
            # Compute therapeutic score - prioritize ATC classification
            # ATC match is the strongest indicator of therapeutic equivalence
            atc_score = atc_level / 5.0
            if atc_level >= 4:  # Chemical subgroup match - bonus
                atc_score = 0.9 + (atc_level - 4) * 0.05
            elif atc_level >= 3:  # Pharmacological subgroup - good
                atc_score = 0.7 + (atc_level - 3) * 0.1
            
            therapeutic_score = (
                0.50 * atc_score +        # ATC most important for clinical substitution
                0.25 * disease_sim +      # Same indications
                0.15 * protein_sim +      # Mechanism similarity
                0.10 * pathway_sim        # Metabolic similarity
            )
            
            cand = DrugAlternative(
                drug_id=cand_id,
                drug_name=self.kg.drugs[cand_id].drug_name,
                protein_similarity=protein_sim,
                disease_similarity=disease_sim,
                atc_match_level=atc_level,
                pathway_similarity=pathway_sim,
                side_effect_burden=se_count / 100.0,  # Normalize
                therapeutic_score=therapeutic_score,
                shared_proteins=[self.kg.protein_names.get(p, p) for p in shared_proteins[:5]],
                shared_diseases=shared_diseases[:5]
            )
            candidates.append(cand)
        
        # Sort by therapeutic score
        candidates.sort(key=lambda x: x.therapeutic_score, reverse=True)
        
        return candidates[:top_k]


# ============================================================================
# KG RECOMMENDATION ENGINE
# ============================================================================

class KGRecommendationEngine:
    """
    Main recommendation engine using Knowledge Graph
    
    Capabilities:
    1. Risk Decomposition - Break down risk by drug pairs
    2. Alternative Finding - Find safer substitutes
    3. Regimen Optimization - Suggest optimal drug combinations
    """
    
    # Weights for multi-objective recommendation score
    WEIGHTS = {
        'therapeutic_similarity': 0.40,
        'safety_improvement': 0.35,
        'ddi_risk_reduction': 0.25
    }
    
    def __init__(self, kg_loader: KnowledgeGraphLoader = None):
        if kg_loader is None:
            kg_loader = KnowledgeGraphLoader().load()
        
        self.kg = kg_loader
        self.risk_assessor = PolypharmacyRiskAssessor(kg_loader)
        self.similarity_engine = TherapeuticSimilarityEngine(kg_loader)
    
    def decompose_risk(self, drugs: List[str]) -> List[RiskContributor]:
        """
        Decompose risk by identifying contribution of each drug pair
        
        Returns list of RiskContributor sorted by risk contribution
        """
        # Resolve drug names to IDs
        drug_ids = []
        drug_names = {}
        for drug in drugs:
            drug_id = self._resolve_drug(drug)
            if drug_id:
                drug_ids.append(drug_id)
                drug_names[drug_id] = drug
        
        if len(drug_ids) < 2:
            return []
        
        # Compute total risk
        total_risk = self.risk_assessor.assess_polypharmacy_risk(drugs).overall_risk_score
        
        contributors = []
        
        for d1, d2 in combinations(drug_ids, 2):
            # Check for DDI
            ddi = self.kg.ddi_index.get((d1, d2))
            
            if ddi:
                ddi_score = ddi.severity_score
                ddi_severity = ddi.severity
            else:
                ddi_score = 0.0
                ddi_severity = "No direct interaction"
            
            # Compute shared side effects
            se1 = self.kg.drug_side_effects.get(d1, set())
            se2 = self.kg.drug_side_effects.get(d2, set())
            shared_se = len(se1 & se2)
            
            # Compute shared proteins
            p1 = self.kg.drug_proteins.get(d1, set())
            p2 = self.kg.drug_proteins.get(d2, set())
            shared_p = len(p1 & p2)
            
            # Compute contribution (normalized)
            contribution = (ddi_score / 10.0 * 0.6 + 
                          min(shared_se, 50) / 50.0 * 0.25 +
                          min(shared_p, 10) / 10.0 * 0.15)
            
            # Find number of alternatives for each
            d1_alts = len(self.similarity_engine.find_similar_drugs(d1, top_k=10))
            d2_alts = len(self.similarity_engine.find_similar_drugs(d2, top_k=10))
            
            # Suggest which to replace (the one with more alternatives)
            if d1_alts >= d2_alts:
                suggested = drug_names.get(d1, d1)
            else:
                suggested = drug_names.get(d2, d2)
            
            contributor = RiskContributor(
                drug1=drug_names.get(d1, d1),
                drug2=drug_names.get(d2, d2),
                ddi_severity=ddi_severity,
                ddi_score=ddi_score,
                shared_side_effects=shared_se,
                shared_proteins=shared_p,
                risk_contribution=contribution,
                drug1_alternatives=d1_alts,
                drug2_alternatives=d2_alts,
                suggested_replacement=suggested
            )
            contributors.append(contributor)
        
        # Sort by contribution
        contributors.sort(key=lambda x: x.risk_contribution, reverse=True)
        
        # Normalize to percentages
        total_contrib = sum(c.risk_contribution for c in contributors)
        if total_contrib > 0:
            for c in contributors:
                c.risk_contribution = c.risk_contribution / total_contrib * 100
        
        return contributors
    
    def _resolve_drug(self, drug_name: str) -> Optional[str]:
        """Resolve drug name to DrugBank ID"""
        name_lower = drug_name.lower().strip()
        
        # Direct lookup
        if name_lower in self.kg.drug_name_to_id:
            return self.kg.drug_name_to_id[name_lower]
        
        # Check aliases
        ALIASES = {
            'aspirin': 'acetylsalicylic acid',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'coumadin': 'warfarin',
            'lipitor': 'atorvastatin',
            'zocor': 'simvastatin',
            'norvasc': 'amlodipine',
            'januvia': 'sitagliptin',
            'lasix': 'furosemide',
            'synthroid': 'levothyroxine',
            'glucophage': 'metformin',
            'zoloft': 'sertraline',
            'prozac': 'fluoxetine',
            'plavix': 'clopidogrel',
            'nexium': 'esomeprazole',
            'prilosec': 'omeprazole',
        }
        
        if name_lower in ALIASES:
            alias = ALIASES[name_lower]
            if alias in self.kg.drug_name_to_id:
                return self.kg.drug_name_to_id[alias]
        
        return None
    
    def find_alternatives(self, drug_to_replace: str, 
                         other_drugs: List[str],
                         top_k: int = 10) -> List[DrugAlternative]:
        """
        Find safer therapeutic alternatives for a drug
        
        Args:
            drug_to_replace: Drug to find alternatives for
            other_drugs: Other drugs in the regimen (to check DDIs against)
            top_k: Number of alternatives to return
        
        Returns:
            List of DrugAlternative sorted by recommendation score
        """
        source_id = self._resolve_drug(drug_to_replace)
        if not source_id:
            return []
        
        other_ids = [self._resolve_drug(d) for d in other_drugs if self._resolve_drug(d)]
        
        # Get therapeutically similar drugs
        candidates = self.similarity_engine.find_similar_drugs(
            source_id, 
            min_protein_sim=0.05,
            min_disease_sim=0.05,
            min_atc_level=3,
            top_k=50
        )
        
        # Score each candidate for safety with other drugs
        for cand in candidates:
            # Compute DDI risk with other drugs in regimen
            ddi_risk = 0.0
            contraindicated = 0
            new_ddis = []
            
            for other_id in other_ids:
                ddi = self.kg.ddi_index.get((cand.drug_id, other_id))
                if ddi:
                    ddi_risk += ddi.severity_score
                    if 'Contraindicated' in ddi.severity:
                        contraindicated += 1
                    new_ddis.append({
                        'drug': self.kg.drugs.get(other_id, DrugProfile(other_id, other_id)).drug_name,
                        'severity': ddi.severity,
                        'score': ddi.severity_score
                    })
            
            cand.ddi_risk_with_regimen = ddi_risk / max(len(other_ids), 1) / 10.0
            cand.contraindicated_count = contraindicated
            cand.new_ddis = new_ddis
            
            # Safety score (higher is better)
            cand.safety_score = max(0, 1.0 - cand.ddi_risk_with_regimen - 0.3 * contraindicated)
            
            # Multi-objective recommendation score
            cand.recommendation_score = (
                self.WEIGHTS['therapeutic_similarity'] * cand.therapeutic_score +
                self.WEIGHTS['safety_improvement'] * cand.safety_score +
                self.WEIGHTS['ddi_risk_reduction'] * (1.0 - cand.ddi_risk_with_regimen)
            )
        
        # Filter out contraindicated alternatives
        candidates = [c for c in candidates if c.contraindicated_count == 0]
        
        # Sort by recommendation score
        candidates.sort(key=lambda x: x.recommendation_score, reverse=True)
        
        return candidates[:top_k]
    
    def recommend(self, drugs: List[str], 
                 max_substitutions: int = 2) -> RecommendationResult:
        """
        Generate complete recommendation for a drug regimen
        
        Args:
            drugs: List of drug names in the regimen
            max_substitutions: Maximum number of drugs to suggest replacing
        
        Returns:
            RecommendationResult with risk decomposition and recommendations
        """
        # Assess original risk
        original_assessment = self.risk_assessor.assess_polypharmacy_risk(drugs)
        
        # Decompose risk
        risk_contributors = self.decompose_risk(drugs)
        
        result = RecommendationResult(
            original_drugs=drugs,
            original_risk_score=original_assessment.overall_risk_score,
            original_risk_level=original_assessment.risk_level,
            risk_contributors=risk_contributors
        )
        
        if risk_contributors:
            result.highest_risk_pair = (
                risk_contributors[0].drug1,
                risk_contributors[0].drug2
            )
        
        # Generate substitution recommendations
        drugs_to_replace = set()
        for rc in risk_contributors[:max_substitutions]:
            drugs_to_replace.add(rc.suggested_replacement)
        
        substitutions = []
        optimized_regimen = list(drugs)
        
        for drug_to_replace in drugs_to_replace:
            other_drugs = [d for d in drugs if d.lower() != drug_to_replace.lower()]
            alternatives = self.find_alternatives(drug_to_replace, other_drugs, top_k=5)
            
            if alternatives:
                best_alt = alternatives[0]
                substitutions.append({
                    'replace': drug_to_replace,
                    'with': best_alt.drug_name,
                    'therapeutic_similarity': round(best_alt.therapeutic_score, 3),
                    'safety_score': round(best_alt.safety_score, 3),
                    'recommendation_score': round(best_alt.recommendation_score, 3),
                    'shared_proteins': best_alt.shared_proteins,
                    'shared_diseases': best_alt.shared_diseases[:3],
                    'new_ddis': best_alt.new_ddis,
                    'alternatives_considered': len(alternatives)
                })
                
                # Update optimized regimen
                idx = next((i for i, d in enumerate(optimized_regimen) 
                           if d.lower() == drug_to_replace.lower()), None)
                if idx is not None:
                    optimized_regimen[idx] = best_alt.drug_name
        
        result.recommended_substitutions = substitutions
        result.optimized_regimen = optimized_regimen
        
        # Assess optimized risk
        if optimized_regimen != drugs:
            optimized_assessment = self.risk_assessor.assess_polypharmacy_risk(optimized_regimen)
            result.optimized_risk_score = optimized_assessment.overall_risk_score
            result.risk_reduction = original_assessment.overall_risk_score - optimized_assessment.overall_risk_score
        
        return result


# ============================================================================
# VALIDATION FRAMEWORK
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for recommendation validation"""
    output_dir: Path = field(default_factory=lambda: Path("publication_kg_recommendations"))
    
    # Validation parameters
    n_bootstrap: int = 1000
    random_seed: int = 42


class KnownSubstitutionDatabase:
    """
    Gold standard drug substitutions from clinical guidelines
    
    Sources:
    - American Geriatrics Society Beers Criteria
    - STOPP/START Criteria 
    - FDA drug safety communications
    - Clinical practice guidelines
    """
    
    # Known therapeutic substitutes (drug pairs that are clinically interchangeable)
    THERAPEUTIC_EQUIVALENTS = [
        # Anticoagulants
        ('warfarin', 'apixaban', 'anticoagulant', 'safer_alternative'),
        ('warfarin', 'rivaroxaban', 'anticoagulant', 'safer_alternative'),
        ('warfarin', 'dabigatran', 'anticoagulant', 'safer_alternative'),
        
        # NSAIDs
        ('ibuprofen', 'celecoxib', 'nsaid', 'gi_safer'),
        ('naproxen', 'celecoxib', 'nsaid', 'gi_safer'),
        ('diclofenac', 'celecoxib', 'nsaid', 'gi_safer'),
        
        # Statins
        ('simvastatin', 'rosuvastatin', 'statin', 'less_interactions'),
        ('lovastatin', 'rosuvastatin', 'statin', 'less_interactions'),
        ('atorvastatin', 'rosuvastatin', 'statin', 'less_interactions'),
        
        # PPIs
        ('omeprazole', 'pantoprazole', 'ppi', 'less_cyp_interactions'),
        ('esomeprazole', 'pantoprazole', 'ppi', 'less_cyp_interactions'),
        ('lansoprazole', 'pantoprazole', 'ppi', 'less_cyp_interactions'),
        
        # Antidepressants
        ('fluoxetine', 'sertraline', 'ssri', 'less_interactions'),
        ('paroxetine', 'sertraline', 'ssri', 'less_interactions'),
        ('fluvoxamine', 'sertraline', 'ssri', 'less_interactions'),
        
        # Antihypertensives
        ('amlodipine', 'nifedipine', 'ccb', 'therapeutic_equivalent'),
        ('diltiazem', 'verapamil', 'ccb', 'therapeutic_equivalent'),
        ('lisinopril', 'ramipril', 'ace_inhibitor', 'therapeutic_equivalent'),
        ('losartan', 'valsartan', 'arb', 'therapeutic_equivalent'),
        
        # Beta blockers
        ('metoprolol', 'bisoprolol', 'beta_blocker', 'therapeutic_equivalent'),
        ('atenolol', 'bisoprolol', 'beta_blocker', 'therapeutic_equivalent'),
        ('propranolol', 'metoprolol', 'beta_blocker', 'therapeutic_equivalent'),
        
        # Antiplatelet
        ('aspirin', 'clopidogrel', 'antiplatelet', 'alternative_mechanism'),
        ('clopidogrel', 'ticagrelor', 'antiplatelet', 'newer_agent'),
        ('clopidogrel', 'prasugrel', 'antiplatelet', 'newer_agent'),
        
        # Benzodiazepines
        ('diazepam', 'lorazepam', 'benzodiazepine', 'shorter_acting'),
        ('alprazolam', 'lorazepam', 'benzodiazepine', 'shorter_acting'),
        
        # Diabetes
        ('glipizide', 'gliclazide', 'sulfonylurea', 'therapeutic_equivalent'),
        ('metformin', 'sitagliptin', 'diabetes', 'alternative_class'),
        ('pioglitazone', 'rosiglitazone', 'thiazolidinedione', 'therapeutic_equivalent'),
    ]
    
    # Known high-risk drug pairs that should trigger recommendations
    HIGH_RISK_PAIRS = [
        ('warfarin', 'aspirin', 'bleeding_risk'),
        ('warfarin', 'ibuprofen', 'bleeding_risk'),
        ('warfarin', 'naproxen', 'bleeding_risk'),
        ('clopidogrel', 'omeprazole', 'reduced_efficacy'),
        ('simvastatin', 'amiodarone', 'myopathy_risk'),
        ('simvastatin', 'diltiazem', 'myopathy_risk'),
        ('methotrexate', 'ibuprofen', 'toxicity_risk'),
        ('methotrexate', 'trimethoprim', 'toxicity_risk'),
        ('lithium', 'ibuprofen', 'toxicity_risk'),
        ('digoxin', 'amiodarone', 'toxicity_risk'),
        ('fluoxetine', 'tramadol', 'serotonin_syndrome'),
        ('sertraline', 'tramadol', 'serotonin_syndrome'),
        ('ace_inhibitor', 'potassium', 'hyperkalemia'),
        ('spironolactone', 'ace_inhibitor', 'hyperkalemia'),
    ]
    
    def __init__(self):
        self.equivalents = self.THERAPEUTIC_EQUIVALENTS
        self.high_risk = self.HIGH_RISK_PAIRS
        
        # Build lookup indices
        self.substitution_index = defaultdict(list)
        for drug1, drug2, cls, reason in self.equivalents:
            self.substitution_index[drug1.lower()].append((drug2, cls, reason))
            self.substitution_index[drug2.lower()].append((drug1, cls, reason))
    
    def get_gold_substitutes(self, drug: str) -> List[Tuple[str, str, str]]:
        """Get clinically validated substitutes for a drug"""
        return self.substitution_index.get(drug.lower(), [])
    
    def is_valid_substitution(self, original: str, replacement: str) -> bool:
        """Check if a substitution is clinically validated"""
        subs = self.substitution_index.get(original.lower(), [])
        return any(s[0].lower() == replacement.lower() for s in subs)


class RecommendationValidator:
    """
    Validate recommendation system against gold standards
    
    Metrics:
    - Precision@K: How many recommendations are clinically valid
    - Recall@K: How many gold standard substitutes are found
    - MRR: Mean Reciprocal Rank of correct substitutes
    - NDCG: Normalized Discounted Cumulative Gain
    """
    
    def __init__(self, engine: KGRecommendationEngine, config: ValidationConfig = None):
        self.engine = engine
        self.config = config or ValidationConfig()
        self.gold_standard = KnownSubstitutionDatabase()
        
        # Results
        self.validation_results = {}
    
    def compute_precision_at_k(self, drug: str, recommendations: List[DrugAlternative], k: int) -> float:
        """Precision@K: fraction of top-K recommendations that are valid"""
        gold_subs = set(s[0].lower() for s in self.gold_standard.get_gold_substitutes(drug))
        
        if not gold_subs:
            return np.nan
        
        top_k = recommendations[:k]
        hits = sum(1 for r in top_k if r.drug_name.lower() in gold_subs)
        
        return hits / k
    
    def compute_recall_at_k(self, drug: str, recommendations: List[DrugAlternative], k: int) -> float:
        """Recall@K: fraction of gold substitutes found in top-K"""
        gold_subs = set(s[0].lower() for s in self.gold_standard.get_gold_substitutes(drug))
        
        if not gold_subs:
            return np.nan
        
        top_k_names = set(r.drug_name.lower() for r in recommendations[:k])
        hits = len(gold_subs & top_k_names)
        
        return hits / len(gold_subs)
    
    def compute_mrr(self, drug: str, recommendations: List[DrugAlternative]) -> float:
        """Mean Reciprocal Rank: 1/rank of first correct substitute"""
        gold_subs = set(s[0].lower() for s in self.gold_standard.get_gold_substitutes(drug))
        
        if not gold_subs:
            return np.nan
        
        for i, rec in enumerate(recommendations):
            if rec.drug_name.lower() in gold_subs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_ndcg(self, drug: str, recommendations: List[DrugAlternative], k: int = 10) -> float:
        """Normalized DCG for relevance ranking"""
        gold_subs = set(s[0].lower() for s in self.gold_standard.get_gold_substitutes(drug))
        
        if not gold_subs:
            return np.nan
        
        # Relevance: 1 if in gold standard, 0 otherwise
        relevance = [1 if r.drug_name.lower() in gold_subs else 0 for r in recommendations[:k]]
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        
        # Ideal DCG (all gold standards at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def validate_substitution_quality(self) -> Dict[str, Any]:
        """
        Validate recommendation quality against gold standard substitutions
        """
        results = {
            'precision_at_1': [],
            'precision_at_5': [],
            'precision_at_10': [],
            'recall_at_5': [],
            'recall_at_10': [],
            'mrr': [],
            'ndcg_at_10': [],
            'drug_results': []
        }
        
        # Test drugs
        test_drugs = list(set([d for d, _, _, _ in self.gold_standard.equivalents]))
        
        for drug in test_drugs:
            drug_id = self.engine._resolve_drug(drug)
            if not drug_id:
                continue
            
            # Get recommendations
            alternatives = self.engine.similarity_engine.find_similar_drugs(drug_id, top_k=20)
            
            if not alternatives:
                continue
            
            # Compute metrics
            p1 = self.compute_precision_at_k(drug, alternatives, 1)
            p5 = self.compute_precision_at_k(drug, alternatives, 5)
            p10 = self.compute_precision_at_k(drug, alternatives, 10)
            r5 = self.compute_recall_at_k(drug, alternatives, 5)
            r10 = self.compute_recall_at_k(drug, alternatives, 10)
            mrr = self.compute_mrr(drug, alternatives)
            ndcg = self.compute_ndcg(drug, alternatives, 10)
            
            if not np.isnan(p1):
                results['precision_at_1'].append(p1)
            if not np.isnan(p5):
                results['precision_at_5'].append(p5)
            if not np.isnan(p10):
                results['precision_at_10'].append(p10)
            if not np.isnan(r5):
                results['recall_at_5'].append(r5)
            if not np.isnan(r10):
                results['recall_at_10'].append(r10)
            if not np.isnan(mrr):
                results['mrr'].append(mrr)
            if not np.isnan(ndcg):
                results['ndcg_at_10'].append(ndcg)
            
            results['drug_results'].append({
                'drug': drug,
                'n_alternatives': len(alternatives),
                'precision_at_5': p5,
                'recall_at_5': r5,
                'mrr': mrr,
                'top_3_recommendations': [a.drug_name for a in alternatives[:3]]
            })
        
        # Aggregate metrics
        results['summary'] = {
            'n_drugs_evaluated': len(results['drug_results']),
            'mean_precision_at_1': np.mean(results['precision_at_1']) if results['precision_at_1'] else 0,
            'mean_precision_at_5': np.mean(results['precision_at_5']) if results['precision_at_5'] else 0,
            'mean_precision_at_10': np.mean(results['precision_at_10']) if results['precision_at_10'] else 0,
            'mean_recall_at_5': np.mean(results['recall_at_5']) if results['recall_at_5'] else 0,
            'mean_recall_at_10': np.mean(results['recall_at_10']) if results['recall_at_10'] else 0,
            'mean_mrr': np.mean(results['mrr']) if results['mrr'] else 0,
            'mean_ndcg_at_10': np.mean(results['ndcg_at_10']) if results['ndcg_at_10'] else 0,
        }
        
        self.validation_results['substitution_quality'] = results
        return results
    
    def validate_risk_reduction(self) -> Dict[str, Any]:
        """
        Validate that recommendations actually reduce risk
        """
        results = {
            'test_cases': [],
            'risk_reductions': [],
            'summary': {}
        }
        
        # Test cases: known high-risk combinations
        test_regimens = [
            ['warfarin', 'aspirin', 'metoprolol'],
            ['clopidogrel', 'omeprazole', 'atorvastatin'],
            ['simvastatin', 'amiodarone', 'lisinopril'],
            ['methotrexate', 'ibuprofen', 'prednisone'],
            ['lithium', 'ibuprofen', 'hydrochlorothiazide'],
            ['fluoxetine', 'tramadol', 'alprazolam'],
            ['digoxin', 'amiodarone', 'furosemide'],
        ]
        
        for regimen in test_regimens:
            try:
                rec = self.engine.recommend(regimen)
                
                case = {
                    'original_regimen': regimen,
                    'original_risk': rec.original_risk_score,
                    'original_level': rec.original_risk_level,
                    'optimized_regimen': rec.optimized_regimen,
                    'optimized_risk': rec.optimized_risk_score,
                    'risk_reduction': rec.risk_reduction,
                    'substitutions': rec.recommended_substitutions
                }
                
                results['test_cases'].append(case)
                
                if rec.risk_reduction > 0:
                    results['risk_reductions'].append(rec.risk_reduction)
                    
            except Exception as e:
                continue
        
        # Summary statistics
        if results['risk_reductions']:
            results['summary'] = {
                'n_cases': len(results['test_cases']),
                'n_improved': len(results['risk_reductions']),
                'mean_risk_reduction': np.mean(results['risk_reductions']),
                'median_risk_reduction': np.median(results['risk_reductions']),
                'max_risk_reduction': max(results['risk_reductions']),
                'improvement_rate': len(results['risk_reductions']) / len(results['test_cases'])
            }
        
        self.validation_results['risk_reduction'] = results
        return results
    
    def generate_validation_report(self) -> str:
        """Generate markdown validation report"""
        report = []
        report.append("# KG-Based Recommendation System Validation Report\n")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        # Substitution quality
        if 'substitution_quality' in self.validation_results:
            sq = self.validation_results['substitution_quality']['summary']
            report.append("## 1. Substitution Quality Metrics\n")
            report.append("Validated against clinical guideline substitution pairs.\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Drugs Evaluated | {sq['n_drugs_evaluated']} |")
            report.append(f"| Precision@1 | {sq['mean_precision_at_1']:.3f} |")
            report.append(f"| Precision@5 | {sq['mean_precision_at_5']:.3f} |")
            report.append(f"| Precision@10 | {sq['mean_precision_at_10']:.3f} |")
            report.append(f"| Recall@5 | {sq['mean_recall_at_5']:.3f} |")
            report.append(f"| Recall@10 | {sq['mean_recall_at_10']:.3f} |")
            report.append(f"| MRR | {sq['mean_mrr']:.3f} |")
            report.append(f"| NDCG@10 | {sq['mean_ndcg_at_10']:.3f} |")
            report.append("")
        
        # Risk reduction
        if 'risk_reduction' in self.validation_results:
            rr = self.validation_results['risk_reduction']['summary']
            report.append("## 2. Risk Reduction Validation\n")
            report.append("Validates that recommendations reduce polypharmacy risk.\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Test Cases | {rr.get('n_cases', 0)} |")
            report.append(f"| Cases Improved | {rr.get('n_improved', 0)} |")
            report.append(f"| Improvement Rate | {rr.get('improvement_rate', 0)*100:.1f}% |")
            report.append(f"| Mean Risk Reduction | {rr.get('mean_risk_reduction', 0):.3f} |")
            report.append(f"| Max Risk Reduction | {rr.get('max_risk_reduction', 0):.3f} |")
            report.append("")
            
            # Case details
            if 'test_cases' in self.validation_results['risk_reduction']:
                report.append("### Test Case Details\n")
                report.append("| Original Regimen | Risk | Optimized | New Risk | Reduction |")
                report.append("|-----------------|------|-----------|----------|-----------|")
                for case in self.validation_results['risk_reduction']['test_cases'][:10]:
                    orig = ', '.join(case['original_regimen'])
                    opt = ', '.join(case['optimized_regimen'])
                    report.append(f"| {orig} | {case['original_risk']:.3f} | {opt} | {case['optimized_risk']:.3f} | {case['risk_reduction']:.3f} |")
        
        return '\n'.join(report)


# ============================================================================
# PUBLICATION OUTPUT GENERATOR
# ============================================================================

class PublicationOutputGenerator:
    """Generate publication-ready outputs"""
    
    def __init__(self, validator: RecommendationValidator, config: ValidationConfig = None):
        self.validator = validator
        self.config = config or ValidationConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / 'tables').mkdir(exist_ok=True)
        (self.config.output_dir / 'figures').mkdir(exist_ok=True)
        (self.config.output_dir / 'data').mkdir(exist_ok=True)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary validation table"""
        rows = []
        
        if 'substitution_quality' in self.validator.validation_results:
            sq = self.validator.validation_results['substitution_quality']['summary']
            rows.extend([
                {'Metric': 'Precision@1', 'Value': f"{sq['mean_precision_at_1']:.3f}", 'Description': 'Top recommendation accuracy'},
                {'Metric': 'Precision@5', 'Value': f"{sq['mean_precision_at_5']:.3f}", 'Description': 'Top-5 recommendation accuracy'},
                {'Metric': 'Recall@5', 'Value': f"{sq['mean_recall_at_5']:.3f}", 'Description': 'Coverage of gold standards in top-5'},
                {'Metric': 'MRR', 'Value': f"{sq['mean_mrr']:.3f}", 'Description': 'Mean Reciprocal Rank'},
                {'Metric': 'NDCG@10', 'Value': f"{sq['mean_ndcg_at_10']:.3f}", 'Description': 'Normalized ranking quality'},
            ])
        
        if 'risk_reduction' in self.validator.validation_results:
            rr = self.validator.validation_results['risk_reduction']['summary']
            rows.extend([
                {'Metric': 'Improvement Rate', 'Value': f"{rr.get('improvement_rate', 0)*100:.1f}%", 'Description': 'Cases where risk is reduced'},
                {'Metric': 'Mean Risk Reduction', 'Value': f"{rr.get('mean_risk_reduction', 0):.3f}", 'Description': 'Average risk score decrease'},
            ])
        
        df = pd.DataFrame(rows)
        
        # Save
        df.to_csv(self.config.output_dir / 'tables' / 'table1_validation_summary.csv', index=False)
        df.to_markdown(self.config.output_dir / 'tables' / 'table1_validation_summary.md', index=False)
        
        return df
    
    def generate_case_study_table(self) -> pd.DataFrame:
        """Generate case study examples table"""
        if 'risk_reduction' not in self.validator.validation_results:
            return pd.DataFrame()
        
        cases = self.validator.validation_results['risk_reduction']['test_cases']
        
        rows = []
        for case in cases:
            subs = case.get('substitutions', [])
            sub_str = '; '.join([f"{s['replace']}→{s['with']}" for s in subs]) if subs else 'None'
            
            rows.append({
                'Original Regimen': ', '.join(case['original_regimen']),
                'Initial Risk': f"{case['original_risk']:.3f}",
                'Risk Level': case['original_level'],
                'Substitutions': sub_str,
                'Optimized Risk': f"{case['optimized_risk']:.3f}",
                'Risk Reduction': f"{case['risk_reduction']:.3f}"
            })
        
        df = pd.DataFrame(rows)
        
        df.to_csv(self.config.output_dir / 'tables' / 'table2_case_studies.csv', index=False)
        df.to_markdown(self.config.output_dir / 'tables' / 'table2_case_studies.md', index=False)
        
        return df
    
    def generate_all(self):
        """Generate all publication outputs"""
        print("\n📊 Generating publication outputs...")
        
        # Tables
        self.generate_summary_table()
        print("   Saved: table1_validation_summary.csv/md")
        
        self.generate_case_study_table()
        print("   Saved: table2_case_studies.csv/md")
        
        # Report
        report = self.validator.generate_validation_report()
        with open(self.config.output_dir / 'VALIDATION_REPORT.md', 'w') as f:
            f.write(report)
        print("   Saved: VALIDATION_REPORT.md")
        
        # Raw data
        with open(self.config.output_dir / 'data' / 'validation_results.json', 'w') as f:
            # Convert to JSON-serializable
            results = {}
            for key, val in self.validator.validation_results.items():
                if isinstance(val, dict):
                    results[key] = {k: v for k, v in val.items() 
                                   if not isinstance(v, (np.ndarray, list)) or len(str(v)) < 10000}
            json.dump(results, f, indent=2, default=str)
        print("   Saved: validation_results.json")
        
        print(f"\n📁 Results saved to: {self.config.output_dir}/")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class KGRecommendationPipeline:
    """Main pipeline for recommendation and validation"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.engine = None
        self.validator = None
    
    def run(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        print("=" * 70)
        print("KG-BASED DRUG RECOMMENDATION SYSTEM")
        print("Publication-Grade Validation")
        print("=" * 70)
        
        # Step 1: Load KG and build engine
        print("\n📚 Step 1: Loading Knowledge Graph...")
        kg = KnowledgeGraphLoader().load()
        self.engine = KGRecommendationEngine(kg)
        
        # Step 2: Initialize validator
        print("\n🔍 Step 2: Initializing validator...")
        self.validator = RecommendationValidator(self.engine, self.config)
        
        # Step 3: Validate substitution quality
        print("\n📊 Step 3: Validating substitution quality...")
        sq_results = self.validator.validate_substitution_quality()
        print(f"   Evaluated {sq_results['summary']['n_drugs_evaluated']} drugs")
        print(f"   Mean Precision@5: {sq_results['summary']['mean_precision_at_5']:.3f}")
        print(f"   Mean MRR: {sq_results['summary']['mean_mrr']:.3f}")
        
        # Step 4: Validate risk reduction
        print("\n📈 Step 4: Validating risk reduction...")
        rr_results = self.validator.validate_risk_reduction()
        print(f"   Tested {rr_results['summary'].get('n_cases', 0)} regimens")
        print(f"   Improvement rate: {rr_results['summary'].get('improvement_rate', 0)*100:.1f}%")
        print(f"   Mean risk reduction: {rr_results['summary'].get('mean_risk_reduction', 0):.3f}")
        
        # Step 5: Generate outputs
        output_gen = PublicationOutputGenerator(self.validator, self.config)
        output_gen.generate_all()
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print("\n📊 Substitution Quality:")
        print(f"   Precision@1: {sq_results['summary']['mean_precision_at_1']:.3f}")
        print(f"   Precision@5: {sq_results['summary']['mean_precision_at_5']:.3f}")
        print(f"   Recall@5: {sq_results['summary']['mean_recall_at_5']:.3f}")
        print(f"   MRR: {sq_results['summary']['mean_mrr']:.3f}")
        print(f"   NDCG@10: {sq_results['summary']['mean_ndcg_at_10']:.3f}")
        
        print("\n📈 Risk Reduction:")
        print(f"   Improvement Rate: {rr_results['summary'].get('improvement_rate', 0)*100:.1f}%")
        print(f"   Mean Reduction: {rr_results['summary'].get('mean_risk_reduction', 0):.3f}")
        
        return {
            'substitution_quality': sq_results,
            'risk_reduction': rr_results
        }


# ============================================================================
# CLI
# ============================================================================

def demo_recommendation():
    """Demo the recommendation system"""
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM DEMO")
    print("=" * 70)
    
    kg = KnowledgeGraphLoader().load()
    engine = KGRecommendationEngine(kg)
    
    # Test case: High-risk anticoagulant + NSAID combination
    test_regimen = ['warfarin', 'aspirin', 'metoprolol', 'lisinopril']
    
    print(f"\n📋 Test Regimen: {', '.join(test_regimen)}")
    
    result = engine.recommend(test_regimen)
    
    print(f"\n⚠️ Original Risk: {result.original_risk_score:.3f} ({result.original_risk_level})")
    
    print("\n🔍 Risk Decomposition:")
    for i, rc in enumerate(result.risk_contributors[:3], 1):
        print(f"   {i}. {rc.drug1} + {rc.drug2}: {rc.ddi_severity}")
        print(f"      Contribution: {rc.risk_contribution:.1f}%")
        print(f"      Suggested replacement: {rc.suggested_replacement}")
    
    print("\n💊 Recommended Substitutions:")
    for sub in result.recommended_substitutions:
        print(f"   Replace: {sub['replace']}")
        print(f"   With: {sub['with']}")
        print(f"   Therapeutic similarity: {sub['therapeutic_similarity']:.3f}")
        print(f"   Safety score: {sub['safety_score']:.3f}")
        if sub.get('shared_proteins'):
            print(f"   Shared targets: {', '.join(sub['shared_proteins'][:3])}")
    
    print(f"\n✅ Optimized Regimen: {', '.join(result.optimized_regimen)}")
    print(f"   New Risk: {result.optimized_risk_score:.3f}")
    print(f"   Risk Reduction: {result.risk_reduction:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='KG-based drug recommendation system'
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demo recommendation')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation pipeline')
    parser.add_argument('--recommend', nargs='+',
                       help='Get recommendation for drug regimen')
    parser.add_argument('--output-dir', type=str, 
                       default='publication_kg_recommendations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_recommendation()
    elif args.validate:
        config = ValidationConfig(output_dir=Path(args.output_dir))
        pipeline = KGRecommendationPipeline(config)
        pipeline.run()
    elif args.recommend:
        kg = KnowledgeGraphLoader().load()
        engine = KGRecommendationEngine(kg)
        result = engine.recommend(args.recommend)
        
        print(f"\nOriginal Risk: {result.original_risk_score:.3f} ({result.original_risk_level})")
        print(f"Risk Contributors: {len(result.risk_contributors)}")
        for rc in result.risk_contributors[:3]:
            print(f"  - {rc.drug1} + {rc.drug2}: {rc.risk_contribution:.1f}%")
        
        print(f"\nRecommended Substitutions:")
        for sub in result.recommended_substitutions:
            print(f"  {sub['replace']} → {sub['with']} (score: {sub['recommendation_score']:.3f})")
        
        print(f"\nOptimized Regimen: {', '.join(result.optimized_regimen)}")
        print(f"New Risk: {result.optimized_risk_score:.3f}")
        print(f"Risk Reduction: {result.risk_reduction:.3f}")
    else:
        # Default: run validation
        config = ValidationConfig(output_dir=Path(args.output_dir))
        pipeline = KGRecommendationPipeline(config)
        pipeline.run()
        
        print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
