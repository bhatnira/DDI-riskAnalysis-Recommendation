#!/usr/bin/env python3
"""
Knowledge Graph-Based Polypharmacy Risk Assessment

Uses the fact-based knowledge graph to provide comprehensive polypharmacy risk analysis:
1. DDI Network Analysis - Drug-drug interactions with severity
2. Side Effect Overlap - Shared adverse effects amplify risk
3. Protein Target Overlap - Mechanism-based risk assessment
4. Disease Associations - Comorbidity considerations
5. Pathway Analysis - Metabolic pathway conflicts

Author: DDI Risk Analysis Research Team
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import networkx as nx
from itertools import combinations


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class KGConfig:
    kg_dir: Path = field(default_factory=lambda: Path("knowledge_graph_fact_based/neo4j_export"))
    
    # Severity weights for DDI risk scoring
    severity_weights: Dict[str, float] = field(default_factory=lambda: {
        'Contraindicated interaction': 10.0,
        'Major interaction': 7.0,
        'Moderate interaction': 4.0,
        'Minor interaction': 1.0
    })
    
    # Risk component weights
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        'ddi_severity': 0.40,        # Direct interaction severity
        'side_effect_overlap': 0.25, # Shared side effects
        'protein_overlap': 0.15,     # Mechanism overlap
        'pathway_conflict': 0.10,    # Metabolic pathway issues
        'network_centrality': 0.10   # Network position risk
    })


# ============================================================================
# KNOWLEDGE GRAPH DATA STRUCTURES
# ============================================================================

@dataclass
class DrugProfile:
    """Complete drug profile from knowledge graph"""
    drug_id: str
    drug_name: str
    
    # Connections
    proteins: Set[str] = field(default_factory=set)
    side_effects: Set[str] = field(default_factory=set)
    diseases: Set[str] = field(default_factory=set)
    pathways: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    
    # DDIs
    ddi_partners: Dict[str, Dict] = field(default_factory=dict)  # drug_id -> {severity, description}
    
    # Network metrics
    degree: int = 0
    weighted_degree: float = 0.0


@dataclass
class DDIRecord:
    """Drug-drug interaction record"""
    drug1_id: str
    drug2_id: str
    severity: str
    description: str
    severity_score: float = 0.0


@dataclass
class PolypharmacyRiskResult:
    """Complete risk assessment result"""
    drugs: List[str]
    overall_risk_score: float
    risk_level: str  # Critical, High, Moderate, Low
    
    # Component scores
    ddi_risk: float = 0.0
    side_effect_risk: float = 0.0
    protein_overlap_risk: float = 0.0
    pathway_risk: float = 0.0
    network_risk: float = 0.0
    
    # Detailed findings
    ddi_pairs: List[Dict] = field(default_factory=list)
    shared_side_effects: List[Dict] = field(default_factory=list)
    shared_proteins: List[Dict] = field(default_factory=list)
    pathway_conflicts: List[Dict] = field(default_factory=list)
    
    # Recommendations
    high_risk_pairs: List[Tuple[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# KNOWLEDGE GRAPH LOADER
# ============================================================================

class KnowledgeGraphLoader:
    """Load and index the fact-based knowledge graph"""
    
    def __init__(self, config: KGConfig = None):
        self.config = config or KGConfig()
        
        # Data stores
        self.drugs: Dict[str, DrugProfile] = {}
        self.drug_name_to_id: Dict[str, str] = {}
        self.ddis: List[DDIRecord] = []
        self.ddi_index: Dict[Tuple[str, str], DDIRecord] = {}
        
        # Side effect data
        self.drug_side_effects: Dict[str, Set[str]] = defaultdict(set)
        self.side_effect_drugs: Dict[str, Set[str]] = defaultdict(set)
        self.side_effect_names: Dict[str, str] = {}
        
        # Protein data
        self.drug_proteins: Dict[str, Set[str]] = defaultdict(set)
        self.protein_drugs: Dict[str, Set[str]] = defaultdict(set)
        self.protein_names: Dict[str, str] = {}
        
        # Pathway data
        self.drug_pathways: Dict[str, Set[str]] = defaultdict(set)
        self.pathway_drugs: Dict[str, Set[str]] = defaultdict(set)
        
        # Disease data
        self.drug_diseases: Dict[str, Set[str]] = defaultdict(set)
        self.disease_drugs: Dict[str, Set[str]] = defaultdict(set)
        
        # Network
        self.ddi_network: Optional[nx.Graph] = None
        
    def load(self) -> 'KnowledgeGraphLoader':
        """Load all knowledge graph data"""
        print("📚 Loading Knowledge Graph...")
        
        self._load_drugs()
        self._load_ddis()
        self._load_side_effects()
        self._load_proteins()
        self._load_pathways()
        self._load_diseases()
        self._build_network()
        self._compute_network_metrics()
        
        print(f"✅ Loaded: {len(self.drugs):,} drugs, {len(self.ddis):,} DDIs")
        print(f"   Side effects: {len(self.side_effect_names):,}")
        print(f"   Proteins: {len(self.protein_names):,}")
        
        return self
    
    def _load_drugs(self):
        """Load drug nodes"""
        path = self.config.kg_dir / "drugs.csv"
        if not path.exists():
            print(f"   ⚠️ Drug file not found: {path}")
            return
            
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            drug_id = row['drugbank_id']
            drug_name = row['name'].lower() if pd.notna(row['name']) else drug_id.lower()
            
            self.drugs[drug_id] = DrugProfile(
                drug_id=drug_id,
                drug_name=drug_name
            )
            self.drug_name_to_id[drug_name] = drug_id
            
            # Also index by DrugBank ID lowercase
            self.drug_name_to_id[drug_id.lower()] = drug_id
            
        print(f"   Loaded {len(self.drugs):,} drugs")
    
    def _load_ddis(self):
        """Load DDI edges"""
        path = self.config.kg_dir / "ddi_edges.csv"
        if not path.exists():
            print(f"   ⚠️ DDI file not found: {path}")
            return
            
        df = pd.read_csv(path)
        
        for _, row in df.iterrows():
            drug1_id = row['drug1_id']
            drug2_id = row['drug2_id']
            severity = row['severity'] if pd.notna(row['severity']) else 'Moderate interaction'
            description = row['description'] if pd.notna(row['description']) else ''
            
            severity_score = self.config.severity_weights.get(severity, 4.0)
            
            ddi = DDIRecord(
                drug1_id=drug1_id,
                drug2_id=drug2_id,
                severity=severity,
                description=description,
                severity_score=severity_score
            )
            
            self.ddis.append(ddi)
            
            # Index by pair (both directions)
            self.ddi_index[(drug1_id, drug2_id)] = ddi
            self.ddi_index[(drug2_id, drug1_id)] = ddi
            
            # Update drug profiles
            if drug1_id in self.drugs:
                self.drugs[drug1_id].ddi_partners[drug2_id] = {
                    'severity': severity,
                    'description': description,
                    'score': severity_score
                }
            if drug2_id in self.drugs:
                self.drugs[drug2_id].ddi_partners[drug1_id] = {
                    'severity': severity,
                    'description': description,
                    'score': severity_score
                }
        
        print(f"   Loaded {len(self.ddis):,} DDIs")
    
    def _load_side_effects(self):
        """Load side effect data"""
        # Load side effect names
        se_path = self.config.kg_dir / "side_effects.csv"
        if se_path.exists():
            df = pd.read_csv(se_path)
            for _, row in df.iterrows():
                se_id = row['umls_cui']  # Column name in actual CSV
                se_name = row['name'] if pd.notna(row.get('name')) else se_id
                self.side_effect_names[se_id] = se_name
        
        # Load drug-side effect edges
        edge_path = self.config.kg_dir / "drug_side_effect_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            for _, row in df.iterrows():
                drug_id = row['drug_id']
                se_id = row['side_effect_id']
                
                self.drug_side_effects[drug_id].add(se_id)
                self.side_effect_drugs[se_id].add(drug_id)
                
                if drug_id in self.drugs:
                    self.drugs[drug_id].side_effects.add(se_id)
            
            print(f"   Loaded {len(df):,} drug-side_effect edges")
    
    def _load_proteins(self):
        """Load protein target data"""
        # Load protein names
        prot_path = self.config.kg_dir / "proteins.csv"
        if prot_path.exists():
            df = pd.read_csv(prot_path)
            for _, row in df.iterrows():
                prot_id = row['protein_id']
                prot_name = row.get('name', '')
                self.protein_names[prot_id] = prot_name if pd.notna(prot_name) and prot_name else prot_id
        
        # Load drug-protein edges
        edge_path = self.config.kg_dir / "drug_protein_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            for _, row in df.iterrows():
                drug_id = row['drug_id']
                prot_id = row['protein_id']
                
                self.drug_proteins[drug_id].add(prot_id)
                self.protein_drugs[prot_id].add(drug_id)
                
                if drug_id in self.drugs:
                    self.drugs[drug_id].proteins.add(prot_id)
            
            print(f"   Loaded {len(df):,} drug-protein edges")
    
    def _load_pathways(self):
        """Load pathway data"""
        edge_path = self.config.kg_dir / "drug_pathway_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            for _, row in df.iterrows():
                drug_id = row['drug_id']
                pathway_id = row['pathway_id']
                
                self.drug_pathways[drug_id].add(pathway_id)
                self.pathway_drugs[pathway_id].add(drug_id)
                
                if drug_id in self.drugs:
                    self.drugs[drug_id].pathways.add(pathway_id)
            
            print(f"   Loaded {len(df):,} drug-pathway edges")
    
    def _load_diseases(self):
        """Load disease association data"""
        edge_path = self.config.kg_dir / "drug_disease_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            for _, row in df.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                
                self.drug_diseases[drug_id].add(disease_id)
                self.disease_drugs[disease_id].add(drug_id)
                
                if drug_id in self.drugs:
                    self.drugs[drug_id].diseases.add(disease_id)
            
            print(f"   Loaded {len(df):,} drug-disease edges")
    
    def _build_network(self):
        """Build NetworkX graph for centrality analysis"""
        self.ddi_network = nx.Graph()
        
        for ddi in self.ddis:
            self.ddi_network.add_edge(
                ddi.drug1_id,
                ddi.drug2_id,
                weight=ddi.severity_score,
                severity=ddi.severity
            )
        
        print(f"   Built network: {self.ddi_network.number_of_nodes()} nodes, "
              f"{self.ddi_network.number_of_edges()} edges")
    
    def _compute_network_metrics(self):
        """Compute network centrality metrics"""
        if self.ddi_network is None:
            return
        
        # Degree
        degrees = dict(self.ddi_network.degree())
        weighted_degrees = dict(self.ddi_network.degree(weight='weight'))
        
        # Normalize
        max_degree = max(degrees.values()) if degrees else 1
        max_weighted = max(weighted_degrees.values()) if weighted_degrees else 1
        
        for drug_id in self.drugs:
            if drug_id in degrees:
                self.drugs[drug_id].degree = degrees[drug_id]
                self.drugs[drug_id].weighted_degree = weighted_degrees.get(drug_id, 0) / max_weighted
    
    # Common brand/common names to generic mappings
    DRUG_ALIASES = {
        'aspirin': 'acetylsalicylic acid',
        'tylenol': 'acetaminophen',
        'advil': 'ibuprofen',
        'motrin': 'ibuprofen',
        'aleve': 'naproxen',
        'coumadin': 'warfarin',
        'plavix': 'clopidogrel',
        'lipitor': 'atorvastatin',
        'zocor': 'simvastatin',
        'crestor': 'rosuvastatin',
        'xarelto': 'rivaroxaban',
        'eliquis': 'apixaban',
        'pradaxa': 'dabigatran',
        'lasix': 'furosemide',
        'hctz': 'hydrochlorothiazide',
        'prilosec': 'omeprazole',
        'nexium': 'esomeprazole',
        'synthroid': 'levothyroxine',
        'lopressor': 'metoprolol',
        'toprol': 'metoprolol',
        'norvasc': 'amlodipine',
        'prinivil': 'lisinopril',
        'zestril': 'lisinopril',
        'vasotec': 'enalapril',
        'cozaar': 'losartan',
        'diovan': 'valsartan',
        'prozac': 'fluoxetine',
        'zoloft': 'sertraline',
        'lexapro': 'escitalopram',
        'cymbalta': 'duloxetine',
        'xanax': 'alprazolam',
        'valium': 'diazepam',
        'ativan': 'lorazepam',
        'ambien': 'zolpidem',
        'glucophage': 'metformin',
    }
    
    def resolve_drug(self, drug_input: str) -> Optional[str]:
        """Resolve drug name/ID to DrugBank ID"""
        # Direct match
        if drug_input in self.drugs:
            return drug_input
        
        # Name lookup (case-insensitive)
        drug_lower = drug_input.lower().strip()
        
        # Check aliases first
        if drug_lower in self.DRUG_ALIASES:
            drug_lower = self.DRUG_ALIASES[drug_lower]
        
        if drug_lower in self.drug_name_to_id:
            return self.drug_name_to_id[drug_lower]
        
        # Partial match (exact substring)
        for name, drug_id in self.drug_name_to_id.items():
            if drug_lower == name or name == drug_lower:
                return drug_id
        
        # Fuzzy partial match (contains)
        for name, drug_id in self.drug_name_to_id.items():
            if drug_lower in name or name in drug_lower:
                return drug_id
        
        return None


# ============================================================================
# POLYPHARMACY RISK ASSESSOR
# ============================================================================

class PolypharmacyRiskAssessor:
    """
    Assess polypharmacy risk using knowledge graph
    
    Components:
    1. DDI Severity Analysis - Direct interaction risk
    2. Side Effect Overlap - Additive toxicity risk
    3. Protein Target Overlap - Mechanism-based risk
    4. Pathway Conflicts - Metabolic competition
    5. Network Position - Systemic risk from drug centrality
    """
    
    RISK_LEVELS = [
        (0.8, 'CRITICAL'),
        (0.6, 'HIGH'),
        (0.4, 'MODERATE'),
        (0.2, 'LOW'),
        (0.0, 'MINIMAL')
    ]
    
    # High-risk side effects (serious adverse events)
    HIGH_RISK_SIDE_EFFECTS = {
        'bleeding', 'hemorrhage', 'cardiac', 'arrhythmia', 'qt prolongation',
        'hepatotoxicity', 'nephrotoxicity', 'seizure', 'respiratory depression',
        'hypotension', 'hypertension', 'hypoglycemia', 'hyperkalemia',
        'serotonin syndrome', 'neuroleptic malignant', 'rhabdomyolysis',
        'thrombocytopenia', 'agranulocytosis', 'anaphylaxis'
    }
    
    def __init__(self, kg: KnowledgeGraphLoader, config: KGConfig = None):
        self.kg = kg
        self.config = config or KGConfig()
    
    def assess_polypharmacy_risk(self, drug_list: List[str]) -> PolypharmacyRiskResult:
        """
        Comprehensive polypharmacy risk assessment
        
        Args:
            drug_list: List of drug names or DrugBank IDs
            
        Returns:
            PolypharmacyRiskResult with scores and recommendations
        """
        # Resolve drug names to IDs
        resolved_drugs = []
        unresolved = []
        
        for drug in drug_list:
            drug_id = self.kg.resolve_drug(drug)
            if drug_id:
                resolved_drugs.append(drug_id)
            else:
                unresolved.append(drug)
        
        if unresolved:
            print(f"⚠️ Could not resolve: {unresolved}")
        
        if len(resolved_drugs) < 2:
            return PolypharmacyRiskResult(
                drugs=drug_list,
                overall_risk_score=0.0,
                risk_level='MINIMAL',
                recommendations=[f"Need at least 2 drugs for polypharmacy risk. Unresolved: {unresolved}"]
            )
        
        # Compute component risks
        ddi_risk, ddi_pairs = self._compute_ddi_risk(resolved_drugs)
        se_risk, shared_se = self._compute_side_effect_risk(resolved_drugs)
        protein_risk, shared_proteins = self._compute_protein_overlap_risk(resolved_drugs)
        pathway_risk, pathway_conflicts = self._compute_pathway_risk(resolved_drugs)
        network_risk = self._compute_network_risk(resolved_drugs)
        
        # Weighted combination
        weights = self.config.risk_weights
        overall_risk = (
            weights['ddi_severity'] * ddi_risk +
            weights['side_effect_overlap'] * se_risk +
            weights['protein_overlap'] * protein_risk +
            weights['pathway_conflict'] * pathway_risk +
            weights['network_centrality'] * network_risk
        )
        
        # Determine risk level
        risk_level = 'MINIMAL'
        for threshold, level in self.RISK_LEVELS:
            if overall_risk >= threshold:
                risk_level = level
                break
        
        # Get high-risk pairs
        high_risk_pairs = [
            (pair['drug1'], pair['drug2'])
            for pair in ddi_pairs
            if pair['severity'] in ('Contraindicated interaction', 'Major interaction')
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            resolved_drugs, ddi_pairs, shared_se, shared_proteins, risk_level
        )
        
        # Get drug names for result
        drug_names = [
            self.kg.drugs[d].drug_name if d in self.kg.drugs else d
            for d in resolved_drugs
        ]
        
        return PolypharmacyRiskResult(
            drugs=drug_names,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            ddi_risk=ddi_risk,
            side_effect_risk=se_risk,
            protein_overlap_risk=protein_risk,
            pathway_risk=pathway_risk,
            network_risk=network_risk,
            ddi_pairs=ddi_pairs,
            shared_side_effects=shared_se,
            shared_proteins=shared_proteins,
            pathway_conflicts=pathway_conflicts,
            high_risk_pairs=high_risk_pairs,
            recommendations=recommendations
        )
    
    def _compute_ddi_risk(self, drug_ids: List[str]) -> Tuple[float, List[Dict]]:
        """Compute DDI-based risk score"""
        ddi_pairs = []
        total_severity = 0.0
        max_possible = 0.0
        
        for d1, d2 in combinations(drug_ids, 2):
            key = (d1, d2)
            if key in self.kg.ddi_index:
                ddi = self.kg.ddi_index[key]
                
                d1_name = self.kg.drugs[d1].drug_name if d1 in self.kg.drugs else d1
                d2_name = self.kg.drugs[d2].drug_name if d2 in self.kg.drugs else d2
                
                ddi_pairs.append({
                    'drug1': d1_name,
                    'drug2': d2_name,
                    'drug1_id': d1,
                    'drug2_id': d2,
                    'severity': ddi.severity,
                    'severity_score': ddi.severity_score,
                    'description': ddi.description[:200] if ddi.description else ''
                })
                
                total_severity += ddi.severity_score
            
            # Max possible is all contraindicated
            max_possible += 10.0
        
        # Normalize to [0, 1]
        risk_score = total_severity / max_possible if max_possible > 0 else 0.0
        
        # Sort by severity
        ddi_pairs.sort(key=lambda x: x['severity_score'], reverse=True)
        
        return risk_score, ddi_pairs
    
    def _compute_side_effect_risk(self, drug_ids: List[str]) -> Tuple[float, List[Dict]]:
        """Compute risk from shared side effects"""
        # Get side effects for each drug
        drug_se = {d: self.kg.drug_side_effects.get(d, set()) for d in drug_ids}
        
        # Find overlapping side effects
        shared_se = []
        se_overlap_count = 0
        total_comparisons = 0
        
        for d1, d2 in combinations(drug_ids, 2):
            common = drug_se[d1] & drug_se[d2]
            total_comparisons += 1
            
            if common:
                se_overlap_count += 1
                
                # Check for high-risk side effects
                high_risk = []
                for se_id in common:
                    se_name = self.kg.side_effect_names.get(se_id, se_id).lower()
                    for risk_term in self.HIGH_RISK_SIDE_EFFECTS:
                        if risk_term in se_name:
                            high_risk.append(se_name)
                            break
                
                d1_name = self.kg.drugs[d1].drug_name if d1 in self.kg.drugs else d1
                d2_name = self.kg.drugs[d2].drug_name if d2 in self.kg.drugs else d2
                
                shared_se.append({
                    'drug1': d1_name,
                    'drug2': d2_name,
                    'shared_count': len(common),
                    'high_risk_effects': high_risk[:5],  # Top 5
                    'sample_effects': [
                        self.kg.side_effect_names.get(se, se)
                        for se in list(common)[:5]
                    ]
                })
        
        # Risk = proportion of pairs with overlap, weighted by high-risk count
        if total_comparisons == 0:
            return 0.0, []
        
        base_risk = se_overlap_count / total_comparisons
        
        # Boost for high-risk side effects
        high_risk_boost = sum(len(se['high_risk_effects']) for se in shared_se)
        high_risk_factor = min(1.0, high_risk_boost / (len(drug_ids) * 2))
        
        risk_score = min(1.0, base_risk * 0.5 + high_risk_factor * 0.5)
        
        shared_se.sort(key=lambda x: len(x['high_risk_effects']), reverse=True)
        
        return risk_score, shared_se
    
    def _compute_protein_overlap_risk(self, drug_ids: List[str]) -> Tuple[float, List[Dict]]:
        """Compute risk from shared protein targets"""
        drug_proteins = {d: self.kg.drug_proteins.get(d, set()) for d in drug_ids}
        
        shared_proteins = []
        overlap_count = 0
        total_comparisons = 0
        
        for d1, d2 in combinations(drug_ids, 2):
            common = drug_proteins[d1] & drug_proteins[d2]
            total_comparisons += 1
            
            if common:
                overlap_count += 1
                
                d1_name = self.kg.drugs[d1].drug_name if d1 in self.kg.drugs else d1
                d2_name = self.kg.drugs[d2].drug_name if d2 in self.kg.drugs else d2
                
                shared_proteins.append({
                    'drug1': d1_name,
                    'drug2': d2_name,
                    'shared_count': len(common),
                    'proteins': [
                        self.kg.protein_names.get(p, p)
                        for p in list(common)[:5]
                    ]
                })
        
        if total_comparisons == 0:
            return 0.0, []
        
        # Protein overlap indicates mechanism similarity -> potential competition
        risk_score = overlap_count / total_comparisons
        
        shared_proteins.sort(key=lambda x: x['shared_count'], reverse=True)
        
        return risk_score, shared_proteins
    
    def _compute_pathway_risk(self, drug_ids: List[str]) -> Tuple[float, List[Dict]]:
        """Compute risk from shared metabolic pathways"""
        drug_pathways = {d: self.kg.drug_pathways.get(d, set()) for d in drug_ids}
        
        pathway_conflicts = []
        overlap_count = 0
        total_comparisons = 0
        
        for d1, d2 in combinations(drug_ids, 2):
            common = drug_pathways[d1] & drug_pathways[d2]
            total_comparisons += 1
            
            if common:
                overlap_count += 1
                
                d1_name = self.kg.drugs[d1].drug_name if d1 in self.kg.drugs else d1
                d2_name = self.kg.drugs[d2].drug_name if d2 in self.kg.drugs else d2
                
                pathway_conflicts.append({
                    'drug1': d1_name,
                    'drug2': d2_name,
                    'shared_count': len(common)
                })
        
        if total_comparisons == 0:
            return 0.0, []
        
        risk_score = overlap_count / total_comparisons
        
        return risk_score, pathway_conflicts
    
    def _compute_network_risk(self, drug_ids: List[str]) -> float:
        """Compute risk based on network centrality of drugs"""
        if not drug_ids:
            return 0.0
        
        # Average weighted degree of selected drugs
        weighted_degrees = []
        for drug_id in drug_ids:
            if drug_id in self.kg.drugs:
                weighted_degrees.append(self.kg.drugs[drug_id].weighted_degree)
        
        return np.mean(weighted_degrees) if weighted_degrees else 0.0
    
    def _generate_recommendations(self, drug_ids: List[str], ddi_pairs: List[Dict],
                                   shared_se: List[Dict], shared_proteins: List[Dict],
                                   risk_level: str) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.append(
                "⛔ CRITICAL RISK: Immediate review recommended. "
                "Consider discontinuing or replacing high-risk medications."
            )
        elif risk_level == 'HIGH':
            recommendations.append(
                "⚠️ HIGH RISK: Close monitoring required. "
                "Review medication necessity and consider alternatives."
            )
        
        # Specific DDI recommendations
        contraindicated = [p for p in ddi_pairs if 'Contraindicated' in p['severity']]
        major = [p for p in ddi_pairs if p['severity'] == 'Major interaction']
        
        if contraindicated:
            recommendations.append(
                f"🚫 {len(contraindicated)} CONTRAINDICATED interaction(s) detected. "
                "These combinations should generally be avoided."
            )
            for pair in contraindicated[:3]:
                recommendations.append(
                    f"   • {pair['drug1']} + {pair['drug2']}: {pair['description'][:100]}..."
                )
        
        if major:
            recommendations.append(
                f"⚠️ {len(major)} MAJOR interaction(s) require monitoring."
            )
        
        # Side effect warnings
        high_risk_se_pairs = [se for se in shared_se if se['high_risk_effects']]
        if high_risk_se_pairs:
            recommendations.append(
                f"💊 {len(high_risk_se_pairs)} drug pair(s) share serious side effect risks. "
                "Monitor for additive toxicity."
            )
            for se_pair in high_risk_se_pairs[:2]:
                effects = ', '.join(se_pair['high_risk_effects'][:3])
                recommendations.append(
                    f"   • {se_pair['drug1']} + {se_pair['drug2']}: {effects}"
                )
        
        # Protein overlap warnings
        if shared_proteins:
            recommendations.append(
                f"🧬 {len(shared_proteins)} drug pair(s) share protein targets. "
                "Potential mechanism-based interactions."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ No critical risks identified. Continue routine monitoring."
            )
        
        return recommendations


# ============================================================================
# RISK REPORT GENERATOR
# ============================================================================

def generate_risk_report(result: PolypharmacyRiskResult) -> str:
    """Generate human-readable risk report"""
    
    lines = [
        "=" * 70,
        "POLYPHARMACY RISK ASSESSMENT REPORT",
        "=" * 70,
        "",
        f"Drugs Analyzed: {', '.join(result.drugs)}",
        f"Number of Drugs: {len(result.drugs)}",
        "",
        "-" * 70,
        "OVERALL RISK ASSESSMENT",
        "-" * 70,
        f"  Overall Risk Score: {result.overall_risk_score:.2f} / 1.00",
        f"  Risk Level: {result.risk_level}",
        "",
        "  Component Scores:",
        f"    • DDI Severity:        {result.ddi_risk:.2f}",
        f"    • Side Effect Overlap: {result.side_effect_risk:.2f}",
        f"    • Protein Overlap:     {result.protein_overlap_risk:.2f}",
        f"    • Pathway Conflicts:   {result.pathway_risk:.2f}",
        f"    • Network Centrality:  {result.network_risk:.2f}",
        "",
    ]
    
    # DDI Details
    if result.ddi_pairs:
        lines.extend([
            "-" * 70,
            f"DRUG-DRUG INTERACTIONS ({len(result.ddi_pairs)} found)",
            "-" * 70,
        ])
        for pair in result.ddi_pairs[:10]:  # Top 10
            lines.append(f"  [{pair['severity']}]")
            lines.append(f"    {pair['drug1']} <-> {pair['drug2']}")
            if pair['description']:
                lines.append(f"    {pair['description'][:100]}...")
            lines.append("")
    
    # High-risk side effects
    high_risk_se = [se for se in result.shared_side_effects if se['high_risk_effects']]
    if high_risk_se:
        lines.extend([
            "-" * 70,
            "SHARED HIGH-RISK SIDE EFFECTS",
            "-" * 70,
        ])
        for se in high_risk_se[:5]:
            lines.append(f"  {se['drug1']} + {se['drug2']}:")
            lines.append(f"    Risks: {', '.join(se['high_risk_effects'])}")
    
    # Recommendations
    lines.extend([
        "",
        "-" * 70,
        "RECOMMENDATIONS",
        "-" * 70,
    ])
    for rec in result.recommendations:
        lines.append(f"  {rec}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """CLI for polypharmacy risk assessment"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Knowledge Graph-Based Polypharmacy Risk Assessment'
    )
    parser.add_argument(
        '--drugs', '-d',
        type=str,
        required=True,
        help='Comma-separated list of drug names (e.g., "warfarin,aspirin,metoprolol")'
    )
    parser.add_argument(
        '--kg-dir',
        type=str,
        default='knowledge_graph_fact_based/neo4j_export',
        help='Path to knowledge graph directory'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of text report'
    )
    
    args = parser.parse_args()
    
    # Parse drug list
    drugs = [d.strip() for d in args.drugs.split(',')]
    
    print(f"\n🔍 Assessing polypharmacy risk for: {drugs}\n")
    
    # Load knowledge graph
    config = KGConfig(kg_dir=Path(args.kg_dir))
    kg = KnowledgeGraphLoader(config).load()
    
    # Assess risk
    assessor = PolypharmacyRiskAssessor(kg, config)
    result = assessor.assess_polypharmacy_risk(drugs)
    
    if args.json:
        # JSON output
        output = {
            'drugs': result.drugs,
            'overall_risk_score': result.overall_risk_score,
            'risk_level': result.risk_level,
            'components': {
                'ddi_risk': result.ddi_risk,
                'side_effect_risk': result.side_effect_risk,
                'protein_overlap_risk': result.protein_overlap_risk,
                'pathway_risk': result.pathway_risk,
                'network_risk': result.network_risk
            },
            'ddi_pairs': result.ddi_pairs[:10],
            'high_risk_pairs': [(p[0], p[1]) for p in result.high_risk_pairs],
            'recommendations': result.recommendations
        }
        print(json.dumps(output, indent=2))
    else:
        # Text report
        report = generate_risk_report(result)
        print(report)


if __name__ == "__main__":
    main()
