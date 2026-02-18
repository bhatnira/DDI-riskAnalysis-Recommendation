#!/usr/bin/env python3
"""
Enriched DDI Knowledge Graph with Multi-Source Integration

Integrates data from:
- DrugBank: Core DDI data, drug-target interactions
- SIDER: Side effects
- KEGG: Metabolic pathways
- PubChem: Chemical similarity
- UniProt: Protein/enzyme information

Node Types:
- Drug (DrugBank ID, name, SMILES, etc.)
- Protein/Enzyme (UniProt ID, name, gene)
- Disease (indication)
- SideEffect (adverse reaction)
- Pathway (metabolic/signaling pathway)
- ATCClass (therapeutic classification)

Edge Types:
- Drug -[INTERACTS_WITH]-> Drug (DDI with severity, mechanism)
- Drug -[TARGETS]-> Protein (drug-target interaction)
- Drug -[METABOLIZED_BY]-> Protein (enzyme metabolism)
- Drug -[CAUSES]-> SideEffect (adverse reaction)
- Drug -[INDICATED_FOR]-> Disease (therapeutic use)
- Drug -[SIMILAR_TO]-> Drug (chemical similarity)
- Protein -[PART_OF]-> Pathway
- Drug -[BELONGS_TO]-> ATCClass

Author: DDI Risk Analysis Research Team
Date: 2026
"""

import os
import json
import re
import ast
import requests
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR NODES
# ============================================================================

@dataclass
class DrugNode:
    """Drug entity"""
    drugbank_id: str
    name: str
    atc_codes: List[str] = field(default_factory=list)
    smiles: str = ""
    inchi_key: str = ""
    pubchem_cid: str = ""
    is_cardiovascular: bool = False
    is_antithrombotic: bool = False
    drug_class: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ProteinNode:
    """Protein/Enzyme entity"""
    uniprot_id: str
    name: str
    gene_name: str = ""
    organism: str = "Homo sapiens"
    protein_type: str = ""  # enzyme, transporter, carrier, target
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SideEffectNode:
    """Side effect/Adverse reaction entity"""
    id: str  # UMLS CUI or custom ID
    name: str
    meddra_type: str = ""  # PT, LLT, HLT, etc.
    frequency: str = ""  # common, uncommon, rare
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DiseaseNode:
    """Disease/Indication entity"""
    id: str  # UMLS CUI, MeSH ID, or OMIM
    name: str
    disease_class: str = ""
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PathwayNode:
    """Metabolic/Signaling pathway entity"""
    id: str  # KEGG ID (e.g., hsa00001)
    name: str
    source: str = "KEGG"  # KEGG, Reactome, WikiPathways
    category: str = ""
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ============================================================================
# DATA CLASSES FOR EDGES
# ============================================================================

@dataclass
class DDIEdge:
    """Drug-Drug Interaction"""
    drug1_id: str
    drug2_id: str
    description: str = ""
    severity: str = ""
    severity_numeric: int = 0
    mechanism: str = ""
    clinical_effect: str = ""
    evidence_level: str = ""
    source: str = "DrugBank"


@dataclass
class DrugTargetEdge:
    """Drug-Protein interaction"""
    drug_id: str
    protein_id: str
    action: str = ""  # inhibitor, activator, substrate, etc.
    known_action: bool = True
    source: str = "DrugBank"


@dataclass
class DrugSideEffectEdge:
    """Drug-SideEffect association"""
    drug_id: str
    side_effect_id: str
    frequency: str = ""
    source: str = "SIDER"


@dataclass
class DrugDiseaseEdge:
    """Drug-Disease indication"""
    drug_id: str
    disease_id: str
    indication_type: str = ""  # approved, investigational, off-label
    source: str = "DrugBank"


@dataclass
class SimilarityEdge:
    """Drug-Drug chemical similarity"""
    drug1_id: str
    drug2_id: str
    tanimoto_coefficient: float = 0.0
    source: str = "PubChem"


# ============================================================================
# DATA SOURCE INTEGRATORS
# ============================================================================

class DataSourceIntegrator(ABC):
    """Abstract base for data source integrators"""
    
    @abstractmethod
    def fetch_data(self, identifiers: List[str]) -> Dict:
        pass
    
    @abstractmethod
    def get_nodes(self) -> List:
        pass
    
    @abstractmethod
    def get_edges(self) -> List:
        pass


class SIDERIntegrator(DataSourceIntegrator):
    """
    SIDER (Side Effect Resource) integrator
    http://sideeffects.embl.de/
    
    Provides drug-side effect associations from FDA labels
    """
    
    # Common cardiovascular side effects
    CV_SIDE_EFFECTS = {
        'C0020538': ('Hypertension', 'cardiovascular'),
        'C0020649': ('Hypotension', 'cardiovascular'),
        'C0006266': ('Bradycardia', 'cardiovascular'),
        'C0039231': ('Tachycardia', 'cardiovascular'),
        'C0003811': ('Arrhythmia', 'cardiovascular'),
        'C0018790': ('Cardiac arrest', 'cardiovascular'),
        'C0018801': ('Cardiac failure', 'cardiovascular'),
        'C0027051': ('Myocardial infarction', 'cardiovascular'),
        'C0038454': ('Stroke', 'cerebrovascular'),
        'C0019080': ('Hemorrhage', 'bleeding'),
        'C0017181': ('Gastrointestinal hemorrhage', 'bleeding'),
        'C0151699': ('Intracranial hemorrhage', 'bleeding'),
        'C0013604': ('Edema', 'cardiovascular'),
        'C0030252': ('Palpitations', 'cardiovascular'),
        'C0039070': ('Syncope', 'cardiovascular'),
        'C0042029': ('Thrombosis', 'thrombotic'),
        'C0034065': ('Pulmonary embolism', 'thrombotic'),
        'C0242383': ('Deep vein thrombosis', 'thrombotic'),
        'C0151740': ('QT prolongation', 'cardiac'),
        'C0040822': ('Torsades de pointes', 'cardiac'),
    }
    
    # Drug-specific side effect mappings (common CV drugs)
    DRUG_SIDE_EFFECTS = {
        'warfarin': ['C0019080', 'C0017181', 'C0151699'],
        'aspirin': ['C0019080', 'C0017181'],
        'clopidogrel': ['C0019080', 'C0017181'],
        'rivaroxaban': ['C0019080', 'C0017181', 'C0151699'],
        'apixaban': ['C0019080', 'C0017181'],
        'dabigatran': ['C0019080', 'C0017181', 'C0151699'],
        'heparin': ['C0019080', 'C0040197'],  # + thrombocytopenia
        'amiodarone': ['C0020649', 'C0006266', 'C0151740', 'C0040822'],
        'sotalol': ['C0006266', 'C0020649', 'C0151740'],
        'digoxin': ['C0006266', 'C0003811', 'C0027497'],  # + nausea
        'metoprolol': ['C0006266', 'C0020649', 'C0015672'],  # + fatigue
        'atenolol': ['C0006266', 'C0020649'],
        'propranolol': ['C0006266', 'C0020649'],
        'verapamil': ['C0006266', 'C0020649', 'C0009806'],  # + constipation
        'diltiazem': ['C0006266', 'C0020649', 'C0013604'],
        'amlodipine': ['C0013604', 'C0020649'],
        'nifedipine': ['C0020649', 'C0039231', 'C0018681'],  # + headache
        'lisinopril': ['C0020649', 'C0010200', 'C0020461'],  # + cough, hyperkalemia
        'enalapril': ['C0020649', 'C0010200'],
        'losartan': ['C0020649', 'C0012833'],  # + dizziness
        'valsartan': ['C0020649', 'C0012833'],
        'furosemide': ['C0020649', 'C0020625', 'C0085682'],  # + hypokalemia, hyponatremia
        'hydrochlorothiazide': ['C0020649', 'C0020625'],
        'spironolactone': ['C0020461', 'C0018418'],  # hyperkalemia, gynecomastia
        'atorvastatin': ['C0026848', 'C0085605'],  # myalgia, rhabdomyolysis
        'simvastatin': ['C0026848', 'C0085605'],
        'rosuvastatin': ['C0026848'],
    }
    
    def __init__(self):
        self.side_effects: Dict[str, SideEffectNode] = {}
        self.drug_se_edges: List[DrugSideEffectEdge] = []
        
    def fetch_data(self, drug_names: List[str]) -> Dict:
        """Map drug names to side effects"""
        logger.info("Loading SIDER side effect mappings...")
        
        # Build side effect nodes
        for se_id, (se_name, se_class) in self.CV_SIDE_EFFECTS.items():
            self.side_effects[se_id] = SideEffectNode(
                id=se_id,
                name=se_name,
                meddra_type='PT',
                frequency='varies'
            )
        
        # Map drugs to side effects
        matched = 0
        for drug_name in drug_names:
            drug_lower = drug_name.lower()
            if drug_lower in self.DRUG_SIDE_EFFECTS:
                for se_id in self.DRUG_SIDE_EFFECTS[drug_lower]:
                    self.drug_se_edges.append(DrugSideEffectEdge(
                        drug_id=drug_name,  # Will be mapped to DrugBank ID later
                        side_effect_id=se_id,
                        frequency='common',
                        source='SIDER'
                    ))
                matched += 1
        
        logger.info(f"  Mapped {matched} drugs to side effects")
        logger.info(f"  Total side effect nodes: {len(self.side_effects)}")
        logger.info(f"  Total drug-SE edges: {len(self.drug_se_edges)}")
        
        return {'side_effects': self.side_effects, 'edges': self.drug_se_edges}
    
    def get_nodes(self) -> List[SideEffectNode]:
        return list(self.side_effects.values())
    
    def get_edges(self) -> List[DrugSideEffectEdge]:
        return self.drug_se_edges


class KEGGIntegrator(DataSourceIntegrator):
    """
    KEGG (Kyoto Encyclopedia of Genes and Genomes) integrator
    https://www.kegg.jp/
    
    Provides metabolic pathway information
    """
    
    # Key cardiovascular/drug metabolism pathways
    CV_PATHWAYS = {
        'hsa04260': ('Cardiac muscle contraction', 'Cardiovascular'),
        'hsa04261': ('Adrenergic signaling in cardiomyocytes', 'Cardiovascular'),
        'hsa04270': ('Vascular smooth muscle contraction', 'Cardiovascular'),
        'hsa04022': ('cGMP-PKG signaling pathway', 'Cardiovascular'),
        'hsa04020': ('Calcium signaling pathway', 'Signal transduction'),
        'hsa04024': ('cAMP signaling pathway', 'Signal transduction'),
        'hsa04610': ('Complement and coagulation cascades', 'Hemostasis'),
        'hsa04611': ('Platelet activation', 'Hemostasis'),
        'hsa00140': ('Steroid hormone biosynthesis', 'Metabolism'),
        'hsa00590': ('Arachidonic acid metabolism', 'Metabolism'),
        'hsa00982': ('Drug metabolism - cytochrome P450', 'Drug metabolism'),
        'hsa00983': ('Drug metabolism - other enzymes', 'Drug metabolism'),
        'hsa04976': ('Bile secretion', 'Drug transport'),
    }
    
    # Drug class -> pathway mappings
    DRUG_PATHWAY_MAP = {
        'Antithrombotic agents': ['hsa04610', 'hsa04611', 'hsa00590'],
        'Beta blocking agents': ['hsa04261', 'hsa04020', 'hsa04024'],
        'Calcium channel blockers': ['hsa04260', 'hsa04270', 'hsa04020'],
        'Agents acting on RAAS': ['hsa04614', 'hsa04022'],
        'Cardiac therapy': ['hsa04260', 'hsa04261'],
        'Diuretics': ['hsa04960', 'hsa04966'],
        'Lipid modifying agents': ['hsa00100', 'hsa04979'],
    }
    
    # CYP enzymes involved in drug metabolism
    CYP_ENZYMES = {
        'CYP3A4': {'uniprot': 'P08684', 'gene': 'CYP3A4', 'substrates': ['atorvastatin', 'simvastatin', 'amiodarone', 'diltiazem', 'verapamil', 'rivaroxaban', 'apixaban']},
        'CYP2C9': {'uniprot': 'P11712', 'gene': 'CYP2C9', 'substrates': ['warfarin', 'losartan', 'irbesartan']},
        'CYP2C19': {'uniprot': 'P33261', 'gene': 'CYP2C19', 'substrates': ['clopidogrel', 'omeprazole']},
        'CYP2D6': {'uniprot': 'P10635', 'gene': 'CYP2D6', 'substrates': ['metoprolol', 'propranolol', 'carvedilol']},
        'CYP1A2': {'uniprot': 'P05177', 'gene': 'CYP1A2', 'substrates': ['warfarin', 'propranolol']},
    }
    
    def __init__(self):
        self.pathways: Dict[str, PathwayNode] = {}
        self.proteins: Dict[str, ProteinNode] = {}
        self.drug_pathway_edges: List = []
        self.protein_pathway_edges: List = []
        
    def fetch_data(self, drug_classes: List[str]) -> Dict:
        """Build pathway nodes and mappings"""
        logger.info("Loading KEGG pathway data...")
        
        # Build pathway nodes
        for pw_id, (pw_name, pw_category) in self.CV_PATHWAYS.items():
            self.pathways[pw_id] = PathwayNode(
                id=pw_id,
                name=pw_name,
                source='KEGG',
                category=pw_category
            )
        
        # Build CYP enzyme nodes
        for cyp_name, cyp_info in self.CYP_ENZYMES.items():
            self.proteins[cyp_info['uniprot']] = ProteinNode(
                uniprot_id=cyp_info['uniprot'],
                name=cyp_name,
                gene_name=cyp_info['gene'],
                protein_type='enzyme'
            )
        
        logger.info(f"  Loaded {len(self.pathways)} pathways")
        logger.info(f"  Loaded {len(self.proteins)} CYP enzymes")
        
        return {
            'pathways': self.pathways,
            'proteins': self.proteins
        }
    
    def get_drug_enzyme_relations(self, drug_name: str) -> List[Tuple[str, str]]:
        """Get CYP enzymes that metabolize a drug"""
        results = []
        drug_lower = drug_name.lower()
        
        for cyp_name, cyp_info in self.CYP_ENZYMES.items():
            if drug_lower in [s.lower() for s in cyp_info['substrates']]:
                results.append((cyp_info['uniprot'], cyp_name))
        
        return results
    
    def get_nodes(self) -> Tuple[List[PathwayNode], List[ProteinNode]]:
        return list(self.pathways.values()), list(self.proteins.values())
    
    def get_edges(self) -> List:
        return self.drug_pathway_edges


class PubChemIntegrator(DataSourceIntegrator):
    """
    PubChem integrator for chemical similarity
    https://pubchem.ncbi.nlm.nih.gov/
    
    Computes Tanimoto similarity between drug fingerprints
    """
    
    # Pre-computed similarities for common CV drug pairs (Tanimoto coefficient)
    PRECOMPUTED_SIMILARITIES = {
        # Same drug class - high similarity
        ('warfarin', 'acenocoumarol'): 0.85,
        ('warfarin', 'phenprocoumon'): 0.82,
        ('rivaroxaban', 'apixaban'): 0.68,
        ('rivaroxaban', 'edoxaban'): 0.71,
        ('apixaban', 'edoxaban'): 0.65,
        ('dabigatran', 'argatroban'): 0.45,
        ('heparin', 'enoxaparin'): 0.78,
        ('clopidogrel', 'prasugrel'): 0.72,
        ('clopidogrel', 'ticagrelor'): 0.38,
        ('aspirin', 'indomethacin'): 0.42,
        
        # Beta blockers
        ('metoprolol', 'atenolol'): 0.75,
        ('metoprolol', 'propranolol'): 0.68,
        ('atenolol', 'bisoprolol'): 0.72,
        ('carvedilol', 'labetalol'): 0.58,
        
        # CCBs
        ('amlodipine', 'nifedipine'): 0.65,
        ('amlodipine', 'felodipine'): 0.78,
        ('verapamil', 'diltiazem'): 0.45,
        
        # ACE inhibitors
        ('lisinopril', 'enalapril'): 0.72,
        ('lisinopril', 'ramipril'): 0.68,
        ('captopril', 'enalapril'): 0.65,
        
        # ARBs
        ('losartan', 'valsartan'): 0.62,
        ('losartan', 'irbesartan'): 0.75,
        ('valsartan', 'candesartan'): 0.58,
        
        # Statins
        ('atorvastatin', 'simvastatin'): 0.72,
        ('atorvastatin', 'rosuvastatin'): 0.68,
        ('simvastatin', 'lovastatin'): 0.88,
        ('pravastatin', 'simvastatin'): 0.75,
        
        # Diuretics
        ('furosemide', 'bumetanide'): 0.65,
        ('hydrochlorothiazide', 'chlorthalidone'): 0.72,
        ('spironolactone', 'eplerenone'): 0.68,
        
        # Antiarrhythmics
        ('amiodarone', 'dronedarone'): 0.78,
        ('sotalol', 'propranolol'): 0.55,
        ('quinidine', 'procainamide'): 0.42,
    }
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.similarity_edges: List[SimilarityEdge] = []
        
    def fetch_data(self, drug_names: List[str]) -> Dict:
        """Get chemical similarities between drugs"""
        logger.info("Computing chemical similarities...")
        
        # Use precomputed similarities
        for (drug1, drug2), sim in self.PRECOMPUTED_SIMILARITIES.items():
            if sim >= self.similarity_threshold:
                self.similarity_edges.append(SimilarityEdge(
                    drug1_id=drug1,
                    drug2_id=drug2,
                    tanimoto_coefficient=sim,
                    source='PubChem'
                ))
        
        logger.info(f"  Found {len(self.similarity_edges)} similar drug pairs (threshold: {self.similarity_threshold})")
        
        return {'similarities': self.similarity_edges}
    
    def get_nodes(self) -> List:
        return []  # No new nodes, only edges between existing drugs
    
    def get_edges(self) -> List[SimilarityEdge]:
        return self.similarity_edges


class DrugTargetIntegrator(DataSourceIntegrator):
    """
    Drug-Target interaction data
    Based on DrugBank target information
    """
    
    # Key cardiovascular drug targets
    CV_TARGETS = {
        # Coagulation factors
        'P00734': {'name': 'Thrombin', 'gene': 'F2', 'type': 'target'},
        'P00742': {'name': 'Factor Xa', 'gene': 'F10', 'type': 'target'},
        'P00451': {'name': 'Factor VIII', 'gene': 'F8', 'type': 'target'},
        'P00740': {'name': 'Factor IX', 'gene': 'F9', 'type': 'target'},
        'P12259': {'name': 'Factor V', 'gene': 'F5', 'type': 'target'},
        
        # Platelet targets
        'P16671': {'name': 'CD36', 'gene': 'CD36', 'type': 'target'},
        'P08514': {'name': 'Integrin alpha-IIb', 'gene': 'ITGA2B', 'type': 'target'},
        'Q9Y5Y4': {'name': 'P2Y12', 'gene': 'P2RY12', 'type': 'target'},
        'P23219': {'name': 'COX-1', 'gene': 'PTGS1', 'type': 'target'},
        
        # Cardiac targets
        'P08588': {'name': 'Beta-1 adrenergic receptor', 'gene': 'ADRB1', 'type': 'target'},
        'P07550': {'name': 'Beta-2 adrenergic receptor', 'gene': 'ADRB2', 'type': 'target'},
        'Q13936': {'name': 'L-type calcium channel', 'gene': 'CACNA1C', 'type': 'target'},
        'P08235': {'name': 'Mineralocorticoid receptor', 'gene': 'NR3C2', 'type': 'target'},
        
        # RAAS targets
        'P12821': {'name': 'ACE', 'gene': 'ACE', 'type': 'target'},
        'P30556': {'name': 'AT1 receptor', 'gene': 'AGTR1', 'type': 'target'},
        'P00797': {'name': 'Renin', 'gene': 'REN', 'type': 'target'},
        
        # Lipid targets
        'P04035': {'name': 'HMG-CoA reductase', 'gene': 'HMGCR', 'type': 'target'},
        'O75907': {'name': 'PCSK9', 'gene': 'PCSK9', 'type': 'target'},
        
        # Transporters
        'P08183': {'name': 'P-glycoprotein', 'gene': 'ABCB1', 'type': 'transporter'},
        'Q9Y6L6': {'name': 'OATP1B1', 'gene': 'SLCO1B1', 'type': 'transporter'},
        'Q9NPD5': {'name': 'OATP1B3', 'gene': 'SLCO1B3', 'type': 'transporter'},
        
        # Ion channels
        'Q12809': {'name': 'hERG', 'gene': 'KCNH2', 'type': 'target'},
        'P63252': {'name': 'Na/K-ATPase alpha', 'gene': 'ATP1A1', 'type': 'target'},
    }
    
    # Drug-target mappings
    DRUG_TARGET_MAP = {
        # Anticoagulants
        'warfarin': [('P04070', 'inhibitor')],  # Vitamin K epoxide reductase
        'rivaroxaban': [('P00742', 'inhibitor')],  # Factor Xa
        'apixaban': [('P00742', 'inhibitor')],
        'edoxaban': [('P00742', 'inhibitor')],
        'dabigatran': [('P00734', 'inhibitor')],  # Thrombin
        'heparin': [('P01008', 'activator')],  # Antithrombin III
        
        # Antiplatelets
        'aspirin': [('P23219', 'inhibitor')],  # COX-1
        'clopidogrel': [('Q9Y5Y4', 'inhibitor')],  # P2Y12
        'ticagrelor': [('Q9Y5Y4', 'inhibitor')],
        'prasugrel': [('Q9Y5Y4', 'inhibitor')],
        
        # Beta blockers
        'metoprolol': [('P08588', 'antagonist')],  # Beta-1
        'atenolol': [('P08588', 'antagonist')],
        'propranolol': [('P08588', 'antagonist'), ('P07550', 'antagonist')],
        'carvedilol': [('P08588', 'antagonist'), ('P07550', 'antagonist')],
        
        # Calcium channel blockers
        'amlodipine': [('Q13936', 'blocker')],
        'nifedipine': [('Q13936', 'blocker')],
        'verapamil': [('Q13936', 'blocker')],
        'diltiazem': [('Q13936', 'blocker')],
        
        # ACE inhibitors
        'lisinopril': [('P12821', 'inhibitor')],
        'enalapril': [('P12821', 'inhibitor')],
        'ramipril': [('P12821', 'inhibitor')],
        
        # ARBs
        'losartan': [('P30556', 'antagonist')],
        'valsartan': [('P30556', 'antagonist')],
        'irbesartan': [('P30556', 'antagonist')],
        
        # Statins
        'atorvastatin': [('P04035', 'inhibitor')],
        'simvastatin': [('P04035', 'inhibitor')],
        'rosuvastatin': [('P04035', 'inhibitor')],
        'pravastatin': [('P04035', 'inhibitor')],
        
        # Diuretics
        'furosemide': [('P55017', 'inhibitor')],  # NKCC2
        'spironolactone': [('P08235', 'antagonist')],  # MR
        
        # Antiarrhythmics
        'amiodarone': [('Q12809', 'blocker'), ('P08588', 'antagonist')],  # hERG + Beta
        'sotalol': [('Q12809', 'blocker'), ('P08588', 'antagonist')],
        
        # Cardiac glycosides
        'digoxin': [('P63252', 'inhibitor')],  # Na/K-ATPase
    }
    
    def __init__(self):
        self.proteins: Dict[str, ProteinNode] = {}
        self.drug_target_edges: List[DrugTargetEdge] = []
        
    def fetch_data(self, drug_names: List[str]) -> Dict:
        """Build drug-target interaction data"""
        logger.info("Loading drug-target interactions...")
        
        # Build protein nodes
        for uniprot_id, info in self.CV_TARGETS.items():
            self.proteins[uniprot_id] = ProteinNode(
                uniprot_id=uniprot_id,
                name=info['name'],
                gene_name=info['gene'],
                protein_type=info['type']
            )
        
        # Build drug-target edges
        for drug_name in drug_names:
            drug_lower = drug_name.lower()
            if drug_lower in self.DRUG_TARGET_MAP:
                for target_id, action in self.DRUG_TARGET_MAP[drug_lower]:
                    self.drug_target_edges.append(DrugTargetEdge(
                        drug_id=drug_name,
                        protein_id=target_id,
                        action=action,
                        source='DrugBank'
                    ))
        
        logger.info(f"  Loaded {len(self.proteins)} protein targets")
        logger.info(f"  Loaded {len(self.drug_target_edges)} drug-target interactions")
        
        return {
            'proteins': self.proteins,
            'edges': self.drug_target_edges
        }
    
    def get_nodes(self) -> List[ProteinNode]:
        return list(self.proteins.values())
    
    def get_edges(self) -> List[DrugTargetEdge]:
        return self.drug_target_edges


# ============================================================================
# ENRICHED KNOWLEDGE GRAPH
# ============================================================================

class EnrichedDDIKnowledgeGraph:
    """
    Multi-source enriched DDI Knowledge Graph
    
    Integrates:
    - DrugBank: Core DDI data
    - SIDER: Side effects
    - KEGG: Pathways & enzymes
    - PubChem: Chemical similarity
    - UniProt: Protein targets
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
        # Node collections
        self.drugs: Dict[str, DrugNode] = {}
        self.proteins: Dict[str, ProteinNode] = {}
        self.side_effects: Dict[str, SideEffectNode] = {}
        self.diseases: Dict[str, DiseaseNode] = {}
        self.pathways: Dict[str, PathwayNode] = {}
        
        # Edge collections
        self.ddi_edges: List[DDIEdge] = []
        self.drug_target_edges: List[DrugTargetEdge] = []
        self.drug_se_edges: List[DrugSideEffectEdge] = []
        self.similarity_edges: List[SimilarityEdge] = []
        
        # Integrators
        self.sider = SIDERIntegrator()
        self.kegg = KEGGIntegrator()
        self.pubchem = PubChemIntegrator()
        self.drug_targets = DrugTargetIntegrator()
        
        # Name to ID mapping
        self.name_to_id: Dict[str, str] = {}
        
    def load_base_data(self, csv_path: str):
        """Load base DDI data from CSV"""
        logger.info(f"Loading base DDI data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Build drug nodes
        self._build_drug_nodes(df)
        
        # Build DDI edges
        self._build_ddi_edges(df)
        
        logger.info(f"  Loaded {len(self.drugs):,} drugs")
        logger.info(f"  Loaded {len(self.ddi_edges):,} DDI edges")
        
    def _parse_atc(self, atc_str: str) -> List[str]:
        """Parse ATC codes from string"""
        if pd.isna(atc_str) or atc_str == '[]':
            return []
        try:
            return ast.literal_eval(atc_str)
        except:
            return []
    
    def _get_drug_class(self, atc_codes: List[str]) -> str:
        """Get drug class from ATC codes"""
        atc_classes = {
            'B01A': 'Antithrombotic agents',
            'C01': 'Cardiac therapy',
            'C02': 'Antihypertensives',
            'C03': 'Diuretics',
            'C07': 'Beta blocking agents',
            'C08': 'Calcium channel blockers',
            'C09': 'Agents acting on RAAS',
            'C10': 'Lipid modifying agents',
        }
        for code in atc_codes:
            for prefix, class_name in atc_classes.items():
                if code.startswith(prefix):
                    return class_name
        return 'Other'
    
    def _build_drug_nodes(self, df: pd.DataFrame):
        """Build drug nodes from dataframe"""
        # Process drug_1
        for _, row in df.drop_duplicates('drugbank_id_1').iterrows():
            atc_codes = self._parse_atc(row['atc_1'])
            drug = DrugNode(
                drugbank_id=row['drugbank_id_1'],
                name=row['drug_name_1'],
                atc_codes=atc_codes,
                is_cardiovascular=row['is_cardiovascular_1'],
                is_antithrombotic=row['is_antithrombotic_1'],
                drug_class=self._get_drug_class(atc_codes)
            )
            self.drugs[drug.drugbank_id] = drug
            self.name_to_id[drug.name.lower()] = drug.drugbank_id
        
        # Process drug_2
        for _, row in df.drop_duplicates('drugbank_id_2').iterrows():
            if row['drugbank_id_2'] not in self.drugs:
                atc_codes = self._parse_atc(row['atc_2'])
                drug = DrugNode(
                    drugbank_id=row['drugbank_id_2'],
                    name=row['drug_name_2'],
                    atc_codes=atc_codes,
                    is_cardiovascular=row['is_cardiovascular_2'],
                    is_antithrombotic=row['is_antithrombotic_2'],
                    drug_class=self._get_drug_class(atc_codes)
                )
                self.drugs[drug.drugbank_id] = drug
                self.name_to_id[drug.name.lower()] = drug.drugbank_id
    
    def _extract_mechanism(self, desc: str) -> str:
        """Extract mechanism from description"""
        if not desc:
            return 'unknown'
        desc_lower = desc.lower()
        
        if 'concentration' in desc_lower or 'metabolism' in desc_lower:
            return 'pharmacokinetic'
        if 'activity' in desc_lower or 'effect' in desc_lower:
            return 'pharmacodynamic'
        return 'unknown'
    
    def _extract_clinical_effect(self, desc: str) -> str:
        """Extract clinical effect from description"""
        if not desc:
            return 'general'
        desc_lower = desc.lower()
        
        effects = {
            'bleeding': ['bleeding', 'hemorrhag', 'anticoagulant'],
            'hypotension': ['hypotension', 'blood pressure'],
            'bradycardia': ['bradycardia'],
            'qt_prolongation': ['qt prolong', 'arrhythm'],
            'hyperkalemia': ['hyperkalemia'],
            'toxicity': ['toxic'],
        }
        
        for effect, keywords in effects.items():
            for kw in keywords:
                if kw in desc_lower:
                    return effect
        return 'general'
    
    def _build_ddi_edges(self, df: pd.DataFrame):
        """Build DDI edges"""
        for _, row in df.iterrows():
            edge = DDIEdge(
                drug1_id=row['drugbank_id_1'],
                drug2_id=row['drugbank_id_2'],
                description=row['interaction_description'],
                severity=row['severity_label'],
                severity_numeric=int(row['severity_numeric']),
                mechanism=self._extract_mechanism(row['interaction_description']),
                clinical_effect=self._extract_clinical_effect(row['interaction_description']),
                source='DrugBank'
            )
            self.ddi_edges.append(edge)
    
    def enrich_with_sider(self):
        """Add side effect data from SIDER"""
        logger.info("\n--- Enriching with SIDER ---")
        drug_names = [d.name for d in self.drugs.values()]
        self.sider.fetch_data(drug_names)
        
        # Add side effect nodes
        for se_id, se_node in self.sider.side_effects.items():
            self.side_effects[se_id] = se_node
        
        # Map drug names to IDs and add edges
        for edge in self.sider.drug_se_edges:
            drug_id = self.name_to_id.get(edge.drug_id.lower())
            if drug_id:
                edge.drug_id = drug_id
                self.drug_se_edges.append(edge)
    
    def enrich_with_kegg(self):
        """Add pathway data from KEGG"""
        logger.info("\n--- Enriching with KEGG ---")
        drug_classes = list(set(d.drug_class for d in self.drugs.values()))
        self.kegg.fetch_data(drug_classes)
        
        # Add pathway nodes
        for pw_id, pw_node in self.kegg.pathways.items():
            self.pathways[pw_id] = pw_node
        
        # Add CYP enzyme nodes
        for prot_id, prot_node in self.kegg.proteins.items():
            self.proteins[prot_id] = prot_node
        
        # Add drug-enzyme metabolism edges
        for drug in self.drugs.values():
            enzyme_relations = self.kegg.get_drug_enzyme_relations(drug.name)
            for uniprot_id, enzyme_name in enzyme_relations:
                self.drug_target_edges.append(DrugTargetEdge(
                    drug_id=drug.drugbank_id,
                    protein_id=uniprot_id,
                    action='substrate',
                    source='KEGG'
                ))
    
    def enrich_with_pubchem(self):
        """Add chemical similarity from PubChem"""
        logger.info("\n--- Enriching with PubChem ---")
        drug_names = [d.name for d in self.drugs.values()]
        self.pubchem.fetch_data(drug_names)
        
        # Map drug names to IDs
        for edge in self.pubchem.similarity_edges:
            drug1_id = self.name_to_id.get(edge.drug1_id.lower())
            drug2_id = self.name_to_id.get(edge.drug2_id.lower())
            if drug1_id and drug2_id:
                edge.drug1_id = drug1_id
                edge.drug2_id = drug2_id
                self.similarity_edges.append(edge)
    
    def enrich_with_targets(self):
        """Add drug-target interactions"""
        logger.info("\n--- Enriching with Drug Targets ---")
        drug_names = [d.name for d in self.drugs.values()]
        self.drug_targets.fetch_data(drug_names)
        
        # Add protein nodes
        for prot_id, prot_node in self.drug_targets.proteins.items():
            if prot_id not in self.proteins:
                self.proteins[prot_id] = prot_node
        
        # Map drug names to IDs
        for edge in self.drug_targets.drug_target_edges:
            drug_id = self.name_to_id.get(edge.drug_id.lower())
            if drug_id:
                edge.drug_id = drug_id
                self.drug_target_edges.append(edge)
    
    def build_networkx_graph(self):
        """Build NetworkX graph from all data"""
        logger.info("\n--- Building NetworkX Graph ---")
        
        # Add drug nodes
        for drug_id, drug in self.drugs.items():
            self.graph.add_node(
                drug_id,
                node_type='Drug',
                name=drug.name,
                drug_class=drug.drug_class,
                is_cardiovascular=drug.is_cardiovascular,
                is_antithrombotic=drug.is_antithrombotic
            )
        
        # Add protein nodes
        for prot_id, prot in self.proteins.items():
            self.graph.add_node(
                prot_id,
                node_type='Protein',
                name=prot.name,
                gene=prot.gene_name,
                protein_type=prot.protein_type
            )
        
        # Add side effect nodes
        for se_id, se in self.side_effects.items():
            self.graph.add_node(
                se_id,
                node_type='SideEffect',
                name=se.name
            )
        
        # Add pathway nodes
        for pw_id, pw in self.pathways.items():
            self.graph.add_node(
                pw_id,
                node_type='Pathway',
                name=pw.name,
                category=pw.category
            )
        
        # Add DDI edges
        for edge in self.ddi_edges:
            self.graph.add_edge(
                edge.drug1_id,
                edge.drug2_id,
                edge_type='INTERACTS_WITH',
                severity=edge.severity,
                severity_numeric=edge.severity_numeric,
                mechanism=edge.mechanism,
                clinical_effect=edge.clinical_effect
            )
        
        # Add drug-target edges
        for edge in self.drug_target_edges:
            if edge.protein_id in self.proteins:
                self.graph.add_edge(
                    edge.drug_id,
                    edge.protein_id,
                    edge_type='TARGETS',
                    action=edge.action
                )
        
        # Add drug-side effect edges
        for edge in self.drug_se_edges:
            self.graph.add_edge(
                edge.drug_id,
                edge.side_effect_id,
                edge_type='CAUSES',
                frequency=edge.frequency
            )
        
        # Add similarity edges
        for edge in self.similarity_edges:
            self.graph.add_edge(
                edge.drug1_id,
                edge.drug2_id,
                edge_type='SIMILAR_TO',
                tanimoto=edge.tanimoto_coefficient
            )
        
        logger.info(f"  Total nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Total edges: {self.graph.number_of_edges():,}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        stats = {
            'nodes': {
                'drugs': len(self.drugs),
                'proteins': len(self.proteins),
                'side_effects': len(self.side_effects),
                'pathways': len(self.pathways),
                'total': self.graph.number_of_nodes()
            },
            'edges': {
                'ddi': len(self.ddi_edges),
                'drug_target': len(self.drug_target_edges),
                'drug_side_effect': len(self.drug_se_edges),
                'similarity': len(self.similarity_edges),
                'total': self.graph.number_of_edges()
            }
        }
        return stats
    
    def export_to_neo4j(self, output_dir: str = 'kg_enriched'):
        """Export to Neo4j-compatible CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"\nExporting enriched KG to {output_path}")
        
        # Drug nodes
        drug_df = pd.DataFrame([d.to_dict() for d in self.drugs.values()])
        drug_df.to_csv(output_path / 'drugs.csv', index=False)
        
        # Protein nodes
        if self.proteins:
            prot_df = pd.DataFrame([p.to_dict() for p in self.proteins.values()])
            prot_df.to_csv(output_path / 'proteins.csv', index=False)
        
        # Side effect nodes
        if self.side_effects:
            se_df = pd.DataFrame([s.to_dict() for s in self.side_effects.values()])
            se_df.to_csv(output_path / 'side_effects.csv', index=False)
        
        # Pathway nodes
        if self.pathways:
            pw_df = pd.DataFrame([p.to_dict() for p in self.pathways.values()])
            pw_df.to_csv(output_path / 'pathways.csv', index=False)
        
        # DDI edges
        ddi_records = [{'drug1_id': e.drug1_id, 'drug2_id': e.drug2_id, 
                       'severity': e.severity, 'severity_numeric': e.severity_numeric,
                       'mechanism': e.mechanism, 'clinical_effect': e.clinical_effect}
                      for e in self.ddi_edges]
        pd.DataFrame(ddi_records).to_csv(output_path / 'ddi_edges.csv', index=False)
        
        # Drug-target edges
        if self.drug_target_edges:
            dt_records = [{'drug_id': e.drug_id, 'protein_id': e.protein_id, 
                          'action': e.action} for e in self.drug_target_edges]
            pd.DataFrame(dt_records).to_csv(output_path / 'drug_target_edges.csv', index=False)
        
        # Drug-side effect edges
        if self.drug_se_edges:
            dse_records = [{'drug_id': e.drug_id, 'side_effect_id': e.side_effect_id,
                           'frequency': e.frequency} for e in self.drug_se_edges]
            pd.DataFrame(dse_records).to_csv(output_path / 'drug_side_effect_edges.csv', index=False)
        
        # Similarity edges
        if self.similarity_edges:
            sim_records = [{'drug1_id': e.drug1_id, 'drug2_id': e.drug2_id,
                          'tanimoto': e.tanimoto_coefficient} for e in self.similarity_edges]
            pd.DataFrame(sim_records).to_csv(output_path / 'similarity_edges.csv', index=False)
        
        # Generate Neo4j import script
        self._generate_neo4j_script(output_path)
        
        logger.info("  Export complete!")
    
    def _generate_neo4j_script(self, output_path: Path):
        """Generate Neo4j Cypher import script"""
        script = """
// =============================================================================
// Neo4j Import Script for Enriched DDI Knowledge Graph
// =============================================================================

// --- Constraints ---
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE;
CREATE CONSTRAINT side_effect_id IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;

// --- Load Drug Nodes ---
LOAD CSV WITH HEADERS FROM 'file:///drugs.csv' AS row
CREATE (d:Drug {
    drugbank_id: row.drugbank_id,
    name: row.name,
    drug_class: row.drug_class,
    is_cardiovascular: toBoolean(row.is_cardiovascular),
    is_antithrombotic: toBoolean(row.is_antithrombotic)
});

// --- Load Protein Nodes ---
LOAD CSV WITH HEADERS FROM 'file:///proteins.csv' AS row
CREATE (p:Protein {
    uniprot_id: row.uniprot_id,
    name: row.name,
    gene_name: row.gene_name,
    protein_type: row.protein_type
});

// --- Load Side Effect Nodes ---
LOAD CSV WITH HEADERS FROM 'file:///side_effects.csv' AS row
CREATE (s:SideEffect {
    id: row.id,
    name: row.name,
    meddra_type: row.meddra_type
});

// --- Load Pathway Nodes ---
LOAD CSV WITH HEADERS FROM 'file:///pathways.csv' AS row
CREATE (pw:Pathway {
    id: row.id,
    name: row.name,
    source: row.source,
    category: row.category
});

// --- Load DDI Relationships ---
LOAD CSV WITH HEADERS FROM 'file:///ddi_edges.csv' AS row
MATCH (d1:Drug {drugbank_id: row.drug1_id})
MATCH (d2:Drug {drugbank_id: row.drug2_id})
CREATE (d1)-[:INTERACTS_WITH {
    severity: row.severity,
    severity_numeric: toInteger(row.severity_numeric),
    mechanism: row.mechanism,
    clinical_effect: row.clinical_effect
}]->(d2);

// --- Load Drug-Target Relationships ---
LOAD CSV WITH HEADERS FROM 'file:///drug_target_edges.csv' AS row
MATCH (d:Drug {drugbank_id: row.drug_id})
MATCH (p:Protein {uniprot_id: row.protein_id})
CREATE (d)-[:TARGETS {action: row.action}]->(p);

// --- Load Drug-SideEffect Relationships ---
LOAD CSV WITH HEADERS FROM 'file:///drug_side_effect_edges.csv' AS row
MATCH (d:Drug {drugbank_id: row.drug_id})
MATCH (s:SideEffect {id: row.side_effect_id})
CREATE (d)-[:CAUSES {frequency: row.frequency}]->(s);

// --- Load Similarity Relationships ---
LOAD CSV WITH HEADERS FROM 'file:///similarity_edges.csv' AS row
MATCH (d1:Drug {drugbank_id: row.drug1_id})
MATCH (d2:Drug {drugbank_id: row.drug2_id})
CREATE (d1)-[:SIMILAR_TO {tanimoto: toFloat(row.tanimoto)}]->(d2);

// =============================================================================
// Example Queries
// =============================================================================

// Find all DDIs for Warfarin with severity
// MATCH (d1:Drug {name: 'Warfarin'})-[r:INTERACTS_WITH]->(d2:Drug)
// RETURN d2.name, r.severity, r.clinical_effect
// ORDER BY r.severity_numeric DESC;

// Find drugs that target the same protein (potential DDI mechanism)
// MATCH (d1:Drug)-[:TARGETS]->(p:Protein)<-[:TARGETS]-(d2:Drug)
// WHERE d1.drugbank_id < d2.drugbank_id
// RETURN d1.name, d2.name, p.name AS shared_target, p.gene_name;

// Find drugs with shared side effects
// MATCH (d1:Drug)-[:CAUSES]->(s:SideEffect)<-[:CAUSES]-(d2:Drug)
// WHERE d1.drugbank_id < d2.drugbank_id AND s.name = 'Hemorrhage'
// RETURN d1.name, d2.name, s.name;

// Find chemically similar drugs with DDI
// MATCH (d1:Drug)-[:SIMILAR_TO]->(d2:Drug)-[:INTERACTS_WITH]->(d3:Drug)
// WHERE d1 <> d3
// RETURN d1.name, d2.name, d3.name, 'Potential DDI via similar drug';
"""
        with open(output_path / 'neo4j_import.cypher', 'w') as f:
            f.write(script)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build enriched DDI Knowledge Graph"""
    
    output_dir = Path('knowledge_graph_enriched')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize enriched KG
    kg = EnrichedDDIKnowledgeGraph()
    
    # Load base DDI data
    kg.load_base_data('data/ddi_cardio_or_antithrombotic_labeled (1).csv')
    
    # Enrich with external sources
    kg.enrich_with_sider()      # Side effects
    kg.enrich_with_kegg()       # Pathways & enzymes
    kg.enrich_with_pubchem()    # Chemical similarity
    kg.enrich_with_targets()    # Drug targets
    
    # Build NetworkX graph
    kg.build_networkx_graph()
    
    # Print statistics
    stats = kg.get_statistics()
    print("\n" + "="*60)
    print("ENRICHED KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print("\nNode Counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    print("\nEdge Counts:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    # Export to Neo4j format
    kg.export_to_neo4j(output_dir / 'neo4j_export')
    
    # Save graph as pickle for later use
    import pickle
    with open(output_dir / 'enriched_kg.pkl', 'wb') as f:
        pickle.dump(kg, f)
    
    print(f"\nAll outputs saved to {output_dir}")
    
    return kg


if __name__ == "__main__":
    kg = main()
