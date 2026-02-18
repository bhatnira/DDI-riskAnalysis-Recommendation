#!/usr/bin/env python3
"""
DDI Knowledge Graph Builder for Cardiovascular/Antithrombotic Drugs

Builds a comprehensive knowledge graph from DrugBank cardiovascular DDI data.

Nodes:
- Drug (DrugBank ID, name, ATC codes, drug class)
- ATCClass (therapeutic classification hierarchy)
- InteractionType (mechanism categories)

Edges:
- Drug -[INTERACTS_WITH]-> Drug (with severity, mechanism)
- Drug -[BELONGS_TO]-> ATCClass
- Drug -[HAS_MECHANISM]-> InteractionType

Supports:
- NetworkX for analysis
- Neo4j export for graph database
- Graph embeddings for DDI prediction

Author: DDI Risk Analysis Research Team
Date: 2026
"""

import os
import json
import re
import ast
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DrugNode:
    """Drug entity in knowledge graph"""
    drugbank_id: str
    name: str
    atc_codes: List[str]
    is_cardiovascular: bool
    is_antithrombotic: bool
    
    # Derived attributes
    drug_class: str = ""  # From ATC level 2
    mechanism_class: str = ""  # From ATC level 4
    
    def to_dict(self) -> dict:
        return {
            'drugbank_id': self.drugbank_id,
            'name': self.name,
            'atc_codes': self.atc_codes,
            'is_cardiovascular': self.is_cardiovascular,
            'is_antithrombotic': self.is_antithrombotic,
            'drug_class': self.drug_class,
            'mechanism_class': self.mechanism_class
        }


@dataclass 
class DDIEdge:
    """Drug-Drug Interaction edge"""
    drug1_id: str
    drug2_id: str
    description: str
    severity: str
    severity_numeric: int
    confidence: float
    mechanism: str = ""  # Extracted from description
    clinical_effect: str = ""  # Extracted from description
    
    def to_dict(self) -> dict:
        return {
            'drug1_id': self.drug1_id,
            'drug2_id': self.drug2_id,
            'description': self.description,
            'severity': self.severity,
            'severity_numeric': self.severity_numeric,
            'confidence': self.confidence,
            'mechanism': self.mechanism,
            'clinical_effect': self.clinical_effect
        }


# ============================================================================
# ATC CODE PARSER
# ============================================================================

class ATCParser:
    """Parse ATC codes into hierarchical classification"""
    
    # ATC Level 1: Anatomical main group
    ATC_LEVEL1 = {
        'A': 'Alimentary tract and metabolism',
        'B': 'Blood and blood forming organs',
        'C': 'Cardiovascular system',
        'D': 'Dermatologicals',
        'G': 'Genito-urinary system',
        'H': 'Systemic hormones',
        'J': 'Antiinfectives for systemic use',
        'L': 'Antineoplastic and immunomodulating',
        'M': 'Musculo-skeletal system',
        'N': 'Nervous system',
        'P': 'Antiparasitic products',
        'R': 'Respiratory system',
        'S': 'Sensory organs',
        'V': 'Various'
    }
    
    # ATC Level 2 for cardiovascular (C) and blood (B) - therapeutic subgroup
    ATC_LEVEL2_CARDIO = {
        'C01': 'Cardiac therapy',
        'C02': 'Antihypertensives',
        'C03': 'Diuretics',
        'C04': 'Peripheral vasodilators',
        'C05': 'Vasoprotectives',
        'C07': 'Beta blocking agents',
        'C08': 'Calcium channel blockers',
        'C09': 'Agents acting on RAAS',
        'C10': 'Lipid modifying agents',
    }
    
    ATC_LEVEL2_BLOOD = {
        'B01': 'Antithrombotic agents',
        'B02': 'Antihemorrhagics',
        'B03': 'Antianemic preparations',
        'B05': 'Blood substitutes',
        'B06': 'Other hematological agents',
    }
    
    # ATC Level 3 for antithrombotics (B01)
    ATC_LEVEL3_ANTITHROMBOTIC = {
        'B01A': 'Antithrombotic agents',
    }
    
    # ATC Level 4 for specific mechanism
    ATC_LEVEL4_ANTITHROMBOTIC = {
        'B01AA': 'Vitamin K antagonists',
        'B01AB': 'Heparin group',
        'B01AC': 'Platelet aggregation inhibitors',
        'B01AD': 'Enzymes',
        'B01AE': 'Direct thrombin inhibitors',
        'B01AF': 'Direct factor Xa inhibitors',
        'B01AX': 'Other antithrombotic agents',
    }
    
    @classmethod
    def parse_atc_list(cls, atc_str: str) -> List[str]:
        """Parse ATC codes from string representation"""
        if pd.isna(atc_str) or atc_str == '[]':
            return []
        try:
            return ast.literal_eval(atc_str)
        except:
            return []
    
    @classmethod
    def get_level1(cls, atc_code: str) -> str:
        """Get anatomical main group (level 1)"""
        if not atc_code:
            return 'Unknown'
        return cls.ATC_LEVEL1.get(atc_code[0], 'Unknown')
    
    @classmethod
    def get_level2(cls, atc_code: str) -> str:
        """Get therapeutic subgroup (level 2)"""
        if not atc_code or len(atc_code) < 3:
            return 'Unknown'
        level2_code = atc_code[:3]
        return (cls.ATC_LEVEL2_CARDIO.get(level2_code) or 
                cls.ATC_LEVEL2_BLOOD.get(level2_code) or 
                level2_code)
    
    @classmethod
    def get_level4(cls, atc_code: str) -> str:
        """Get chemical/therapeutic subgroup (level 4)"""
        if not atc_code or len(atc_code) < 5:
            return 'Unknown'
        level4_code = atc_code[:5]
        return cls.ATC_LEVEL4_ANTITHROMBOTIC.get(level4_code, level4_code)
    
    @classmethod
    def get_drug_class(cls, atc_codes: List[str]) -> str:
        """Get primary drug class from ATC codes"""
        for code in atc_codes:
            if code.startswith('C'):
                return cls.get_level2(code)
            if code.startswith('B01'):
                return cls.get_level4(code)
        return 'Other'


# ============================================================================
# MECHANISM EXTRACTOR
# ============================================================================

class MechanismExtractor:
    """Extract interaction mechanism from DDI description"""
    
    # Pharmacokinetic mechanisms
    PK_PATTERNS = {
        'metabolism_increase': [
            r'increase.*metabolism',
            r'induc.*cyp',
            r'induc.*enzyme',
        ],
        'metabolism_decrease': [
            r'decrease.*metabolism',
            r'inhibit.*cyp',
            r'inhibit.*enzyme',
        ],
        'concentration_increase': [
            r'increase.*concentration',
            r'increase.*serum',
            r'increase.*plasma',
            r'increase.*auc',
            r'increase.*level',
        ],
        'concentration_decrease': [
            r'decrease.*concentration',
            r'decrease.*serum',
            r'decrease.*plasma',
            r'decrease.*auc',
            r'decrease.*level',
            r'reduce.*absorption',
        ],
        'excretion_change': [
            r'excretion.*rate',
            r'renal.*clearance',
        ],
    }
    
    # Pharmacodynamic mechanisms
    PD_PATTERNS = {
        'additive_effect': [
            r'increase.*activit',
            r'increase.*effect',
            r'additive',
            r'synergist',
        ],
        'antagonistic_effect': [
            r'decrease.*activit',
            r'decrease.*effect',
            r'antagoni',
            r'reduce.*efficacy',
        ],
        'bleeding_risk': [
            r'bleeding',
            r'hemorrhag',
            r'anticoagulant.*activit',
            r'antithrombotic.*activit',
        ],
        'cardiac_effect': [
            r'qt.*prolong',
            r'bradycardia',
            r'hypotension',
            r'arrhythm',
        ],
        'toxicity': [
            r'toxic',
            r'adverse',
            r'risk.*severity',
        ],
    }
    
    # Clinical effects
    CLINICAL_EFFECTS = {
        'bleeding': ['bleeding', 'hemorrhag', 'anticoagulant'],
        'hypotension': ['hypotension', 'blood pressure'],
        'bradycardia': ['bradycardia', 'heart rate'],
        'qt_prolongation': ['qt prolong', 'torsade', 'arrhythm'],
        'hyperkalemia': ['hyperkalemia', 'potassium'],
        'hypoglycemia': ['hypoglycemia', 'blood sugar'],
        'nephrotoxicity': ['nephrotox', 'kidney', 'renal'],
        'hepatotoxicity': ['hepatotox', 'liver'],
        'myopathy': ['myopathy', 'rhabdomyolysis', 'muscle'],
        'cns_depression': ['sedation', 'drowsiness', 'cns depress'],
    }
    
    @classmethod
    def extract_mechanism(cls, description: str) -> str:
        """Extract primary mechanism from description"""
        if not description:
            return 'unknown'
        
        desc_lower = description.lower()
        
        # Check PK mechanisms first
        for mechanism, patterns in cls.PK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    return f'pk_{mechanism}'
        
        # Check PD mechanisms
        for mechanism, patterns in cls.PD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    return f'pd_{mechanism}'
        
        return 'unknown'
    
    @classmethod
    def extract_clinical_effect(cls, description: str) -> str:
        """Extract clinical effect from description"""
        if not description:
            return 'unknown'
        
        desc_lower = description.lower()
        
        for effect, keywords in cls.CLINICAL_EFFECTS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return effect
        
        return 'general'


# ============================================================================
# KNOWLEDGE GRAPH BUILDER
# ============================================================================

class DDIKnowledgeGraph:
    """
    Build and analyze DDI Knowledge Graph
    
    Graph Structure:
    - Nodes: Drugs, ATC Classes, Mechanisms
    - Edges: INTERACTS_WITH, BELONGS_TO, HAS_MECHANISM
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.drugs: Dict[str, DrugNode] = {}
        self.interactions: List[DDIEdge] = []
        self.atc_classes: Set[str] = set()
        self.mechanisms: Set[str] = set()
        
    def load_from_csv(self, csv_path: str):
        """Load DDI data from CSV and build knowledge graph"""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} DDI pairs")
        
        # Build drug nodes
        self._build_drug_nodes(df)
        
        # Build interaction edges
        self._build_interaction_edges(df)
        
        # Build NetworkX graph
        self._build_networkx_graph()
        
        logger.info(f"\nKnowledge Graph Statistics:")
        logger.info(f"  Drug nodes: {len(self.drugs):,}")
        logger.info(f"  DDI edges: {len(self.interactions):,}")
        logger.info(f"  ATC classes: {len(self.atc_classes)}")
        logger.info(f"  Mechanism types: {len(self.mechanisms)}")
        
    def _build_drug_nodes(self, df: pd.DataFrame):
        """Extract unique drug nodes from dataframe"""
        logger.info("Building drug nodes...")
        
        # Process drug_1
        for _, row in tqdm(df.drop_duplicates('drugbank_id_1').iterrows(), 
                          desc="Processing drugs", total=df['drugbank_id_1'].nunique()):
            atc_codes = ATCParser.parse_atc_list(row['atc_1'])
            drug = DrugNode(
                drugbank_id=row['drugbank_id_1'],
                name=row['drug_name_1'],
                atc_codes=atc_codes,
                is_cardiovascular=row['is_cardiovascular_1'],
                is_antithrombotic=row['is_antithrombotic_1'],
                drug_class=ATCParser.get_drug_class(atc_codes)
            )
            self.drugs[drug.drugbank_id] = drug
            
            # Track ATC classes
            for code in atc_codes:
                self.atc_classes.add(ATCParser.get_level2(code))
        
        # Process drug_2 (add any not already seen)
        for _, row in df.drop_duplicates('drugbank_id_2').iterrows():
            if row['drugbank_id_2'] not in self.drugs:
                atc_codes = ATCParser.parse_atc_list(row['atc_2'])
                drug = DrugNode(
                    drugbank_id=row['drugbank_id_2'],
                    name=row['drug_name_2'],
                    atc_codes=atc_codes,
                    is_cardiovascular=row['is_cardiovascular_2'],
                    is_antithrombotic=row['is_antithrombotic_2'],
                    drug_class=ATCParser.get_drug_class(atc_codes)
                )
                self.drugs[drug.drugbank_id] = drug
    
    def _build_interaction_edges(self, df: pd.DataFrame):
        """Build DDI edges with mechanism extraction"""
        logger.info("Building interaction edges...")
        
        for _, row in tqdm(df.iterrows(), desc="Processing DDIs", total=len(df)):
            mechanism = MechanismExtractor.extract_mechanism(row['interaction_description'])
            clinical_effect = MechanismExtractor.extract_clinical_effect(row['interaction_description'])
            
            edge = DDIEdge(
                drug1_id=row['drugbank_id_1'],
                drug2_id=row['drugbank_id_2'],
                description=row['interaction_description'],
                severity=row['severity_label'],
                severity_numeric=int(row['severity_numeric']),
                confidence=row['severity_confidence'],
                mechanism=mechanism,
                clinical_effect=clinical_effect
            )
            self.interactions.append(edge)
            self.mechanisms.add(mechanism)
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for analysis"""
        logger.info("Building NetworkX graph...")
        
        # Add drug nodes
        for drug_id, drug in self.drugs.items():
            self.graph.add_node(
                drug_id,
                node_type='drug',
                name=drug.name,
                drug_class=drug.drug_class,
                is_cardiovascular=drug.is_cardiovascular,
                is_antithrombotic=drug.is_antithrombotic
            )
        
        # Add ATC class nodes
        for atc_class in self.atc_classes:
            self.graph.add_node(
                f"ATC_{atc_class}",
                node_type='atc_class',
                name=atc_class
            )
        
        # Add mechanism nodes
        for mechanism in self.mechanisms:
            self.graph.add_node(
                f"MECH_{mechanism}",
                node_type='mechanism',
                name=mechanism
            )
        
        # Add DDI edges
        for edge in self.interactions:
            self.graph.add_edge(
                edge.drug1_id,
                edge.drug2_id,
                edge_type='INTERACTS_WITH',
                severity=edge.severity,
                severity_numeric=edge.severity_numeric,
                mechanism=edge.mechanism,
                clinical_effect=edge.clinical_effect
            )
        
        # Add drug -> ATC class edges
        for drug_id, drug in self.drugs.items():
            for atc_code in drug.atc_codes:
                atc_class = ATCParser.get_level2(atc_code)
                self.graph.add_edge(
                    drug_id,
                    f"ATC_{atc_class}",
                    edge_type='BELONGS_TO'
                )
        
        logger.info(f"  Graph nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Graph edges: {self.graph.number_of_edges():,}")
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def get_drug_interaction_count(self) -> pd.DataFrame:
        """Get interaction count per drug"""
        drug_counts = defaultdict(lambda: {'total': 0, 'major': 0, 'moderate': 0, 'minor': 0})
        
        for edge in self.interactions:
            drug_counts[edge.drug1_id]['total'] += 1
            drug_counts[edge.drug2_id]['total'] += 1
            
            if 'Major' in edge.severity:
                drug_counts[edge.drug1_id]['major'] += 1
                drug_counts[edge.drug2_id]['major'] += 1
            elif 'Moderate' in edge.severity:
                drug_counts[edge.drug1_id]['moderate'] += 1
                drug_counts[edge.drug2_id]['moderate'] += 1
            else:
                drug_counts[edge.drug1_id]['minor'] += 1
                drug_counts[edge.drug2_id]['minor'] += 1
        
        rows = []
        for drug_id, counts in drug_counts.items():
            drug = self.drugs.get(drug_id)
            rows.append({
                'drugbank_id': drug_id,
                'drug_name': drug.name if drug else 'Unknown',
                'drug_class': drug.drug_class if drug else 'Unknown',
                'total_interactions': counts['total'],
                'major_interactions': counts['major'],
                'moderate_interactions': counts['moderate'],
                'minor_interactions': counts['minor'],
            })
        
        df = pd.DataFrame(rows).sort_values('total_interactions', ascending=False)
        return df
    
    def get_mechanism_distribution(self) -> pd.DataFrame:
        """Get distribution of interaction mechanisms"""
        mechanism_counts = Counter(edge.mechanism for edge in self.interactions)
        
        df = pd.DataFrame([
            {'mechanism': mech, 'count': count}
            for mech, count in mechanism_counts.most_common()
        ])
        df['percentage'] = df['count'] / df['count'].sum() * 100
        return df
    
    def get_clinical_effect_distribution(self) -> pd.DataFrame:
        """Get distribution of clinical effects"""
        effect_counts = Counter(edge.clinical_effect for edge in self.interactions)
        
        df = pd.DataFrame([
            {'clinical_effect': effect, 'count': count}
            for effect, count in effect_counts.most_common()
        ])
        df['percentage'] = df['count'] / df['count'].sum() * 100
        return df
    
    def get_drug_class_interactions(self) -> pd.DataFrame:
        """Get interaction matrix between drug classes"""
        class_interactions = defaultdict(lambda: defaultdict(int))
        
        for edge in self.interactions:
            drug1 = self.drugs.get(edge.drug1_id)
            drug2 = self.drugs.get(edge.drug2_id)
            
            if drug1 and drug2:
                class1 = drug1.drug_class
                class2 = drug2.drug_class
                class_interactions[class1][class2] += 1
                if class1 != class2:
                    class_interactions[class2][class1] += 1
        
        # Convert to matrix
        classes = sorted(class_interactions.keys())
        matrix = pd.DataFrame(
            [[class_interactions[c1][c2] for c2 in classes] for c1 in classes],
            index=classes,
            columns=classes
        )
        return matrix
    
    def find_drugs_interacting_with(self, drug_name: str) -> pd.DataFrame:
        """Find all drugs that interact with a specific drug"""
        drug_name_lower = drug_name.lower()
        
        # Find drug ID
        drug_id = None
        for did, drug in self.drugs.items():
            if drug.name.lower() == drug_name_lower:
                drug_id = did
                break
        
        if not drug_id:
            return pd.DataFrame()
        
        # Find interactions
        results = []
        for edge in self.interactions:
            if edge.drug1_id == drug_id:
                other_drug = self.drugs.get(edge.drug2_id)
                results.append({
                    'drug': other_drug.name if other_drug else edge.drug2_id,
                    'severity': edge.severity,
                    'mechanism': edge.mechanism,
                    'clinical_effect': edge.clinical_effect,
                    'description': edge.description[:100] + '...'
                })
            elif edge.drug2_id == drug_id:
                other_drug = self.drugs.get(edge.drug1_id)
                results.append({
                    'drug': other_drug.name if other_drug else edge.drug1_id,
                    'severity': edge.severity,
                    'mechanism': edge.mechanism,
                    'clinical_effect': edge.clinical_effect,
                    'description': edge.description[:100] + '...'
                })
        
        return pd.DataFrame(results).drop_duplicates()
    
    def find_high_risk_combinations(self, min_severity: int = 3) -> pd.DataFrame:
        """Find high-risk drug combinations"""
        high_risk = []
        
        for edge in self.interactions:
            if edge.severity_numeric >= min_severity:
                drug1 = self.drugs.get(edge.drug1_id)
                drug2 = self.drugs.get(edge.drug2_id)
                
                high_risk.append({
                    'drug1': drug1.name if drug1 else edge.drug1_id,
                    'drug2': drug2.name if drug2 else edge.drug2_id,
                    'severity': edge.severity,
                    'mechanism': edge.mechanism,
                    'clinical_effect': edge.clinical_effect,
                })
        
        return pd.DataFrame(high_risk).drop_duplicates()
    
    # =========================================================================
    # GRAPH METRICS
    # =========================================================================
    
    def compute_centrality_metrics(self) -> pd.DataFrame:
        """Compute centrality metrics for drugs"""
        # Create undirected version for some metrics
        G_undirected = self.graph.to_undirected()
        
        # Only drug nodes
        drug_nodes = [n for n in self.graph.nodes() if not n.startswith('ATC_') and not n.startswith('MECH_')]
        G_drugs = G_undirected.subgraph(drug_nodes)
        
        logger.info("Computing centrality metrics...")
        
        # Degree centrality
        degree = nx.degree_centrality(G_drugs)
        
        # Betweenness centrality (sample for performance)
        if len(drug_nodes) > 1000:
            betweenness = nx.betweenness_centrality(G_drugs, k=min(500, len(drug_nodes)))
        else:
            betweenness = nx.betweenness_centrality(G_drugs)
        
        # PageRank
        pagerank = nx.pagerank(G_drugs)
        
        # Build results
        results = []
        for drug_id in drug_nodes:
            drug = self.drugs.get(drug_id)
            results.append({
                'drugbank_id': drug_id,
                'drug_name': drug.name if drug else 'Unknown',
                'drug_class': drug.drug_class if drug else 'Unknown',
                'degree_centrality': degree.get(drug_id, 0),
                'betweenness_centrality': betweenness.get(drug_id, 0),
                'pagerank': pagerank.get(drug_id, 0),
            })
        
        df = pd.DataFrame(results).sort_values('pagerank', ascending=False)
        return df
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def export_to_neo4j_csv(self, output_dir: str = 'kg_export'):
        """Export graph to CSV files for Neo4j import"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Exporting to {output_path}...")
        
        # Export drug nodes
        drug_df = pd.DataFrame([drug.to_dict() for drug in self.drugs.values()])
        drug_df.to_csv(output_path / 'drugs.csv', index=False)
        
        # Export DDI edges
        ddi_df = pd.DataFrame([edge.to_dict() for edge in self.interactions])
        ddi_df.to_csv(output_path / 'interactions.csv', index=False)
        
        # Export ATC classes
        atc_df = pd.DataFrame([{'class_name': c} for c in self.atc_classes])
        atc_df.to_csv(output_path / 'atc_classes.csv', index=False)
        
        # Export mechanisms
        mech_df = pd.DataFrame([{'mechanism': m} for m in self.mechanisms])
        mech_df.to_csv(output_path / 'mechanisms.csv', index=False)
        
        # Generate Cypher import script
        cypher_script = """
// Neo4j Import Script for DDI Knowledge Graph
// Generated automatically

// Create constraints
CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE;
CREATE CONSTRAINT atc_class IF NOT EXISTS FOR (a:ATCClass) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT mechanism IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE;

// Load drugs
LOAD CSV WITH HEADERS FROM 'file:///drugs.csv' AS row
CREATE (d:Drug {
    drugbank_id: row.drugbank_id,
    name: row.name,
    drug_class: row.drug_class,
    is_cardiovascular: toBoolean(row.is_cardiovascular),
    is_antithrombotic: toBoolean(row.is_antithrombotic)
});

// Load ATC classes
LOAD CSV WITH HEADERS FROM 'file:///atc_classes.csv' AS row
CREATE (a:ATCClass {name: row.class_name});

// Load mechanisms
LOAD CSV WITH HEADERS FROM 'file:///mechanisms.csv' AS row
CREATE (m:Mechanism {name: row.mechanism});

// Load DDI relationships
LOAD CSV WITH HEADERS FROM 'file:///interactions.csv' AS row
MATCH (d1:Drug {drugbank_id: row.drug1_id})
MATCH (d2:Drug {drugbank_id: row.drug2_id})
CREATE (d1)-[:INTERACTS_WITH {
    severity: row.severity,
    severity_numeric: toInteger(row.severity_numeric),
    mechanism: row.mechanism,
    clinical_effect: row.clinical_effect
}]->(d2);

// Create BELONGS_TO relationships (run separately after loading drugs)
// MATCH (d:Drug), (a:ATCClass)
// WHERE d.drug_class = a.name
// CREATE (d)-[:BELONGS_TO]->(a);
"""
        
        with open(output_path / 'neo4j_import.cypher', 'w') as f:
            f.write(cypher_script)
        
        logger.info(f"Exported files to {output_path}")
        logger.info(f"  - drugs.csv: {len(drug_df)} drugs")
        logger.info(f"  - interactions.csv: {len(ddi_df)} DDIs")
        logger.info(f"  - atc_classes.csv: {len(atc_df)} classes")
        logger.info(f"  - mechanisms.csv: {len(mech_df)} mechanisms")
        logger.info(f"  - neo4j_import.cypher: Import script")
    
    def export_graph_json(self, output_path: str = 'kg_export/graph.json'):
        """Export graph as JSON for visualization"""
        nodes = []
        edges = []
        
        # Drug nodes only for cleaner visualization
        for drug_id, drug in self.drugs.items():
            nodes.append({
                'id': drug_id,
                'label': drug.name,
                'group': drug.drug_class,
                'type': 'drug'
            })
        
        # DDI edges
        for i, edge in enumerate(self.interactions[:10000]):  # Limit for viz
            edges.append({
                'id': f'e{i}',
                'source': edge.drug1_id,
                'target': edge.drug2_id,
                'severity': edge.severity_numeric,
                'mechanism': edge.mechanism
            })
        
        graph_data = {'nodes': nodes, 'edges': edges}
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Exported graph JSON to {output_path}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_drug_class_heatmap(kg: DDIKnowledgeGraph, output_path: str = None):
    """Visualize interaction heatmap between drug classes"""
    matrix = kg.get_drug_class_interactions()
    
    # Filter to top classes
    top_classes = matrix.sum(axis=1).nlargest(15).index.tolist()
    matrix = matrix.loc[top_classes, top_classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        np.log1p(matrix),  # Log scale for better visualization
        annot=False,
        cmap='YlOrRd',
        xticklabels=True,
        yticklabels=True
    )
    plt.title('Drug Class Interaction Heatmap (log scale)')
    plt.xlabel('Drug Class')
    plt.ylabel('Drug Class')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
    plt.close()


def visualize_mechanism_distribution(kg: DDIKnowledgeGraph, output_path: str = None):
    """Visualize distribution of interaction mechanisms"""
    mech_df = kg.get_mechanism_distribution()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mech_df.head(15), x='count', y='mechanism', palette='viridis')
    plt.title('Top 15 Interaction Mechanisms')
    plt.xlabel('Count')
    plt.ylabel('Mechanism')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved mechanism plot to {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build and analyze DDI Knowledge Graph"""
    
    # Create output directory
    output_dir = Path('knowledge_graph')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    
    # Build knowledge graph
    kg = DDIKnowledgeGraph()
    kg.load_from_csv('data/ddi_cardio_or_antithrombotic_labeled (1).csv')
    
    # Analysis
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH ANALYSIS")
    print("="*60)
    
    # Top interacting drugs
    drug_counts = kg.get_drug_interaction_count()
    print("\nTop 10 Drugs by Interaction Count:")
    print(drug_counts[['drug_name', 'drug_class', 'total_interactions', 'major_interactions']].head(10).to_string(index=False))
    drug_counts.to_csv(output_dir / 'tables' / 'drug_interaction_counts.csv', index=False)
    
    # Mechanism distribution
    mech_df = kg.get_mechanism_distribution()
    print("\nInteraction Mechanisms:")
    print(mech_df.head(10).to_string(index=False))
    mech_df.to_csv(output_dir / 'tables' / 'mechanism_distribution.csv', index=False)
    
    # Clinical effects
    effects_df = kg.get_clinical_effect_distribution()
    print("\nClinical Effects:")
    print(effects_df.head(10).to_string(index=False))
    effects_df.to_csv(output_dir / 'tables' / 'clinical_effects.csv', index=False)
    
    # High-risk combinations
    high_risk = kg.find_high_risk_combinations()
    print(f"\nHigh-risk combinations: {len(high_risk):,}")
    high_risk.head(100).to_csv(output_dir / 'tables' / 'high_risk_combinations.csv', index=False)
    
    # Centrality metrics
    centrality = kg.compute_centrality_metrics()
    print("\nTop 10 Drugs by PageRank (Hub Drugs):")
    print(centrality[['drug_name', 'drug_class', 'pagerank', 'degree_centrality']].head(10).to_string(index=False))
    centrality.to_csv(output_dir / 'tables' / 'drug_centrality.csv', index=False)
    
    # Example query
    print("\n" + "="*60)
    print("Example: Warfarin Interactions")
    print("="*60)
    warfarin_interactions = kg.find_drugs_interacting_with('Warfarin')
    print(f"Total interactions: {len(warfarin_interactions)}")
    print("\nSample interactions:")
    print(warfarin_interactions[['drug', 'severity', 'mechanism', 'clinical_effect']].head(10).to_string(index=False))
    
    # Visualizations
    visualize_drug_class_heatmap(kg, output_dir / 'figures' / 'drug_class_heatmap.png')
    visualize_mechanism_distribution(kg, output_dir / 'figures' / 'mechanism_distribution.png')
    
    # Export for Neo4j
    kg.export_to_neo4j_csv(output_dir / 'neo4j_export')
    kg.export_graph_json(output_dir / 'graph.json')
    
    print(f"\nAll outputs saved to {output_dir}")
    
    return kg


if __name__ == "__main__":
    kg = main()
