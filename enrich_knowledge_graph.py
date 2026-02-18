#!/usr/bin/env python3
"""
Enrich DDI Knowledge Graph with External Databases.

Integrates data from:
1. SIDER - Side effects (download TSV files)
2. STITCH - Chemical-protein interactions
3. STRING - Protein-protein interactions  
4. DisGeNET - Disease-gene associations
5. GO - Gene Ontology annotations
6. CTD - Comparative Toxicogenomics Database
7. PharmGKB - Pharmacogenomics data

Uses existing external IDs from DrugBank (PubChem, UniProt, ChEMBL, KEGG).
"""

import os
import sys
import gzip
import pickle
import json
import logging
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd
import networkx as nx
import time

# Import dataclasses from original KG module for pickle compatibility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddi_knowledge_graph_real import (
    DrugNode, ProteinNode, PathwayNode, CategoryNode,
    SNPEffect, SNPAdverseReaction, DDIEdge, DrugProteinEdge,
    DrugPathwayEdge, DrugCategoryEdge
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES FOR NEW NODE TYPES
# =============================================================================

@dataclass
class SideEffectNode:
    """Side effect from SIDER."""
    id: str  # UMLS CUI or MedDRA ID
    name: str
    meddra_type: str = ""  # PT (Preferred Term), LLT (Lowest Level Term)


@dataclass
class DiseaseNode:
    """Disease from DisGeNET or CTD."""
    id: str  # UMLS CUI, MeSH, or OMIM ID
    name: str
    source: str = ""
    semantic_type: str = ""


@dataclass  
class GOTermNode:
    """Gene Ontology term."""
    id: str  # GO:XXXXXXX
    name: str
    namespace: str = ""  # molecular_function, biological_process, cellular_component


@dataclass
class DrugSideEffectEdge:
    """Drug-side effect relationship from SIDER."""
    drug_id: str
    side_effect_id: str
    frequency: str = ""
    source: str = "SIDER"


@dataclass
class DrugDiseaseEdge:
    """Drug-disease relationship (indication or adverse effect)."""
    drug_id: str
    disease_id: str
    relationship_type: str = ""  # therapeutic, marker/mechanism
    source: str = ""


@dataclass
class ProteinDiseaseEdge:
    """Protein-disease association from DisGeNET."""
    protein_id: str
    disease_id: str
    score: float = 0.0
    source: str = ""


@dataclass
class ProteinGOEdge:
    """Protein-GO term annotation."""
    protein_id: str
    go_id: str
    evidence_code: str = ""


@dataclass
class ProteinProteinEdge:
    """Protein-protein interaction from STRING."""
    protein1_id: str
    protein2_id: str
    combined_score: int = 0
    interaction_types: List[str] = field(default_factory=list)


# =============================================================================
# DATABASE DOWNLOADERS
# =============================================================================

class SIDERIntegrator:
    """
    Integrates SIDER (Side Effect Resource) data.
    Download from: http://sideeffects.embl.de/download/
    
    Files needed:
    - meddra_all_se.tsv.gz (all side effects)
    - meddra_freq.tsv.gz (frequency information)
    - drug_names.tsv (drug name mappings)
    """
    
    SIDER_BASE_URL = "http://sideeffects.embl.de/media/download/"
    
    def __init__(self, data_dir: str = "external_data/sider"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.side_effects: Dict[str, SideEffectNode] = {}
        self.drug_se_edges: List[DrugSideEffectEdge] = []
        
        # Mapping from SIDER compound IDs to DrugBank IDs
        self.stitch_to_drugbank: Dict[str, str] = {}
        self.pubchem_to_drugbank: Dict[str, str] = {}
    
    def download_files(self) -> bool:
        """Download SIDER data files."""
        files_to_download = [
            "meddra_all_se.tsv.gz",
            "meddra_freq.tsv.gz", 
            "drug_names.tsv",
        ]
        
        for filename in files_to_download:
            filepath = self.data_dir / filename
            if not filepath.exists():
                url = self.SIDER_BASE_URL + filename
                logger.info(f"Downloading {filename}...")
                try:
                    response = requests.get(url, timeout=60)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"  Downloaded {filename}")
                    else:
                        logger.warning(f"  Failed to download {filename}: {response.status_code}")
                        return False
                except Exception as e:
                    logger.warning(f"  Error downloading {filename}: {e}")
                    return False
        return True
    
    def build_id_mapping(self, drugs: Dict) -> None:
        """Build mapping from PubChem/STITCH IDs to DrugBank IDs."""
        for drug_id, drug in drugs.items():
            # Map PubChem Compound ID
            if 'PubChem Compound' in drug.external_ids:
                pubchem_id = drug.external_ids['PubChem Compound']
                self.pubchem_to_drugbank[pubchem_id] = drug_id
                # STITCH uses CID with prefix (CID followed by zeros)
                stitch_id = f"CID1{pubchem_id.zfill(8)}"
                self.stitch_to_drugbank[stitch_id] = drug_id
        
        logger.info(f"  Built mappings for {len(self.pubchem_to_drugbank)} PubChem IDs")
    
    def load_side_effects(self) -> None:
        """Load side effects from SIDER meddra_all_se.tsv.gz."""
        filepath = self.data_dir / "meddra_all_se.tsv.gz"
        if not filepath.exists():
            logger.warning("SIDER side effects file not found")
            return
        
        logger.info("Loading SIDER side effects...")
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        stitch_id = parts[0]  # STITCH compound ID (flat)
                        # stitch_stereo = parts[1]  # STITCH compound ID (stereo)
                        umls_cui = parts[2]  # UMLS concept ID
                        meddra_type = parts[3]  # MedDRA type (PT/LLT)
                        # umls_cui_meddra = parts[4]  # UMLS CUI for MedDRA
                        side_effect_name = parts[5]  # Side effect name
                        
                        # Convert STITCH flat ID to our format
                        # STITCH flat IDs are like CID000000001
                        if stitch_id.startswith('CID'):
                            drugbank_id = self.stitch_to_drugbank.get(stitch_id)
                            
                            if drugbank_id:
                                # Add side effect node
                                if umls_cui not in self.side_effects:
                                    self.side_effects[umls_cui] = SideEffectNode(
                                        id=umls_cui,
                                        name=side_effect_name,
                                        meddra_type=meddra_type,
                                    )
                                
                                # Add edge
                                self.drug_se_edges.append(DrugSideEffectEdge(
                                    drug_id=drugbank_id,
                                    side_effect_id=umls_cui,
                                    source="SIDER",
                                ))
            
            logger.info(f"  Loaded {len(self.side_effects)} unique side effects")
            logger.info(f"  Loaded {len(self.drug_se_edges)} drug-side effect associations")
            
        except Exception as e:
            logger.error(f"Error loading SIDER data: {e}")
    
    def load_frequencies(self) -> None:
        """Load frequency information from SIDER."""
        filepath = self.data_dir / "meddra_freq.tsv.gz"
        if not filepath.exists():
            return
        
        # Build lookup for existing edges
        edge_lookup = {}
        for i, edge in enumerate(self.drug_se_edges):
            key = (edge.drug_id, edge.side_effect_id)
            edge_lookup[key] = i
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        stitch_id = parts[0]
                        umls_cui = parts[2] 
                        freq = parts[4] if len(parts) > 4 else ""
                        
                        if stitch_id.startswith('CID'):
                            drugbank_id = self.stitch_to_drugbank.get(stitch_id)
                            if drugbank_id:
                                key = (drugbank_id, umls_cui)
                                if key in edge_lookup:
                                    self.drug_se_edges[edge_lookup[key]].frequency = freq
        except Exception as e:
            logger.warning(f"Error loading SIDER frequencies: {e}")


class STITCHIntegrator:
    """
    Integrates STITCH (Chemical-Protein Interactions) data.
    Download from: http://stitch.embl.de/download/
    """
    
    STITCH_BASE_URL = "http://stitch.embl.de/"
    
    def __init__(self, data_dir: str = "external_data/stitch"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.chemical_protein_edges: List[Tuple] = []
    
    def query_interactions(self, pubchem_ids: List[str], limit: int = 100) -> List[Dict]:
        """Query STITCH API for chemical-protein interactions."""
        interactions = []
        
        # STITCH API endpoint
        api_url = "http://stitch.embl.de/api/tsv/interactionsList"
        
        # Process in batches
        batch_size = 50
        for i in range(0, min(len(pubchem_ids), limit), batch_size):
            batch = pubchem_ids[i:i+batch_size]
            
            # Convert PubChem IDs to STITCH format
            stitch_ids = [f"CID1{pid.zfill(8)}" for pid in batch]
            
            params = {
                'identifiers': '%0d'.join(stitch_ids),
                'species': '9606',  # Human
                'limit': 10,
            }
            
            try:
                response = requests.get(api_url, params=params, timeout=30)
                if response.status_code == 200:
                    for line in response.text.strip().split('\n')[1:]:  # Skip header
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            interactions.append({
                                'chemical': parts[0],
                                'protein': parts[1],
                                'score': parts[2] if len(parts) > 2 else '',
                            })
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"STITCH API error: {e}")
                continue
        
        return interactions


class STRINGIntegrator:
    """
    Integrates STRING (Protein-Protein Interactions) database.
    Download from: https://string-db.org/cgi/download
    """
    
    STRING_API_URL = "https://string-db.org/api"
    
    def __init__(self, data_dir: str = "external_data/string"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.ppi_edges: List[ProteinProteinEdge] = []
    
    def query_interactions(self, uniprot_ids: List[str], score_threshold: int = 400) -> List[Dict]:
        """Query STRING API for protein-protein interactions."""
        interactions = []
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i+batch_size]
            
            api_url = f"{self.STRING_API_URL}/tsv/network"
            params = {
                'identifiers': '%0d'.join(batch),
                'species': 9606,  # Human
                'required_score': score_threshold,
                'caller_identity': 'ddi_kg_enrichment',
            }
            
            try:
                response = requests.get(api_url, params=params, timeout=60)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        header = lines[0].split('\t')
                        for line in lines[1:]:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                interactions.append({
                                    'protein1': parts[0],
                                    'protein2': parts[1],
                                    'score': int(parts[-1]) if parts[-1].isdigit() else 0,
                                })
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"STRING API error: {e}")
                continue
        
        logger.info(f"  Retrieved {len(interactions)} protein-protein interactions")
        return interactions
    
    def build_ppi_edges(self, interactions: List[Dict], uniprot_to_protein: Dict[str, str]) -> None:
        """Build PPI edges from STRING interactions."""
        for interaction in interactions:
            p1 = interaction['protein1']
            p2 = interaction['protein2']
            
            # Try to map to our protein IDs
            p1_id = uniprot_to_protein.get(p1)
            p2_id = uniprot_to_protein.get(p2)
            
            if p1_id and p2_id:
                self.ppi_edges.append(ProteinProteinEdge(
                    protein1_id=p1_id,
                    protein2_id=p2_id,
                    combined_score=interaction.get('score', 0),
                ))


class DisGeNETIntegrator:
    """
    Integrates DisGeNET (Disease-Gene Associations) database.
    Download from: https://www.disgenet.org/downloads
    
    Free version available for academic use.
    """
    
    def __init__(self, data_dir: str = "external_data/disgenet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.diseases: Dict[str, DiseaseNode] = {}
        self.protein_disease_edges: List[ProteinDiseaseEdge] = []
        self.gene_symbol_to_protein: Dict[str, str] = {}
    
    def build_gene_mapping(self, proteins: Dict) -> None:
        """Build mapping from gene symbols to protein IDs."""
        for protein_id, protein in proteins.items():
            if protein.gene_name:
                self.gene_symbol_to_protein[protein.gene_name.upper()] = protein_id
        
        logger.info(f"  Built gene symbol mapping for {len(self.gene_symbol_to_protein)} proteins")
    
    def load_curated_associations(self, filepath: Optional[str] = None) -> None:
        """
        Load gene-disease associations from DisGeNET curated file.
        File: curated_gene_disease_associations.tsv
        """
        if filepath is None:
            filepath = self.data_dir / "curated_gene_disease_associations.tsv"
        
        if not Path(filepath).exists():
            logger.warning(f"DisGeNET file not found: {filepath}")
            return
        
        logger.info("Loading DisGeNET associations...")
        
        try:
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
            
            for _, row in df.iterrows():
                gene_symbol = str(row.get('geneSymbol', '')).upper()
                disease_id = str(row.get('diseaseId', ''))
                disease_name = str(row.get('diseaseName', ''))
                score = float(row.get('score', 0))
                
                protein_id = self.gene_symbol_to_protein.get(gene_symbol)
                
                if protein_id and disease_id:
                    # Add disease node
                    if disease_id not in self.diseases:
                        self.diseases[disease_id] = DiseaseNode(
                            id=disease_id,
                            name=disease_name,
                            source="DisGeNET",
                        )
                    
                    # Add edge
                    self.protein_disease_edges.append(ProteinDiseaseEdge(
                        protein_id=protein_id,
                        disease_id=disease_id,
                        score=score,
                        source="DisGeNET",
                    ))
            
            logger.info(f"  Loaded {len(self.diseases)} diseases")
            logger.info(f"  Loaded {len(self.protein_disease_edges)} protein-disease associations")
            
        except Exception as e:
            logger.error(f"Error loading DisGeNET: {e}")


class CTDIntegrator:
    """
    Integrates CTD (Comparative Toxicogenomics Database).
    Download from: http://ctdbase.org/downloads/
    
    Files:
    - CTD_chemicals_diseases.tsv.gz
    - CTD_genes_diseases.tsv.gz
    - CTD_chem_gene_ixns.tsv.gz
    """
    
    CTD_BASE_URL = "http://ctdbase.org/reports/"
    
    def __init__(self, data_dir: str = "external_data/ctd"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.diseases: Dict[str, DiseaseNode] = {}
        self.drug_disease_edges: List[DrugDiseaseEdge] = []
        
        self.mesh_to_drugbank: Dict[str, str] = {}
        self.cas_to_drugbank: Dict[str, str] = {}
    
    def build_id_mapping(self, drugs: Dict) -> None:
        """Build mapping from MeSH/CAS IDs to DrugBank IDs."""
        for drug_id, drug in drugs.items():
            # CAS number mapping
            if drug.cas_number:
                self.cas_to_drugbank[drug.cas_number] = drug_id
        
        logger.info(f"  Built CAS number mapping for {len(self.cas_to_drugbank)} drugs")
    
    def download_files(self) -> bool:
        """Download CTD data files."""
        files = [
            "CTD_chemicals_diseases.tsv.gz",
        ]
        
        for filename in files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                url = self.CTD_BASE_URL + filename
                logger.info(f"Downloading {filename}...")
                try:
                    response = requests.get(url, timeout=120, stream=True)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logger.info(f"  Downloaded {filename}")
                    else:
                        logger.warning(f"  Failed: {response.status_code}")
                        return False
                except Exception as e:
                    logger.warning(f"  Error: {e}")
                    return False
        return True
    
    def load_chemical_disease(self) -> None:
        """Load chemical-disease associations from CTD."""
        filepath = self.data_dir / "CTD_chemicals_diseases.tsv.gz"
        if not filepath.exists():
            logger.warning("CTD chemical-disease file not found")
            return
        
        logger.info("Loading CTD chemical-disease associations...")
        
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        # chemical_name = parts[0]
                        # chemical_id = parts[1]  # MeSH ID
                        cas_rn = parts[2] if len(parts) > 2 else ""
                        disease_name = parts[3] if len(parts) > 3 else ""
                        disease_id = parts[4] if len(parts) > 4 else ""  # MeSH or OMIM
                        direct_evidence = parts[5] if len(parts) > 5 else ""
                        # inference_gene = parts[6] if len(parts) > 6 else ""
                        # inference_score = parts[7] if len(parts) > 7 else ""
                        
                        # Only include direct evidence
                        if direct_evidence and cas_rn:
                            drugbank_id = self.cas_to_drugbank.get(cas_rn)
                            
                            if drugbank_id and disease_id:
                                # Add disease node
                                if disease_id not in self.diseases:
                                    self.diseases[disease_id] = DiseaseNode(
                                        id=disease_id,
                                        name=disease_name,
                                        source="CTD",
                                    )
                                
                                # Add edge
                                self.drug_disease_edges.append(DrugDiseaseEdge(
                                    drug_id=drugbank_id,
                                    disease_id=disease_id,
                                    relationship_type=direct_evidence,
                                    source="CTD",
                                ))
            
            logger.info(f"  Loaded {len(self.diseases)} diseases")
            logger.info(f"  Loaded {len(self.drug_disease_edges)} drug-disease associations")
            
        except Exception as e:
            logger.error(f"Error loading CTD: {e}")


class GOAnnotationIntegrator:
    """
    Integrates Gene Ontology annotations.
    Uses UniProt GO annotations or QuickGO API.
    """
    
    QUICKGO_API = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch"
    
    def __init__(self, data_dir: str = "external_data/go"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.go_terms: Dict[str, GOTermNode] = {}
        self.protein_go_edges: List[ProteinGOEdge] = []
    
    def query_annotations(self, uniprot_ids: List[str], limit: int = 1000) -> List[Dict]:
        """Query QuickGO API for GO annotations."""
        annotations = []
        
        # Process in batches
        batch_size = 100
        for i in range(0, min(len(uniprot_ids), limit), batch_size):
            batch = uniprot_ids[i:i+batch_size]
            
            params = {
                'geneProductId': ','.join(batch),
                'taxonId': '9606',
                'geneProductType': 'protein',
                'limit': 1000,
            }
            
            headers = {'Accept': 'text/tsv'}
            
            try:
                response = requests.get(self.QUICKGO_API, params=params, headers=headers, timeout=60)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        for line in lines[1:]:
                            parts = line.split('\t')
                            if len(parts) >= 7:
                                annotations.append({
                                    'uniprot_id': parts[1],
                                    'go_id': parts[4],
                                    'go_name': parts[5] if len(parts) > 5 else '',
                                    'aspect': parts[6] if len(parts) > 6 else '',
                                    'evidence': parts[7] if len(parts) > 7 else '',
                                })
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"QuickGO API error: {e}")
                continue
        
        logger.info(f"  Retrieved {len(annotations)} GO annotations")
        return annotations
    
    def build_go_edges(self, annotations: List[Dict], uniprot_to_protein: Dict[str, str]) -> None:
        """Build GO annotation edges."""
        for annot in annotations:
            uniprot_id = annot['uniprot_id']
            go_id = annot['go_id']
            
            protein_id = uniprot_to_protein.get(uniprot_id)
            
            if protein_id and go_id:
                # Add GO term node
                if go_id not in self.go_terms:
                    namespace_map = {'F': 'molecular_function', 'P': 'biological_process', 'C': 'cellular_component'}
                    self.go_terms[go_id] = GOTermNode(
                        id=go_id,
                        name=annot.get('go_name', ''),
                        namespace=namespace_map.get(annot.get('aspect', ''), ''),
                    )
                
                # Add edge
                self.protein_go_edges.append(ProteinGOEdge(
                    protein_id=protein_id,
                    go_id=go_id,
                    evidence_code=annot.get('evidence', ''),
                ))


# =============================================================================
# MAIN ENRICHMENT CLASS
# =============================================================================

class KnowledgeGraphEnricher:
    """
    Main class to enrich DDI Knowledge Graph with external databases.
    """
    
    def __init__(self, kg_path: str):
        """Load existing knowledge graph."""
        logger.info(f"Loading knowledge graph from {kg_path}")
        
        with open(kg_path, 'rb') as f:
            data = pickle.load(f)
        
        self.drugs = data['drugs']
        self.proteins = data['proteins']
        self.pathways = data['pathways']
        self.categories = data['categories']
        self.ddi_edges = data['ddi_edges']
        self.drug_protein_edges = data['drug_protein_edges']
        self.drug_pathway_edges = data['drug_pathway_edges']
        self.drug_category_edges = data['drug_category_edges']
        self.snp_effects = data.get('snp_effects', [])
        self.snp_adverse_reactions = data.get('snp_adverse_reactions', [])
        self.graph = data.get('graph')
        
        # New data from enrichment
        self.side_effects: Dict[str, SideEffectNode] = {}
        self.diseases: Dict[str, DiseaseNode] = {}  
        self.go_terms: Dict[str, GOTermNode] = {}
        
        self.drug_se_edges: List[DrugSideEffectEdge] = []
        self.drug_disease_edges: List[DrugDiseaseEdge] = []
        self.protein_disease_edges: List[ProteinDiseaseEdge] = []
        self.protein_go_edges: List[ProteinGOEdge] = []
        self.ppi_edges: List[ProteinProteinEdge] = []
        
        # Build lookup maps
        self.uniprot_to_protein: Dict[str, str] = {}
        for protein_id, protein in self.proteins.items():
            if protein.uniprot_id:
                self.uniprot_to_protein[protein.uniprot_id] = protein_id
        
        logger.info(f"  Loaded {len(self.drugs)} drugs, {len(self.proteins)} proteins")
    
    def enrich_with_sider(self, download: bool = True) -> None:
        """Enrich with SIDER side effect data."""
        logger.info("\n--- Enriching with SIDER ---")
        
        sider = SIDERIntegrator()
        
        if download:
            if not sider.download_files():
                logger.warning("Could not download SIDER files, skipping...")
                return
        
        sider.build_id_mapping(self.drugs)
        sider.load_side_effects()
        sider.load_frequencies()
        
        self.side_effects.update(sider.side_effects)
        self.drug_se_edges.extend(sider.drug_se_edges)
    
    def enrich_with_string(self, score_threshold: int = 700) -> None:
        """Enrich with STRING protein-protein interactions."""
        logger.info("\n--- Enriching with STRING ---")
        
        string = STRINGIntegrator()
        
        # Get UniProt IDs
        uniprot_ids = [p.uniprot_id for p in self.proteins.values() if p.uniprot_id]
        logger.info(f"  Querying STRING for {len(uniprot_ids)} proteins...")
        
        if len(uniprot_ids) > 0:
            # Limit to avoid API overload
            interactions = string.query_interactions(uniprot_ids[:500], score_threshold)
            string.build_ppi_edges(interactions, self.uniprot_to_protein)
            self.ppi_edges.extend(string.ppi_edges)
    
    def enrich_with_disgenet(self) -> None:
        """Enrich with DisGeNET disease associations."""
        logger.info("\n--- Enriching with DisGeNET ---")
        
        disgenet = DisGeNETIntegrator()
        disgenet.build_gene_mapping(self.proteins)
        disgenet.load_curated_associations()
        
        self.diseases.update(disgenet.diseases)
        self.protein_disease_edges.extend(disgenet.protein_disease_edges)
    
    def enrich_with_ctd(self, download: bool = True) -> None:
        """Enrich with CTD chemical-disease associations."""
        logger.info("\n--- Enriching with CTD ---")
        
        ctd = CTDIntegrator()
        
        if download:
            ctd.download_files()
        
        ctd.build_id_mapping(self.drugs)
        ctd.load_chemical_disease()
        
        # Merge diseases (avoid duplicates)
        for disease_id, disease in ctd.diseases.items():
            if disease_id not in self.diseases:
                self.diseases[disease_id] = disease
        
        self.drug_disease_edges.extend(ctd.drug_disease_edges)
    
    def enrich_with_go(self, limit: int = 500) -> None:
        """Enrich with Gene Ontology annotations."""
        logger.info("\n--- Enriching with GO annotations ---")
        
        go_integrator = GOAnnotationIntegrator()
        
        # Get UniProt IDs
        uniprot_ids = [p.uniprot_id for p in self.proteins.values() if p.uniprot_id]
        logger.info(f"  Querying QuickGO for {min(len(uniprot_ids), limit)} proteins...")
        
        if len(uniprot_ids) > 0:
            annotations = go_integrator.query_annotations(uniprot_ids[:limit])
            go_integrator.build_go_edges(annotations, self.uniprot_to_protein)
            
            self.go_terms.update(go_integrator.go_terms)
            self.protein_go_edges.extend(go_integrator.protein_go_edges)
    
    def build_enriched_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph with all enriched data."""
        logger.info("\n--- Building Enriched NetworkX Graph ---")
        
        # Start with existing graph or create new
        if self.graph is not None:
            G = self.graph.copy()
        else:
            G = nx.MultiDiGraph()
            
            # Add existing nodes
            for drug_id, drug in self.drugs.items():
                G.add_node(drug_id, type='drug', name=drug.name)
            
            for protein_id, protein in self.proteins.items():
                G.add_node(protein_id, type='protein', name=protein.name)
        
        # Add new node types
        for se_id, se in self.side_effects.items():
            G.add_node(f"SE:{se_id}", type='side_effect', name=se.name, meddra_type=se.meddra_type)
        
        for disease_id, disease in self.diseases.items():
            G.add_node(f"DIS:{disease_id}", type='disease', name=disease.name, source=disease.source)
        
        for go_id, go_term in self.go_terms.items():
            G.add_node(go_id, type='go_term', name=go_term.name, namespace=go_term.namespace)
        
        # Add new edges
        for edge in self.drug_se_edges:
            if edge.drug_id in self.drugs:
                G.add_edge(edge.drug_id, f"SE:{edge.side_effect_id}", 
                          type='causes_side_effect', frequency=edge.frequency)
        
        for edge in self.drug_disease_edges:
            if edge.drug_id in self.drugs:
                G.add_edge(edge.drug_id, f"DIS:{edge.disease_id}",
                          type='drug_disease', relationship=edge.relationship_type)
        
        for edge in self.protein_disease_edges:
            if edge.protein_id in self.proteins:
                G.add_edge(edge.protein_id, f"DIS:{edge.disease_id}",
                          type='protein_disease', score=edge.score)
        
        for edge in self.protein_go_edges:
            if edge.protein_id in self.proteins:
                G.add_edge(edge.protein_id, edge.go_id,
                          type='has_function', evidence=edge.evidence_code)
        
        for edge in self.ppi_edges:
            if edge.protein1_id in self.proteins and edge.protein2_id in self.proteins:
                G.add_edge(edge.protein1_id, edge.protein2_id,
                          type='interacts_with', score=edge.combined_score)
        
        self.graph = G
        
        logger.info(f"  Total nodes: {G.number_of_nodes()}")
        logger.info(f"  Total edges: {G.number_of_edges()}")
        
        return G
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            'nodes': {
                'drugs': len(self.drugs),
                'proteins': len(self.proteins),
                'pathways': len(self.pathways),
                'categories': len(self.categories),
                'side_effects': len(self.side_effects),
                'diseases': len(self.diseases),
                'go_terms': len(self.go_terms),
            },
            'edges': {
                'ddi': len(self.ddi_edges),
                'drug_protein': len(self.drug_protein_edges),
                'drug_pathway': len(self.drug_pathway_edges),
                'drug_category': len(self.drug_category_edges),
                'drug_side_effect': len(self.drug_se_edges),
                'drug_disease': len(self.drug_disease_edges),
                'protein_disease': len(self.protein_disease_edges),
                'protein_go': len(self.protein_go_edges),
                'protein_protein': len(self.ppi_edges),
            },
            'snp_data': {
                'snp_effects': len(self.snp_effects),
                'snp_adverse_reactions': len(self.snp_adverse_reactions),
            },
        }
        
        stats['nodes']['total'] = sum(stats['nodes'].values())
        stats['edges']['total'] = sum(stats['edges'].values())
        
        return stats
    
    def export_enriched(self, output_dir: str) -> None:
        """Export enriched knowledge graph."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nExporting enriched KG to {output_dir}")
        
        # Export side effects
        if self.side_effects:
            se_data = [{'id': se.id, 'name': se.name, 'meddra_type': se.meddra_type} 
                      for se in self.side_effects.values()]
            pd.DataFrame(se_data).to_csv(output_path / 'side_effects.csv', index=False)
        
        # Export diseases
        if self.diseases:
            disease_data = [{'id': d.id, 'name': d.name, 'source': d.source}
                          for d in self.diseases.values()]
            pd.DataFrame(disease_data).to_csv(output_path / 'diseases.csv', index=False)
        
        # Export GO terms
        if self.go_terms:
            go_data = [{'id': g.id, 'name': g.name, 'namespace': g.namespace}
                      for g in self.go_terms.values()]
            pd.DataFrame(go_data).to_csv(output_path / 'go_terms.csv', index=False)
        
        # Export drug-side effect edges
        if self.drug_se_edges:
            dse_data = [{'drug_id': e.drug_id, 'side_effect_id': e.side_effect_id, 
                        'frequency': e.frequency, 'source': e.source}
                       for e in self.drug_se_edges]
            pd.DataFrame(dse_data).to_csv(output_path / 'drug_side_effect_edges.csv', index=False)
        
        # Export drug-disease edges
        if self.drug_disease_edges:
            dd_data = [{'drug_id': e.drug_id, 'disease_id': e.disease_id,
                       'relationship_type': e.relationship_type, 'source': e.source}
                      for e in self.drug_disease_edges]
            pd.DataFrame(dd_data).to_csv(output_path / 'drug_disease_edges.csv', index=False)
        
        # Export protein-disease edges
        if self.protein_disease_edges:
            pd_data = [{'protein_id': e.protein_id, 'disease_id': e.disease_id,
                       'score': e.score, 'source': e.source}
                      for e in self.protein_disease_edges]
            pd.DataFrame(pd_data).to_csv(output_path / 'protein_disease_edges.csv', index=False)
        
        # Export protein-GO edges
        if self.protein_go_edges:
            pgo_data = [{'protein_id': e.protein_id, 'go_id': e.go_id,
                        'evidence_code': e.evidence_code}
                       for e in self.protein_go_edges]
            pd.DataFrame(pgo_data).to_csv(output_path / 'protein_go_edges.csv', index=False)
        
        # Export PPI edges
        if self.ppi_edges:
            ppi_data = [{'protein1_id': e.protein1_id, 'protein2_id': e.protein2_id,
                        'combined_score': e.combined_score}
                       for e in self.ppi_edges]
            pd.DataFrame(ppi_data).to_csv(output_path / 'ppi_edges.csv', index=False)
        
        # Save statistics
        stats = self.get_statistics()
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save enriched pickle
        with open(output_path / 'enriched_kg.pkl', 'wb') as f:
            pickle.dump({
                'drugs': self.drugs,
                'proteins': self.proteins,
                'pathways': self.pathways,
                'categories': self.categories,
                'side_effects': self.side_effects,
                'diseases': self.diseases,
                'go_terms': self.go_terms,
                'ddi_edges': self.ddi_edges,
                'drug_protein_edges': self.drug_protein_edges,
                'drug_pathway_edges': self.drug_pathway_edges,
                'drug_category_edges': self.drug_category_edges,
                'drug_se_edges': self.drug_se_edges,
                'drug_disease_edges': self.drug_disease_edges,
                'protein_disease_edges': self.protein_disease_edges,
                'protein_go_edges': self.protein_go_edges,
                'ppi_edges': self.ppi_edges,
                'snp_effects': self.snp_effects,
                'snp_adverse_reactions': self.snp_adverse_reactions,
                'graph': self.graph,
            }, f)
        
        logger.info("  Export complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Enrich DDI Knowledge Graph with external databases."""
    
    kg_path = "knowledge_graph_real/knowledge_graph.pkl"
    output_dir = "knowledge_graph_enriched_full"
    
    # Initialize enricher
    enricher = KnowledgeGraphEnricher(kg_path)
    
    # Run enrichments (comment out any that fail or you don't want)
    try:
        enricher.enrich_with_sider(download=True)
    except Exception as e:
        logger.warning(f"SIDER enrichment failed: {e}")
    
    try:
        enricher.enrich_with_ctd(download=True)
    except Exception as e:
        logger.warning(f"CTD enrichment failed: {e}")
    
    try:
        enricher.enrich_with_string(score_threshold=700)
    except Exception as e:
        logger.warning(f"STRING enrichment failed: {e}")
    
    try:
        enricher.enrich_with_go(limit=300)
    except Exception as e:
        logger.warning(f"GO enrichment failed: {e}")
    
    # Build final graph
    enricher.build_enriched_graph()
    
    # Get statistics
    stats = enricher.get_statistics()
    
    print("\n" + "=" * 60)
    print("ENRICHED KNOWLEDGE GRAPH STATISTICS")
    print("=" * 60)
    
    print("\nNode Counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nEdge Counts:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    print("\nSNP Data:")
    for snp_type, count in stats['snp_data'].items():
        print(f"  {snp_type}: {count:,}")
    
    # Export
    enricher.export_enriched(output_dir)
    
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == '__main__':
    main()
