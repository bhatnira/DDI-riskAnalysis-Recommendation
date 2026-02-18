#!/usr/bin/env python3
"""
Comprehensive DDI Knowledge Graph from DrugBank XML.

Extracts ALL available information from DrugBank XML database:
- Drugs with full pharmacological properties
- Targets (protein targets)
- Enzymes (metabolizing enzymes like CYP450s)
- Carriers (drug carrier proteins)
- Transporters (drug transporter proteins)
- Pathways (SMPDB pathways)
- Categories (MeSH therapeutic categories)
- Drug-drug interactions
- SNP effects and adverse reactions
- External identifiers (PubChem, KEGG, UniProt, etc.)
- Calculated properties (SMILES, InChI, logP, etc.)
- ATC codes with full hierarchy
- Food interactions

This creates a truly comprehensive knowledge graph from real DrugBank data.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pandas as pd
import networkx as nx
import pickle
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DrugBank XML namespace
NS = {'db': 'http://www.drugbank.ca'}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DrugNode:
    """Full drug information from DrugBank."""
    drugbank_id: str
    name: str
    type: str = ""  # small molecule or biotech
    description: str = ""
    cas_number: str = ""
    unii: str = ""
    state: str = ""  # solid/liquid/gas
    groups: List[str] = field(default_factory=list)  # approved, withdrawn, etc.
    
    # Pharmacology
    indication: str = ""
    pharmacodynamics: str = ""
    mechanism_of_action: str = ""
    toxicity: str = ""
    metabolism: str = ""
    absorption: str = ""
    half_life: str = ""
    protein_binding: str = ""
    route_of_elimination: str = ""
    volume_of_distribution: str = ""
    clearance: str = ""
    
    # Classification
    kingdom: str = ""
    superclass: str = ""
    drug_class: str = ""
    subclass: str = ""
    direct_parent: str = ""
    
    # ATC codes
    atc_codes: List[str] = field(default_factory=list)
    
    # External identifiers
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    # Calculated properties
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    molecular_weight: str = ""
    logp: str = ""
    molecular_formula: str = ""
    psa: str = ""  # polar surface area
    
    # Food interactions
    food_interactions: List[str] = field(default_factory=list)


@dataclass
class ProteinNode:
    """Protein target, enzyme, carrier, or transporter."""
    id: str  # DrugBank internal ID
    name: str
    type: str  # target, enzyme, carrier, transporter
    organism: str = ""
    gene_name: str = ""
    uniprot_id: str = ""
    general_function: str = ""
    specific_function: str = ""
    cellular_location: str = ""
    chromosome_location: str = ""
    molecular_weight: str = ""
    
    # For enzymes
    inhibition_strength: str = ""
    induction_strength: str = ""
    
    # Actions (for targets)
    actions: List[str] = field(default_factory=list)
    known_action: str = ""


@dataclass
class PathwayNode:
    """SMPDB pathway information."""
    smpdb_id: str
    name: str
    category: str = ""
    drugs: List[str] = field(default_factory=list)  # DrugBank IDs
    enzymes: List[str] = field(default_factory=list)  # UniProt IDs


@dataclass
class CategoryNode:
    """MeSH therapeutic category."""
    name: str
    mesh_id: str = ""


@dataclass
class SNPEffect:
    """SNP effect on drug response."""
    protein_name: str
    gene_symbol: str
    uniprot_id: str
    rs_id: str
    allele: str
    defining_change: str
    description: str
    pubmed_id: str = ""


@dataclass
class SNPAdverseReaction:
    """SNP-related adverse drug reaction."""
    protein_name: str
    gene_symbol: str
    uniprot_id: str
    rs_id: str
    allele: str
    adverse_reaction: str
    description: str
    pubmed_id: str = ""


@dataclass
class DDIEdge:
    """Drug-drug interaction from DrugBank."""
    drug1_id: str
    drug2_id: str
    drug2_name: str
    description: str


@dataclass
class DrugProteinEdge:
    """Drug-protein relationship."""
    drug_id: str
    protein_id: str
    type: str  # target, enzyme, carrier, transporter
    actions: List[str] = field(default_factory=list)
    known_action: str = ""
    inhibition_strength: str = ""
    induction_strength: str = ""


@dataclass
class DrugPathwayEdge:
    """Drug-pathway relationship."""
    drug_id: str
    pathway_id: str


@dataclass
class DrugCategoryEdge:
    """Drug-category relationship."""
    drug_id: str
    category_name: str


# =============================================================================
# XML PARSER - STREAMING FOR LARGE FILES
# =============================================================================

class DrugBankXMLParser:
    """
    Streaming XML parser for DrugBank database.
    Uses iterparse to handle large files efficiently.
    """
    
    def __init__(self, xml_path: str, cv_drug_ids: Optional[Set[str]] = None):
        """
        Initialize parser.
        
        Args:
            xml_path: Path to DrugBank XML file
            cv_drug_ids: Optional set of cardiovascular drug IDs to filter
        """
        self.xml_path = xml_path
        self.cv_drug_ids = cv_drug_ids
        
        # Storage
        self.drugs: Dict[str, DrugNode] = {}
        self.proteins: Dict[str, ProteinNode] = {}
        self.pathways: Dict[str, PathwayNode] = {}
        self.categories: Dict[str, CategoryNode] = {}
        self.snp_effects: List[SNPEffect] = []
        self.snp_adverse_reactions: List[SNPAdverseReaction] = []
        
        # Edges
        self.ddi_edges: List[DDIEdge] = []
        self.drug_protein_edges: List[DrugProteinEdge] = []
        self.drug_pathway_edges: List[DrugPathwayEdge] = []
        self.drug_category_edges: List[DrugCategoryEdge] = []
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _get_text(self, elem, tag: str, ns: dict = NS) -> str:
        """Safely get text from XML element."""
        child = elem.find(f'db:{tag}', ns)
        if child is not None and child.text:
            return child.text.strip()
        return ""
    
    def _get_all_text(self, elem, tag: str, ns: dict = NS) -> List[str]:
        """Get text from all matching child elements."""
        result = []
        for child in elem.findall(f'db:{tag}', ns):
            if child.text:
                result.append(child.text.strip())
        return result
    
    def _parse_drug(self, drug_elem) -> Optional[DrugNode]:
        """Parse a single drug element."""
        # Get primary DrugBank ID
        drugbank_id = ""
        for id_elem in drug_elem.findall('db:drugbank-id', NS):
            if id_elem.get('primary') == 'true':
                drugbank_id = id_elem.text.strip() if id_elem.text else ""
                break
        
        if not drugbank_id:
            return None
        
        # Filter for CV drugs if specified
        if self.cv_drug_ids and drugbank_id not in self.cv_drug_ids:
            return None
        
        drug = DrugNode(
            drugbank_id=drugbank_id,
            name=self._get_text(drug_elem, 'name'),
            type=drug_elem.get('type', ''),
            description=self._get_text(drug_elem, 'description'),
            cas_number=self._get_text(drug_elem, 'cas-number'),
            unii=self._get_text(drug_elem, 'unii'),
            state=self._get_text(drug_elem, 'state'),
        )
        
        # Groups
        groups_elem = drug_elem.find('db:groups', NS)
        if groups_elem is not None:
            drug.groups = self._get_all_text(groups_elem, 'group')
        
        # Pharmacology
        drug.indication = self._get_text(drug_elem, 'indication')
        drug.pharmacodynamics = self._get_text(drug_elem, 'pharmacodynamics')
        drug.mechanism_of_action = self._get_text(drug_elem, 'mechanism-of-action')
        drug.toxicity = self._get_text(drug_elem, 'toxicity')
        drug.metabolism = self._get_text(drug_elem, 'metabolism')
        drug.absorption = self._get_text(drug_elem, 'absorption')
        drug.half_life = self._get_text(drug_elem, 'half-life')
        drug.protein_binding = self._get_text(drug_elem, 'protein-binding')
        drug.route_of_elimination = self._get_text(drug_elem, 'route-of-elimination')
        drug.volume_of_distribution = self._get_text(drug_elem, 'volume-of-distribution')
        drug.clearance = self._get_text(drug_elem, 'clearance')
        
        # Classification
        class_elem = drug_elem.find('db:classification', NS)
        if class_elem is not None:
            drug.kingdom = self._get_text(class_elem, 'kingdom')
            drug.superclass = self._get_text(class_elem, 'superclass')
            drug.drug_class = self._get_text(class_elem, 'class')
            drug.subclass = self._get_text(class_elem, 'subclass')
            drug.direct_parent = self._get_text(class_elem, 'direct-parent')
        
        # ATC codes
        atc_elem = drug_elem.find('db:atc-codes', NS)
        if atc_elem is not None:
            for code in atc_elem.findall('db:atc-code', NS):
                atc_code = code.get('code', '')
                if atc_code:
                    drug.atc_codes.append(atc_code)
        
        # External identifiers
        ext_ids_elem = drug_elem.find('db:external-identifiers', NS)
        if ext_ids_elem is not None:
            for ext_id in ext_ids_elem.findall('db:external-identifier', NS):
                resource = self._get_text(ext_id, 'resource')
                identifier = self._get_text(ext_id, 'identifier')
                if resource and identifier:
                    drug.external_ids[resource] = identifier
        
        # Calculated properties
        calc_props = drug_elem.find('db:calculated-properties', NS)
        if calc_props is not None:
            for prop in calc_props.findall('db:property', NS):
                kind = self._get_text(prop, 'kind')
                value = self._get_text(prop, 'value')
                if kind == 'SMILES':
                    drug.smiles = value
                elif kind == 'InChI':
                    drug.inchi = value
                elif kind == 'InChIKey':
                    drug.inchi_key = value
                elif kind == 'Molecular Weight':
                    drug.molecular_weight = value
                elif kind == 'logP':
                    drug.logp = value
                elif kind == 'Molecular Formula':
                    drug.molecular_formula = value
                elif kind == 'Polar Surface Area (PSA)':
                    drug.psa = value
        
        # Food interactions
        food_elem = drug_elem.find('db:food-interactions', NS)
        if food_elem is not None:
            drug.food_interactions = self._get_all_text(food_elem, 'food-interaction')
        
        return drug
    
    def _parse_protein(self, elem, protein_type: str, drug_id: str) -> Optional[ProteinNode]:
        """Parse target/enzyme/carrier/transporter element."""
        protein_id = self._get_text(elem, 'id')
        if not protein_id:
            return None
        
        # Check if we already have this protein
        if protein_id in self.proteins:
            protein = self.proteins[protein_id]
        else:
            protein = ProteinNode(
                id=protein_id,
                name=self._get_text(elem, 'name'),
                type=protein_type,
                organism=self._get_text(elem, 'organism'),
                known_action=self._get_text(elem, 'known-action'),
            )
            
            # Get actions
            actions_elem = elem.find('db:actions', NS)
            if actions_elem is not None:
                protein.actions = self._get_all_text(actions_elem, 'action')
            
            # For enzymes, get inhibition/induction strength
            if protein_type == 'enzyme':
                protein.inhibition_strength = self._get_text(elem, 'inhibition-strength')
                protein.induction_strength = self._get_text(elem, 'induction-strength')
            
            # Get polypeptide info (first one if multiple)
            polypeptide = elem.find('db:polypeptide', NS)
            if polypeptide is not None:
                protein.gene_name = self._get_text(polypeptide, 'gene-name')
                protein.general_function = self._get_text(polypeptide, 'general-function')
                protein.specific_function = self._get_text(polypeptide, 'specific-function')
                protein.cellular_location = self._get_text(polypeptide, 'cellular-location')
                protein.chromosome_location = self._get_text(polypeptide, 'chromosome-location')
                protein.molecular_weight = self._get_text(polypeptide, 'molecular-weight')
                
                # Get UniProt ID from polypeptide external identifiers
                ext_ids = polypeptide.find('db:external-identifiers', NS)
                if ext_ids is not None:
                    for ext_id in ext_ids.findall('db:external-identifier', NS):
                        resource = self._get_text(ext_id, 'resource')
                        if resource in ['UniProtKB', 'UniProt Accession']:
                            protein.uniprot_id = self._get_text(ext_id, 'identifier')
                            break
            
            self.proteins[protein_id] = protein
        
        # Create drug-protein edge
        edge = DrugProteinEdge(
            drug_id=drug_id,
            protein_id=protein_id,
            type=protein_type,
            actions=protein.actions.copy(),
            known_action=protein.known_action,
        )
        if protein_type == 'enzyme':
            edge.inhibition_strength = protein.inhibition_strength
            edge.induction_strength = protein.induction_strength
        
        self.drug_protein_edges.append(edge)
        return protein
    
    def _parse_pathway(self, elem, drug_id: str) -> Optional[PathwayNode]:
        """Parse pathway element."""
        smpdb_id = self._get_text(elem, 'smpdb-id')
        if not smpdb_id:
            return None
        
        if smpdb_id in self.pathways:
            pathway = self.pathways[smpdb_id]
            if drug_id not in pathway.drugs:
                pathway.drugs.append(drug_id)
        else:
            pathway = PathwayNode(
                smpdb_id=smpdb_id,
                name=self._get_text(elem, 'name'),
                category=self._get_text(elem, 'category'),
                drugs=[drug_id],
            )
            
            # Get enzymes (UniProt IDs)
            enzymes_elem = elem.find('db:enzymes', NS)
            if enzymes_elem is not None:
                pathway.enzymes = self._get_all_text(enzymes_elem, 'uniprot-id')
            
            self.pathways[smpdb_id] = pathway
        
        # Create drug-pathway edge
        self.drug_pathway_edges.append(DrugPathwayEdge(drug_id=drug_id, pathway_id=smpdb_id))
        return pathway
    
    def _parse_category(self, elem, drug_id: str) -> CategoryNode:
        """Parse category element."""
        cat_name = self._get_text(elem, 'category')
        mesh_id = self._get_text(elem, 'mesh-id')
        
        if cat_name not in self.categories:
            self.categories[cat_name] = CategoryNode(name=cat_name, mesh_id=mesh_id)
        
        # Create drug-category edge
        self.drug_category_edges.append(DrugCategoryEdge(drug_id=drug_id, category_name=cat_name))
        return self.categories[cat_name]
    
    def _parse_ddi(self, elem, drug_id: str) -> DDIEdge:
        """Parse drug-drug interaction element."""
        drug2_id = ""
        for id_elem in elem.findall('db:drugbank-id', NS):
            drug2_id = id_elem.text.strip() if id_elem.text else ""
            break
        
        return DDIEdge(
            drug1_id=drug_id,
            drug2_id=drug2_id,
            drug2_name=self._get_text(elem, 'name'),
            description=self._get_text(elem, 'description'),
        )
    
    def _parse_snp_effect(self, elem) -> SNPEffect:
        """Parse SNP effect element."""
        return SNPEffect(
            protein_name=self._get_text(elem, 'protein-name'),
            gene_symbol=self._get_text(elem, 'gene-symbol'),
            uniprot_id=self._get_text(elem, 'uniprot-id'),
            rs_id=self._get_text(elem, 'rs-id'),
            allele=self._get_text(elem, 'allele'),
            defining_change=self._get_text(elem, 'defining-change'),
            description=self._get_text(elem, 'description'),
            pubmed_id=self._get_text(elem, 'pubmed-id'),
        )
    
    def _parse_snp_adr(self, elem) -> SNPAdverseReaction:
        """Parse SNP adverse drug reaction element."""
        return SNPAdverseReaction(
            protein_name=self._get_text(elem, 'protein-name'),
            gene_symbol=self._get_text(elem, 'gene-symbol'),
            uniprot_id=self._get_text(elem, 'uniprot-id'),
            rs_id=self._get_text(elem, 'rs-id'),
            allele=self._get_text(elem, 'allele'),
            adverse_reaction=self._get_text(elem, 'adverse-reaction'),
            description=self._get_text(elem, 'description'),
            pubmed_id=self._get_text(elem, 'pubmed-id'),
        )
    
    def parse(self) -> None:
        """Parse the DrugBank XML file using streaming."""
        logger.info(f"Parsing DrugBank XML: {self.xml_path}")
        logger.info("This may take several minutes for large files...")
        
        # Use iterparse for memory efficiency
        context = ET.iterparse(self.xml_path, events=('end',))
        
        drug_count = 0
        for event, elem in context:
            # Only process drug elements at the top level
            if elem.tag == '{http://www.drugbank.ca}drug' and elem.getparent() is not None:
                # Check if parent is drugbank (top-level drug)
                parent = elem.getparent()
                if parent is not None and parent.tag == '{http://www.drugbank.ca}drugbank':
                    drug = self._parse_drug(elem)
                    
                    if drug:
                        self.drugs[drug.drugbank_id] = drug
                        drug_count += 1
                        
                        # Parse targets
                        targets_elem = elem.find('db:targets', NS)
                        if targets_elem is not None:
                            for target in targets_elem.findall('db:target', NS):
                                self._parse_protein(target, 'target', drug.drugbank_id)
                                self.stats['targets'] += 1
                        
                        # Parse enzymes
                        enzymes_elem = elem.find('db:enzymes', NS)
                        if enzymes_elem is not None:
                            for enzyme in enzymes_elem.findall('db:enzyme', NS):
                                self._parse_protein(enzyme, 'enzyme', drug.drugbank_id)
                                self.stats['enzymes'] += 1
                        
                        # Parse carriers
                        carriers_elem = elem.find('db:carriers', NS)
                        if carriers_elem is not None:
                            for carrier in carriers_elem.findall('db:carrier', NS):
                                self._parse_protein(carrier, 'carrier', drug.drugbank_id)
                                self.stats['carriers'] += 1
                        
                        # Parse transporters
                        transporters_elem = elem.find('db:transporters', NS)
                        if transporters_elem is not None:
                            for transporter in transporters_elem.findall('db:transporter', NS):
                                self._parse_protein(transporter, 'transporter', drug.drugbank_id)
                                self.stats['transporters'] += 1
                        
                        # Parse pathways
                        pathways_elem = elem.find('db:pathways', NS)
                        if pathways_elem is not None:
                            for pathway in pathways_elem.findall('db:pathway', NS):
                                self._parse_pathway(pathway, drug.drugbank_id)
                                self.stats['pathways'] += 1
                        
                        # Parse categories
                        categories_elem = elem.find('db:categories', NS)
                        if categories_elem is not None:
                            for category in categories_elem.findall('db:category', NS):
                                self._parse_category(category, drug.drugbank_id)
                                self.stats['categories'] += 1
                        
                        # Parse drug-drug interactions
                        ddi_elem = elem.find('db:drug-interactions', NS)
                        if ddi_elem is not None:
                            for ddi in ddi_elem.findall('db:drug-interaction', NS):
                                edge = self._parse_ddi(ddi, drug.drugbank_id)
                                if edge.drug2_id:
                                    self.ddi_edges.append(edge)
                                    self.stats['ddi'] += 1
                        
                        # Parse SNP effects
                        snp_effects_elem = elem.find('db:snp-effects', NS)
                        if snp_effects_elem is not None:
                            for effect in snp_effects_elem.findall('db:effect', NS):
                                self.snp_effects.append(self._parse_snp_effect(effect))
                                self.stats['snp_effects'] += 1
                        
                        # Parse SNP adverse reactions
                        snp_adr_elem = elem.find('db:snp-adverse-drug-reactions', NS)
                        if snp_adr_elem is not None:
                            for reaction in snp_adr_elem.findall('db:reaction', NS):
                                self.snp_adverse_reactions.append(self._parse_snp_adr(reaction))
                                self.stats['snp_adr'] += 1
                        
                        if drug_count % 500 == 0:
                            logger.info(f"  Processed {drug_count} drugs...")
                    
                    # Clear element to free memory
                    elem.clear()
        
        logger.info(f"Parsing complete! Found {len(self.drugs)} drugs")
    
    def parse_simple(self) -> None:
        """Simple parse without streaming (for smaller files or filtering)."""
        logger.info(f"Parsing DrugBank XML: {self.xml_path}")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        drug_count = 0
        for drug_elem in root.findall('db:drug', NS):
            drug = self._parse_drug(drug_elem)
            
            if drug:
                self.drugs[drug.drugbank_id] = drug
                drug_count += 1
                
                # Parse targets
                targets_elem = drug_elem.find('db:targets', NS)
                if targets_elem is not None:
                    for target in targets_elem.findall('db:target', NS):
                        self._parse_protein(target, 'target', drug.drugbank_id)
                        self.stats['targets'] += 1
                
                # Parse enzymes
                enzymes_elem = drug_elem.find('db:enzymes', NS)
                if enzymes_elem is not None:
                    for enzyme in enzymes_elem.findall('db:enzyme', NS):
                        self._parse_protein(enzyme, 'enzyme', drug.drugbank_id)
                        self.stats['enzymes'] += 1
                
                # Parse carriers
                carriers_elem = drug_elem.find('db:carriers', NS)
                if carriers_elem is not None:
                    for carrier in carriers_elem.findall('db:carrier', NS):
                        self._parse_protein(carrier, 'carrier', drug.drugbank_id)
                        self.stats['carriers'] += 1
                
                # Parse transporters
                transporters_elem = drug_elem.find('db:transporters', NS)
                if transporters_elem is not None:
                    for transporter in transporters_elem.findall('db:transporter', NS):
                        self._parse_protein(transporter, 'transporter', drug.drugbank_id)
                        self.stats['transporters'] += 1
                
                # Parse pathways
                pathways_elem = drug_elem.find('db:pathways', NS)
                if pathways_elem is not None:
                    for pathway in pathways_elem.findall('db:pathway', NS):
                        self._parse_pathway(pathway, drug.drugbank_id)
                        self.stats['pathways'] += 1
                
                # Parse categories
                categories_elem = drug_elem.find('db:categories', NS)
                if categories_elem is not None:
                    for category in categories_elem.findall('db:category', NS):
                        self._parse_category(category, drug.drugbank_id)
                        self.stats['categories'] += 1
                
                # Parse drug-drug interactions
                ddi_elem = drug_elem.find('db:drug-interactions', NS)
                if ddi_elem is not None:
                    for ddi in ddi_elem.findall('db:drug-interaction', NS):
                        edge = self._parse_ddi(ddi, drug.drugbank_id)
                        if edge.drug2_id:
                            self.ddi_edges.append(edge)
                            self.stats['ddi'] += 1
                
                # Parse SNP effects
                snp_effects_elem = drug_elem.find('db:snp-effects', NS)
                if snp_effects_elem is not None:
                    for effect in snp_effects_elem.findall('db:effect', NS):
                        self.snp_effects.append(self._parse_snp_effect(effect))
                        self.stats['snp_effects'] += 1
                
                # Parse SNP adverse reactions
                snp_adr_elem = drug_elem.find('db:snp-adverse-drug-reactions', NS)
                if snp_adr_elem is not None:
                    for reaction in snp_adr_elem.findall('db:reaction', NS):
                        self.snp_adverse_reactions.append(self._parse_snp_adr(reaction))
                        self.stats['snp_adr'] += 1
                
                if drug_count % 500 == 0:
                    logger.info(f"  Processed {drug_count} drugs...")
        
        logger.info(f"Parsing complete! Found {len(self.drugs)} drugs")


# =============================================================================
# KNOWLEDGE GRAPH BUILDER
# =============================================================================

class RealDDIKnowledgeGraph:
    """
    Comprehensive DDI Knowledge Graph built from DrugBank XML.
    """
    
    def __init__(self):
        self.parser: Optional[DrugBankXMLParser] = None
        self.graph: Optional[nx.MultiDiGraph] = None
        
        # CV drug filter
        self.cv_drug_ids: Set[str] = set()
    
    def load_cv_drug_ids(self, csv_path: str) -> None:
        """Load cardiovascular drug IDs from CSV."""
        logger.info(f"Loading CV drug IDs from {csv_path}")
        df = pd.read_csv(csv_path)
        
        self.cv_drug_ids = set(df['drugbank_id_1'].unique()) | set(df['drugbank_id_2'].unique())
        logger.info(f"  Found {len(self.cv_drug_ids)} unique CV drug IDs")
    
    def parse_drugbank(self, xml_path: str, filter_cv: bool = True) -> None:
        """Parse DrugBank XML and extract all information."""
        cv_ids = self.cv_drug_ids if filter_cv and self.cv_drug_ids else None
        self.parser = DrugBankXMLParser(xml_path, cv_ids)
        self.parser.parse_simple()  # Use simple parse since we're filtering
    
    def build_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph from parsed data."""
        if not self.parser:
            raise ValueError("Must call parse_drugbank first")
        
        logger.info("Building NetworkX MultiDiGraph...")
        self.graph = nx.MultiDiGraph()
        
        # Add drug nodes
        for drug_id, drug in self.parser.drugs.items():
            self.graph.add_node(
                drug_id,
                type='drug',
                name=drug.name,
                drug_type=drug.type,
                cas_number=drug.cas_number,
                groups=drug.groups,
                indication=drug.indication[:500] if drug.indication else "",
                mechanism=drug.mechanism_of_action[:500] if drug.mechanism_of_action else "",
                metabolism=drug.metabolism[:500] if drug.metabolism else "",
                half_life=drug.half_life,
                atc_codes=drug.atc_codes,
                smiles=drug.smiles,
                inchi_key=drug.inchi_key,
                molecular_weight=drug.molecular_weight,
                logp=drug.logp,
            )
        
        # Add protein nodes
        for protein_id, protein in self.parser.proteins.items():
            self.graph.add_node(
                protein_id,
                type='protein',
                subtype=protein.type,
                name=protein.name,
                organism=protein.organism,
                gene_name=protein.gene_name,
                uniprot_id=protein.uniprot_id,
                general_function=protein.general_function[:300] if protein.general_function else "",
                specific_function=protein.specific_function[:300] if protein.specific_function else "",
                cellular_location=protein.cellular_location,
            )
        
        # Add pathway nodes
        for pathway_id, pathway in self.parser.pathways.items():
            self.graph.add_node(
                pathway_id,
                type='pathway',
                name=pathway.name,
                category=pathway.category,
            )
        
        # Add category nodes
        for cat_name, category in self.parser.categories.items():
            self.graph.add_node(
                f"CAT:{cat_name}",
                type='category',
                name=cat_name,
                mesh_id=category.mesh_id,
            )
        
        # Add DDI edges (from parsed DrugBank interactions)
        ddi_count = 0
        for edge in self.parser.ddi_edges:
            if edge.drug1_id in self.parser.drugs and edge.drug2_id in self.parser.drugs:
                self.graph.add_edge(
                    edge.drug1_id,
                    edge.drug2_id,
                    type='ddi',
                    description=edge.description[:300] if edge.description else "",
                )
                ddi_count += 1
        
        # Add drug-protein edges
        dp_count = 0
        for edge in self.parser.drug_protein_edges:
            if edge.drug_id in self.parser.drugs and edge.protein_id in self.parser.proteins:
                self.graph.add_edge(
                    edge.drug_id,
                    edge.protein_id,
                    type=f'drug_{edge.type}',
                    actions=edge.actions,
                    known_action=edge.known_action,
                    inhibition_strength=edge.inhibition_strength,
                    induction_strength=edge.induction_strength,
                )
                dp_count += 1
        
        # Add drug-pathway edges
        path_count = 0
        for edge in self.parser.drug_pathway_edges:
            if edge.drug_id in self.parser.drugs and edge.pathway_id in self.parser.pathways:
                self.graph.add_edge(
                    edge.drug_id,
                    edge.pathway_id,
                    type='drug_pathway',
                )
                path_count += 1
        
        # Add drug-category edges
        cat_count = 0
        for edge in self.parser.drug_category_edges:
            if edge.drug_id in self.parser.drugs:
                self.graph.add_edge(
                    edge.drug_id,
                    f"CAT:{edge.category_name}",
                    type='drug_category',
                )
                cat_count += 1
        
        logger.info(f"  Drug nodes: {len(self.parser.drugs)}")
        logger.info(f"  Protein nodes: {len(self.parser.proteins)}")
        logger.info(f"  Pathway nodes: {len(self.parser.pathways)}")
        logger.info(f"  Category nodes: {len(self.parser.categories)}")
        logger.info(f"  DDI edges: {ddi_count}")
        logger.info(f"  Drug-protein edges: {dp_count}")
        logger.info(f"  Drug-pathway edges: {path_count}")
        logger.info(f"  Drug-category edges: {cat_count}")
        logger.info(f"  Total nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  Total edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph."""
        if not self.parser:
            return {}
        
        stats = {
            'nodes': {
                'drugs': len(self.parser.drugs),
                'proteins': len(self.parser.proteins),
                'pathways': len(self.parser.pathways),
                'categories': len(self.parser.categories),
                'total': (len(self.parser.drugs) + len(self.parser.proteins) + 
                         len(self.parser.pathways) + len(self.parser.categories)),
            },
            'edges': {
                'ddi': len(self.parser.ddi_edges),
                'drug_protein': len(self.parser.drug_protein_edges),
                'drug_pathway': len(self.parser.drug_pathway_edges),
                'drug_category': len(self.parser.drug_category_edges),
            },
            'protein_types': {
                'targets': sum(1 for p in self.parser.proteins.values() if p.type == 'target'),
                'enzymes': sum(1 for p in self.parser.proteins.values() if p.type == 'enzyme'),
                'carriers': sum(1 for p in self.parser.proteins.values() if p.type == 'carrier'),
                'transporters': sum(1 for p in self.parser.proteins.values() if p.type == 'transporter'),
            },
            'snp_data': {
                'snp_effects': len(self.parser.snp_effects),
                'snp_adverse_reactions': len(self.parser.snp_adverse_reactions),
            },
            'external_ids': {},
            'calculated_properties': {
                'drugs_with_smiles': sum(1 for d in self.parser.drugs.values() if d.smiles),
                'drugs_with_inchi': sum(1 for d in self.parser.drugs.values() if d.inchi),
                'drugs_with_logp': sum(1 for d in self.parser.drugs.values() if d.logp),
            },
        }
        
        # Count external ID types
        for drug in self.parser.drugs.values():
            for resource in drug.external_ids:
                stats['external_ids'][resource] = stats['external_ids'].get(resource, 0) + 1
        
        return stats
    
    def export_neo4j(self, output_dir: str) -> None:
        """Export knowledge graph to Neo4j-compatible CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting to Neo4j format: {output_dir}")
        
        if not self.parser:
            raise ValueError("Must call parse_drugbank first")
        
        # Export drugs
        drugs_data = []
        for drug_id, drug in self.parser.drugs.items():
            drugs_data.append({
                'drugbank_id:ID': drug_id,
                'name': drug.name,
                'type': drug.type,
                'cas_number': drug.cas_number,
                'unii': drug.unii,
                'state': drug.state,
                'groups': '|'.join(drug.groups),
                'indication': drug.indication[:1000] if drug.indication else "",
                'mechanism_of_action': drug.mechanism_of_action[:1000] if drug.mechanism_of_action else "",
                'metabolism': drug.metabolism[:500] if drug.metabolism else "",
                'half_life': drug.half_life,
                'atc_codes': '|'.join(drug.atc_codes),
                'smiles': drug.smiles,
                'inchi_key': drug.inchi_key,
                'molecular_weight': drug.molecular_weight,
                'logp': drug.logp,
                'pubchem_id': drug.external_ids.get('PubChem Compound', ''),
                'kegg_id': drug.external_ids.get('KEGG Drug', drug.external_ids.get('KEGG Compound', '')),
                'chembl_id': drug.external_ids.get('ChEMBL', ''),
                ':LABEL': 'Drug',
            })
        pd.DataFrame(drugs_data).to_csv(output_path / 'drugs.csv', index=False)
        
        # Export proteins
        proteins_data = []
        for protein_id, protein in self.parser.proteins.items():
            proteins_data.append({
                'protein_id:ID': protein_id,
                'name': protein.name,
                'type': protein.type,
                'organism': protein.organism,
                'gene_name': protein.gene_name,
                'uniprot_id': protein.uniprot_id,
                'general_function': protein.general_function[:500] if protein.general_function else "",
                'specific_function': protein.specific_function[:500] if protein.specific_function else "",
                'cellular_location': protein.cellular_location,
                'actions': '|'.join(protein.actions),
                ':LABEL': 'Protein',
            })
        pd.DataFrame(proteins_data).to_csv(output_path / 'proteins.csv', index=False)
        
        # Export pathways
        pathways_data = []
        for pathway_id, pathway in self.parser.pathways.items():
            pathways_data.append({
                'smpdb_id:ID': pathway_id,
                'name': pathway.name,
                'category': pathway.category,
                'enzymes': '|'.join(pathway.enzymes),
                ':LABEL': 'Pathway',
            })
        pd.DataFrame(pathways_data).to_csv(output_path / 'pathways.csv', index=False)
        
        # Export categories
        categories_data = []
        for cat_name, category in self.parser.categories.items():
            categories_data.append({
                'category_id:ID': f"CAT:{cat_name}",
                'name': cat_name,
                'mesh_id': category.mesh_id,
                ':LABEL': 'Category',
            })
        pd.DataFrame(categories_data).to_csv(output_path / 'categories.csv', index=False)
        
        # Export DDI edges
        ddi_data = []
        seen_ddi = set()
        for edge in self.parser.ddi_edges:
            if edge.drug1_id in self.parser.drugs and edge.drug2_id in self.parser.drugs:
                key = (edge.drug1_id, edge.drug2_id)
                if key not in seen_ddi:
                    ddi_data.append({
                        ':START_ID': edge.drug1_id,
                        ':END_ID': edge.drug2_id,
                        'description': edge.description[:500] if edge.description else "",
                        ':TYPE': 'INTERACTS_WITH',
                    })
                    seen_ddi.add(key)
        pd.DataFrame(ddi_data).to_csv(output_path / 'ddi_edges.csv', index=False)
        
        # Export drug-protein edges
        dp_data = []
        for edge in self.parser.drug_protein_edges:
            if edge.drug_id in self.parser.drugs and edge.protein_id in self.parser.proteins:
                dp_data.append({
                    ':START_ID': edge.drug_id,
                    ':END_ID': edge.protein_id,
                    'relationship_type': edge.type,
                    'actions': '|'.join(edge.actions),
                    'known_action': edge.known_action,
                    'inhibition_strength': edge.inhibition_strength,
                    'induction_strength': edge.induction_strength,
                    ':TYPE': edge.type.upper(),
                })
        pd.DataFrame(dp_data).to_csv(output_path / 'drug_protein_edges.csv', index=False)
        
        # Export drug-pathway edges
        pathway_edges = []
        for edge in self.parser.drug_pathway_edges:
            if edge.drug_id in self.parser.drugs and edge.pathway_id in self.parser.pathways:
                pathway_edges.append({
                    ':START_ID': edge.drug_id,
                    ':END_ID': edge.pathway_id,
                    ':TYPE': 'IN_PATHWAY',
                })
        pd.DataFrame(pathway_edges).to_csv(output_path / 'drug_pathway_edges.csv', index=False)
        
        # Export drug-category edges
        category_edges = []
        for edge in self.parser.drug_category_edges:
            if edge.drug_id in self.parser.drugs:
                category_edges.append({
                    ':START_ID': edge.drug_id,
                    ':END_ID': f"CAT:{edge.category_name}",
                    ':TYPE': 'HAS_CATEGORY',
                })
        pd.DataFrame(category_edges).to_csv(output_path / 'drug_category_edges.csv', index=False)
        
        # Export SNP effects
        if self.parser.snp_effects:
            snp_data = []
            for snp in self.parser.snp_effects:
                snp_data.append({
                    'protein_name': snp.protein_name,
                    'gene_symbol': snp.gene_symbol,
                    'uniprot_id': snp.uniprot_id,
                    'rs_id': snp.rs_id,
                    'allele': snp.allele,
                    'defining_change': snp.defining_change,
                    'description': snp.description[:500] if snp.description else "",
                    'pubmed_id': snp.pubmed_id,
                })
            pd.DataFrame(snp_data).to_csv(output_path / 'snp_effects.csv', index=False)
        
        # Export SNP adverse reactions
        if self.parser.snp_adverse_reactions:
            snp_adr_data = []
            for snp in self.parser.snp_adverse_reactions:
                snp_adr_data.append({
                    'protein_name': snp.protein_name,
                    'gene_symbol': snp.gene_symbol,
                    'uniprot_id': snp.uniprot_id,
                    'rs_id': snp.rs_id,
                    'allele': snp.allele,
                    'adverse_reaction': snp.adverse_reaction,
                    'description': snp.description[:500] if snp.description else "",
                    'pubmed_id': snp.pubmed_id,
                })
            pd.DataFrame(snp_adr_data).to_csv(output_path / 'snp_adverse_reactions.csv', index=False)
        
        # Generate Cypher import script
        cypher_script = """
// Neo4j Import Script for DDI Knowledge Graph
// Generated from DrugBank XML

// Create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.protein_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.smpdb_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.category_id IS UNIQUE;

// Load drugs
LOAD CSV WITH HEADERS FROM 'file:///drugs.csv' AS row
CREATE (d:Drug {
    drugbank_id: row.`drugbank_id:ID`,
    name: row.name,
    type: row.type,
    cas_number: row.cas_number,
    groups: row.groups,
    indication: row.indication,
    mechanism_of_action: row.mechanism_of_action,
    metabolism: row.metabolism,
    half_life: row.half_life,
    atc_codes: row.atc_codes,
    smiles: row.smiles,
    inchi_key: row.inchi_key,
    molecular_weight: row.molecular_weight,
    logp: row.logp,
    pubchem_id: row.pubchem_id,
    kegg_id: row.kegg_id,
    chembl_id: row.chembl_id
});

// Load proteins
LOAD CSV WITH HEADERS FROM 'file:///proteins.csv' AS row
CREATE (p:Protein {
    protein_id: row.`protein_id:ID`,
    name: row.name,
    type: row.type,
    organism: row.organism,
    gene_name: row.gene_name,
    uniprot_id: row.uniprot_id,
    general_function: row.general_function,
    specific_function: row.specific_function,
    cellular_location: row.cellular_location,
    actions: row.actions
});

// Load pathways
LOAD CSV WITH HEADERS FROM 'file:///pathways.csv' AS row
CREATE (pw:Pathway {
    smpdb_id: row.`smpdb_id:ID`,
    name: row.name,
    category: row.category,
    enzymes: row.enzymes
});

// Load categories
LOAD CSV WITH HEADERS FROM 'file:///categories.csv' AS row
CREATE (c:Category {
    category_id: row.`category_id:ID`,
    name: row.name,
    mesh_id: row.mesh_id
});

// Load DDI relationships
LOAD CSV WITH HEADERS FROM 'file:///ddi_edges.csv' AS row
MATCH (d1:Drug {drugbank_id: row.`:START_ID`})
MATCH (d2:Drug {drugbank_id: row.`:END_ID`})
CREATE (d1)-[:INTERACTS_WITH {description: row.description}]->(d2);

// Load drug-protein relationships
LOAD CSV WITH HEADERS FROM 'file:///drug_protein_edges.csv' AS row
MATCH (d:Drug {drugbank_id: row.`:START_ID`})
MATCH (p:Protein {protein_id: row.`:END_ID`})
CREATE (d)-[:ACTS_ON {
    type: row.relationship_type,
    actions: row.actions,
    known_action: row.known_action,
    inhibition_strength: row.inhibition_strength,
    induction_strength: row.induction_strength
}]->(p);

// Load drug-pathway relationships
LOAD CSV WITH HEADERS FROM 'file:///drug_pathway_edges.csv' AS row
MATCH (d:Drug {drugbank_id: row.`:START_ID`})
MATCH (pw:Pathway {smpdb_id: row.`:END_ID`})
CREATE (d)-[:IN_PATHWAY]->(pw);

// Load drug-category relationships
LOAD CSV WITH HEADERS FROM 'file:///drug_category_edges.csv' AS row
MATCH (d:Drug {drugbank_id: row.`:START_ID`})
MATCH (c:Category {category_id: row.`:END_ID`})
CREATE (d)-[:HAS_CATEGORY]->(c);
"""
        
        with open(output_path / 'neo4j_import.cypher', 'w') as f:
            f.write(cypher_script)
        
        logger.info("  Neo4j export complete!")
    
    def save(self, path: str) -> None:
        """Save the knowledge graph to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({
                'drugs': self.parser.drugs if self.parser else {},
                'proteins': self.parser.proteins if self.parser else {},
                'pathways': self.parser.pathways if self.parser else {},
                'categories': self.parser.categories if self.parser else {},
                'ddi_edges': self.parser.ddi_edges if self.parser else [],
                'drug_protein_edges': self.parser.drug_protein_edges if self.parser else [],
                'drug_pathway_edges': self.parser.drug_pathway_edges if self.parser else [],
                'drug_category_edges': self.parser.drug_category_edges if self.parser else [],
                'snp_effects': self.parser.snp_effects if self.parser else [],
                'snp_adverse_reactions': self.parser.snp_adverse_reactions if self.parser else [],
                'graph': self.graph,
            }, f)
        logger.info(f"Knowledge graph saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Build comprehensive DDI Knowledge Graph from DrugBank XML."""
    
    # Paths
    xml_path = "data/full database.xml"
    csv_path = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"
    output_dir = "knowledge_graph_real"
    
    # Initialize
    kg = RealDDIKnowledgeGraph()
    
    # Load CV drug IDs for filtering
    kg.load_cv_drug_ids(csv_path)
    
    # Parse DrugBank XML (filtered to CV drugs)
    kg.parse_drugbank(xml_path, filter_cv=True)
    
    # Build NetworkX graph
    kg.build_graph()
    
    # Get statistics
    stats = kg.get_statistics()
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH STATISTICS (FROM DRUGBANK XML)")
    print("=" * 60)
    
    print("\nNode Counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    
    print("\nEdge Counts:")
    for edge_type, count in stats['edges'].items():
        print(f"  {edge_type}: {count:,}")
    
    print("\nProtein Types:")
    for ptype, count in stats['protein_types'].items():
        print(f"  {ptype}: {count:,}")
    
    print("\nSNP Data:")
    for snp_type, count in stats['snp_data'].items():
        print(f"  {snp_type}: {count:,}")
    
    print("\nExternal Identifiers:")
    for resource, count in sorted(stats['external_ids'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {resource}: {count:,}")
    
    print("\nCalculated Properties:")
    for prop, count in stats['calculated_properties'].items():
        print(f"  {prop}: {count:,}")
    
    # Export
    os.makedirs(output_dir, exist_ok=True)
    kg.export_neo4j(f"{output_dir}/neo4j_export")
    kg.save(f"{output_dir}/knowledge_graph.pkl")
    
    # Save statistics
    with open(f"{output_dir}/statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == '__main__':
    main()
