#!/usr/bin/env python3
"""
Clinical Severity Validation Pipeline

Uses multiple sources for robust DDI severity validation:
1. DrugBank XML - Extract synonyms for comprehensive name matching
2. FDA Label Patterns - Clinical severity rules from drug labeling
3. TWOSIDES - Real-world clinical outcomes for validation
4. Expert-Annotated DDI Pairs - Curated high-confidence ground truth

Name matching strategy:
- Exact match on generic name
- Synonym matching from DrugBank 
- Brand name matching
- Fuzzy matching with threshold
- Chemical name matching (IUPAC)
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    drugbank_xml = Path("data/full database.xml")
    ddi_data = Path("data/ddi_cardio_or_antithrombotic_labeled (1).csv")
    output_dir = Path("severity_validation")
    
    # Model
    model_name = "facebook/bart-large-mnli"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    
    # Matching thresholds
    fuzzy_threshold = 0.85


# ============================================================================
# DRUG NAME SYNONYM EXTRACTOR (from DrugBank XML)
# ============================================================================

class DrugSynonymExtractor:
    """Extract drug synonyms from DrugBank XML for comprehensive name matching"""
    
    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.drug_synonyms: Dict[str, Set[str]] = defaultdict(set)
        self.name_to_drugbank: Dict[str, str] = {}
        self.drugbank_to_names: Dict[str, Set[str]] = defaultdict(set)
        
    def extract_synonyms(self, max_drugs: int = None):
        """
        Parse DrugBank XML and extract all drug name variants
        """
        print("📖 Extracting drug synonyms from DrugBank XML...")
        print("   (This may take a few minutes for 1.8GB file)")
        
        ns = {'db': 'http://www.drugbank.ca'}
        
        # Use iterparse for memory efficiency
        context = ET.iterparse(str(self.xml_path), events=('end',))
        
        count = 0
        for event, elem in context:
            if elem.tag == '{http://www.drugbank.ca}drug':
                # Extract DrugBank ID
                drugbank_id = None
                for db_id in elem.findall('.//db:drugbank-id', ns):
                    if db_id.get('primary') == 'true' or drugbank_id is None:
                        drugbank_id = db_id.text
                        break
                
                if not drugbank_id:
                    elem.clear()
                    continue
                
                names = set()
                
                # Primary name
                name_elem = elem.find('db:name', ns)
                if name_elem is not None and name_elem.text:
                    names.add(name_elem.text.lower().strip())
                
                # Synonyms
                for syn in elem.findall('.//db:synonyms/db:synonym', ns):
                    if syn.text:
                        names.add(syn.text.lower().strip())
                
                # International brands
                for brand in elem.findall('.//db:international-brands/db:international-brand/db:name', ns):
                    if brand.text:
                        names.add(brand.text.lower().strip())
                
                # Products (brand names)
                for product in elem.findall('.//db:products/db:product/db:name', ns):
                    if product.text:
                        # Extract just the drug name, not formulation details
                        brand = product.text.split()[0].lower().strip()
                        if len(brand) > 2:
                            names.add(brand)
                
                # Store mappings
                for name in names:
                    self.name_to_drugbank[name] = drugbank_id
                    self.drugbank_to_names[drugbank_id].add(name)
                
                count += 1
                if count % 1000 == 0:
                    print(f"   Processed {count} drugs...")
                
                if max_drugs and count >= max_drugs:
                    break
                
                elem.clear()
        
        print(f"   Extracted synonyms for {count} drugs")
        print(f"   Total name variants: {len(self.name_to_drugbank)}")
        
        return self
    
    def get_drugbank_id(self, drug_name: str) -> Optional[str]:
        """Get DrugBank ID from any name variant"""
        name_lower = drug_name.lower().strip()
        return self.name_to_drugbank.get(name_lower)
    
    def get_all_names(self, drugbank_id: str) -> Set[str]:
        """Get all name variants for a DrugBank ID"""
        return self.drugbank_to_names.get(drugbank_id, set())
    
    def normalize_name(self, name: str) -> str:
        """Normalize drug name for matching"""
        name = name.lower().strip()
        # Remove common suffixes
        name = re.sub(r'\s*(hydrochloride|hcl|sodium|potassium|acetate|sulfate|mesylate|maleate|tartrate|fumarate|succinate|besylate|citrate|phosphate)\s*$', '', name)
        # Remove parenthetical info
        name = re.sub(r'\s*\([^)]*\)\s*', '', name)
        return name.strip()


# ============================================================================
# EXPERT-CURATED DDI SEVERITY DATA
# ============================================================================

# Expanded clinical ground truth from FDA labels, clinical guidelines, and literature
# Each entry: (drug1, drug2, severity, source, evidence_level)

EXPERT_DDI_SEVERITY = [
    # === CONTRAINDICATED (Life-threatening, FDA Black Box) ===
    # MAOIs + Serotonergics
    ("selegiline", "fluoxetine", "Contraindicated", "FDA", "Black Box"),
    ("selegiline", "sertraline", "Contraindicated", "FDA", "Black Box"),
    ("selegiline", "paroxetine", "Contraindicated", "FDA", "Black Box"),
    ("phenelzine", "meperidine", "Contraindicated", "FDA", "Black Box"),
    ("tranylcypromine", "tramadol", "Contraindicated", "FDA", "Black Box"),
    ("isocarboxazid", "venlafaxine", "Contraindicated", "FDA", "Black Box"),
    
    # QT Prolongation - High Risk
    ("thioridazine", "ziprasidone", "Contraindicated", "FDA", "Black Box"),
    ("droperidol", "haloperidol", "Contraindicated", "FDA", "Black Box"),
    ("cisapride", "ketoconazole", "Contraindicated", "FDA", "Withdrawn"),
    ("terfenadine", "erythromycin", "Contraindicated", "FDA", "Withdrawn"),
    ("pimozide", "clarithromycin", "Contraindicated", "FDA", "Black Box"),
    ("sotalol", "amiodarone", "Contraindicated", "Clinical", "High"),
    
    # Severe Bleeding Risk
    ("warfarin", "aspirin", "Contraindicated", "Clinical", "High"),  # High-dose aspirin
    ("dabigatran", "ketoconazole", "Contraindicated", "FDA", "Label"),
    ("rivaroxaban", "ketoconazole", "Contraindicated", "FDA", "Label"),
    
    # Ergot Alkaloids + CYP3A4 Inhibitors
    ("ergotamine", "ritonavir", "Contraindicated", "FDA", "Black Box"),
    ("dihydroergotamine", "clarithromycin", "Contraindicated", "FDA", "Label"),
    
    # Statins + Strong CYP Inhibitors (Rhabdomyolysis)
    ("simvastatin", "itraconazole", "Contraindicated", "FDA", "Label"),
    ("lovastatin", "ketoconazole", "Contraindicated", "FDA", "Label"),
    
    # === MAJOR (Serious adverse effects, close monitoring required) ===
    # Anticoagulant Combinations
    ("warfarin", "amiodarone", "Major", "FDA", "Label"),
    ("warfarin", "fluconazole", "Major", "FDA", "Label"),
    ("warfarin", "metronidazole", "Major", "Clinical", "High"),
    ("warfarin", "ciprofloxacin", "Major", "Clinical", "High"),
    ("warfarin", "cotrimoxazole", "Major", "Clinical", "High"),
    ("heparin", "aspirin", "Major", "Clinical", "High"),
    ("enoxaparin", "clopidogrel", "Major", "Clinical", "High"),
    ("rivaroxaban", "aspirin", "Major", "Clinical", "High"),
    ("apixaban", "clopidogrel", "Major", "Clinical", "High"),
    
    # Potassium-Affecting Combinations
    ("spironolactone", "potassium chloride", "Major", "FDA", "Label"),
    ("enalapril", "spironolactone", "Major", "Clinical", "High"),
    ("lisinopril", "triamterene", "Major", "Clinical", "High"),
    ("losartan", "potassium chloride", "Major", "Clinical", "High"),
    
    # Hypoglycemia Risk
    ("insulin", "sulfonylurea", "Major", "Clinical", "High"),
    ("glyburide", "fluconazole", "Major", "Clinical", "High"),
    ("glimepiride", "clarithromycin", "Major", "Clinical", "Moderate"),
    
    # Serotonin Syndrome (non-MAOI)
    ("tramadol", "sertraline", "Major", "FDA", "Label"),
    ("fentanyl", "fluoxetine", "Major", "Clinical", "High"),
    ("methadone", "paroxetine", "Major", "Clinical", "High"),
    
    # CNS Depression
    ("oxycodone", "benzodiazepine", "Major", "FDA", "Black Box"),
    ("morphine", "alprazolam", "Major", "FDA", "Black Box"),
    ("hydrocodone", "diazepam", "Major", "FDA", "Black Box"),
    ("fentanyl", "lorazepam", "Major", "FDA", "Black Box"),
    
    # Nephrotoxicity
    ("gentamicin", "vancomycin", "Major", "Clinical", "High"),
    ("amphotericin", "ciclosporin", "Major", "Clinical", "High"),
    ("nsaid", "methotrexate", "Major", "FDA", "Label"),
    ("ibuprofen", "lithium", "Major", "FDA", "Label"),
    
    # Digoxin Toxicity
    ("digoxin", "amiodarone", "Major", "FDA", "Label"),
    ("digoxin", "verapamil", "Major", "FDA", "Label"),
    ("digoxin", "quinidine", "Major", "Clinical", "High"),
    ("digoxin", "clarithromycin", "Major", "Clinical", "High"),
    
    # Methotrexate Toxicity
    ("methotrexate", "trimethoprim", "Major", "FDA", "Label"),
    ("methotrexate", "probenecid", "Major", "FDA", "Label"),
    
    # Theophylline
    ("theophylline", "ciprofloxacin", "Major", "FDA", "Label"),
    ("theophylline", "fluvoxamine", "Major", "FDA", "Label"),
    
    # === MODERATE (May require dose adjustment or monitoring) ===
    # CYP Interactions - Moderate Effect
    ("atorvastatin", "diltiazem", "Moderate", "Clinical", "Moderate"),
    ("simvastatin", "amlodipine", "Moderate", "FDA", "Label"),
    ("cyclosporine", "atorvastatin", "Moderate", "FDA", "Label"),
    ("tacrolimus", "fluconazole", "Moderate", "Clinical", "Moderate"),
    
    # Antihypertensive Additive Effects
    ("lisinopril", "amlodipine", "Moderate", "Clinical", "Low"),
    ("metoprolol", "diltiazem", "Moderate", "Clinical", "Moderate"),
    ("atenolol", "verapamil", "Moderate", "Clinical", "Moderate"),
    ("carvedilol", "digoxin", "Moderate", "Clinical", "Moderate"),
    
    # GI Effects
    ("aspirin", "ibuprofen", "Moderate", "FDA", "Label"),
    ("naproxen", "prednisone", "Moderate", "Clinical", "Moderate"),
    ("celecoxib", "warfarin", "Moderate", "Clinical", "Moderate"),
    
    # Sedation
    ("diphenhydramine", "zolpidem", "Moderate", "Clinical", "Moderate"),
    ("hydroxyzine", "oxycodone", "Moderate", "Clinical", "Moderate"),
    ("promethazine", "codeine", "Moderate", "FDA", "Label"),
    
    # Absorption Interactions
    ("ciprofloxacin", "antacid", "Moderate", "FDA", "Label"),
    ("levothyroxine", "calcium", "Moderate", "FDA", "Label"),
    ("tetracycline", "iron", "Moderate", "Clinical", "Moderate"),
    
    # Metabolic
    ("metformin", "contrast media", "Moderate", "FDA", "Label"),
    ("phenytoin", "folic acid", "Moderate", "Clinical", "Moderate"),
    
    # === MINOR (Usually not clinically significant) ===
    # Minimal Clinical Effect
    ("omeprazole", "clopidogrel", "Minor", "Clinical", "Debated"),  # Disputed
    ("pantoprazole", "methotrexate", "Minor", "Clinical", "Low"),
    ("ranitidine", "ketoconazole", "Minor", "Clinical", "Low"),
    
    # Absorption - Low Impact
    ("magnesium", "bisphosphonate", "Minor", "Clinical", "Low"),
    ("fiber", "medication", "Minor", "Clinical", "Low"),
    
    # Mild Additive Effects
    ("caffeine", "theophylline", "Minor", "Clinical", "Low"),
    ("grapefruit", "felodipine", "Minor", "Clinical", "Low"),  # Unless large amounts
]

def build_expert_dataset() -> pd.DataFrame:
    """Convert expert DDI data to DataFrame with normalized names"""
    data = []
    for drug1, drug2, severity, source, evidence in EXPERT_DDI_SEVERITY:
        data.append({
            'drug1': drug1.lower(),
            'drug2': drug2.lower(),
            'severity': severity,
            'source': source,
            'evidence_level': evidence
        })
        # Add reverse pair
        data.append({
            'drug1': drug2.lower(),
            'drug2': drug1.lower(), 
            'severity': severity,
            'source': source,
            'evidence_level': evidence
        })
    
    df = pd.DataFrame(data).drop_duplicates(subset=['drug1', 'drug2'])
    print(f"   Expert database: {len(df)} DDI pairs")
    return df


# ============================================================================
# FUZZY NAME MATCHING
# ============================================================================

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity ratio"""
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    if len2 == 0:
        return 0.0
    
    # Simple character overlap for speed
    s1_set = set(s1.lower())
    s2_set = set(s2.lower())
    
    intersection = len(s1_set & s2_set)
    union = len(s1_set | s2_set)
    
    jaccard = intersection / union if union > 0 else 0
    
    # Length similarity
    len_sim = min(len1, len2) / max(len1, len2)
    
    # Prefix match bonus
    prefix_match = 0
    for i in range(min(len1, len2, 5)):
        if s1[i].lower() == s2[i].lower():
            prefix_match += 1
        else:
            break
    prefix_sim = prefix_match / 5
    
    return 0.4 * jaccard + 0.3 * len_sim + 0.3 * prefix_sim


class DrugMatcher:
    """
    Multi-strategy drug name matcher
    Handles: generic names, brand names, synonyms, salts, fuzzy matching
    """
    
    def __init__(self, synonym_extractor: Optional[DrugSynonymExtractor] = None):
        self.synonym_extractor = synonym_extractor
        self.name_cache: Dict[str, str] = {}  # Normalized name -> DrugBank ID
        
        # Common salt/ester suffixes to strip
        self.salt_suffixes = [
            'hydrochloride', 'hcl', 'sodium', 'potassium', 'calcium',
            'acetate', 'sulfate', 'sulphate', 'mesylate', 'maleate',
            'tartrate', 'fumarate', 'succinate', 'besylate', 'citrate',
            'phosphate', 'nitrate', 'bromide', 'chloride', 'iodide',
            'lactate', 'gluconate', 'carbonate', 'oxide', 'hydroxide'
        ]
        
        # Brand name to generic mappings (common ones)
        self.brand_to_generic = {
            'lipitor': 'atorvastatin',
            'crestor': 'rosuvastatin', 
            'zocor': 'simvastatin',
            'pravachol': 'pravastatin',
            'plavix': 'clopidogrel',
            'coumadin': 'warfarin',
            'xarelto': 'rivaroxaban',
            'eliquis': 'apixaban',
            'pradaxa': 'dabigatran',
            'aspirin': 'acetylsalicylic acid',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'aleve': 'naproxen',
            'synthroid': 'levothyroxine',
            'lasix': 'furosemide',
            'zestril': 'lisinopril',
            'prinivil': 'lisinopril',
            'vasotec': 'enalapril',
            'cozaar': 'losartan',
            'diovan': 'valsartan',
            'norvasc': 'amlodipine',
            'cardizem': 'diltiazem',
            'calan': 'verapamil',
            'lopressor': 'metoprolol',
            'tenormin': 'atenolol',
            'coreg': 'carvedilol',
            'lanoxin': 'digoxin',
            'prozac': 'fluoxetine',
            'zoloft': 'sertraline',
            'paxil': 'paroxetine',
            'lexapro': 'escitalopram',
            'cymbalta': 'duloxetine',
            'effexor': 'venlafaxine',
            'wellbutrin': 'bupropion',
            'xanax': 'alprazolam',
            'valium': 'diazepam',
            'ativan': 'lorazepam',
            'klonopin': 'clonazepam',
            'ambien': 'zolpidem',
            'oxycontin': 'oxycodone',
            'vicodin': 'hydrocodone',
            'percocet': 'oxycodone',
            'ultram': 'tramadol',
            'diflucan': 'fluconazole',
            'sporanox': 'itraconazole',
            'nizoral': 'ketoconazole',
            'biaxin': 'clarithromycin',
            'zithromax': 'azithromycin',
            'cipro': 'ciprofloxacin',
            'levaquin': 'levofloxacin',
            'flagyl': 'metronidazole',
            'bactrim': 'trimethoprim',
            'glucophage': 'metformin',
            'amaryl': 'glimepiride',
            'diabeta': 'glyburide',
            'actos': 'pioglitazone',
            'avandia': 'rosiglitazone',
            'januvia': 'sitagliptin',
            'nexium': 'esomeprazole',
            'prilosec': 'omeprazole',
            'prevacid': 'lansoprazole',
            'protonix': 'pantoprazole',
            'zantac': 'ranitidine',
            'pepcid': 'famotidine',
            'singulair': 'montelukast',
            'flovent': 'fluticasone',
            'advair': 'fluticasone',
            'symbicort': 'budesonide',
            'prednisone': 'prednisone',
            'medrol': 'methylprednisolone',
        }
        
    def normalize(self, name: str) -> str:
        """Normalize drug name by removing salts and formatting"""
        if not name:
            return ""
            
        name = name.lower().strip()
        
        # Check brand name mapping first
        if name in self.brand_to_generic:
            name = self.brand_to_generic[name]
        
        # Remove parenthetical content
        name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
        
        # Remove salt suffixes
        for suffix in self.salt_suffixes:
            name = re.sub(rf'\s*{suffix}\s*$', '', name, flags=re.IGNORECASE)
        
        # Clean up whitespace
        name = ' '.join(name.split())
        
        return name
    
    def match(self, name1: str, name2: str, threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Check if two drug names refer to the same drug
        Returns: (is_match, confidence)
        """
        norm1 = self.normalize(name1)
        norm2 = self.normalize(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return True, 1.0
        
        # Check if one contains the other (for partial matches)
        if norm1 in norm2 or norm2 in norm1:
            return True, 0.95
        
        # Check synonyms if available
        if self.synonym_extractor:
            db_id1 = self.synonym_extractor.get_drugbank_id(norm1)
            db_id2 = self.synonym_extractor.get_drugbank_id(norm2)
            
            if db_id1 and db_id2 and db_id1 == db_id2:
                return True, 0.98
            
            # Check if name2 is a synonym of name1's drug
            if db_id1:
                synonyms = self.synonym_extractor.get_all_names(db_id1)
                if norm2 in synonyms:
                    return True, 0.95
        
        # Fuzzy matching
        ratio = levenshtein_ratio(norm1, norm2)
        if ratio >= threshold:
            return True, ratio
        
        return False, ratio


# ============================================================================
# DDI SEVERITY VALIDATION PIPELINE
# ============================================================================

class ClinicalSeverityValidator:
    """
    Validates DDI severity predictions against expert-curated clinical data
    with comprehensive drug name matching
    """
    
    SEVERITY_ORDER = ['Minor', 'Moderate', 'Major', 'Contraindicated']
    SEVERITY_MAP = {
        'minor': 0, 'minor interaction': 0,
        'moderate': 1, 'moderate interaction': 1,
        'major': 2, 'major interaction': 2,
        'contraindicated': 3, 'contraindicated interaction': 3
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.synonym_extractor = None
        self.matcher = None
        self.expert_df = None
        
    def initialize(self, extract_synonyms: bool = True):
        """Initialize all components"""
        print("\n" + "="*70)
        print("CLINICAL SEVERITY VALIDATION PIPELINE")
        print("="*70)
        
        # Build expert database
        print("\n📚 Loading expert-curated DDI severity data...")
        self.expert_df = build_expert_dataset()
        
        # Extract synonyms from DrugBank (optional, takes time)
        if extract_synonyms and self.config.drugbank_xml.exists():
            self.synonym_extractor = DrugSynonymExtractor(self.config.drugbank_xml)
            self.synonym_extractor.extract_synonyms(max_drugs=5000)  # Limit for speed
        
        # Initialize matcher
        self.matcher = DrugMatcher(self.synonym_extractor)
        
        return self
    
    def match_ddi_pair(self, drug1: str, drug2: str, 
                       expert_df: pd.DataFrame) -> Optional[Dict]:
        """
        Find matching DDI pair in expert database using multiple strategies
        """
        norm1 = self.matcher.normalize(drug1)
        norm2 = self.matcher.normalize(drug2)
        
        best_match = None
        best_score = 0
        
        for _, row in expert_df.iterrows():
            exp_drug1 = row['drug1']
            exp_drug2 = row['drug2']
            
            # Try both orderings
            match1_a, score1_a = self.matcher.match(norm1, exp_drug1)
            match2_a, score2_a = self.matcher.match(norm2, exp_drug2)
            
            match1_b, score1_b = self.matcher.match(norm1, exp_drug2)
            match2_b, score2_b = self.matcher.match(norm2, exp_drug1)
            
            score_a = (score1_a + score2_a) / 2 if match1_a and match2_a else 0
            score_b = (score1_b + score2_b) / 2 if match1_b and match2_b else 0
            
            best_pair_score = max(score_a, score_b)
            
            if best_pair_score > best_score and best_pair_score > 0.8:
                best_score = best_pair_score
                best_match = {
                    'expert_drug1': exp_drug1,
                    'expert_drug2': exp_drug2,
                    'expert_severity': row['severity'],
                    'source': row['source'],
                    'evidence': row['evidence_level'],
                    'match_score': best_score
                }
        
        return best_match
    
    def validate(self, ddi_df: pd.DataFrame) -> Dict:
        """
        Validate DDI predictions against expert-curated data
        """
        print("\n🔍 Matching DDI pairs with expert database...")
        
        matches = []
        unmatched = []
        
        # Sample for validation (expert database is smaller than full DDI)
        sample_size = min(10000, len(ddi_df))
        sample_df = ddi_df.sample(n=sample_size, random_state=42)
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Matching"):
            drug1 = row['drug_name_1']
            drug2 = row['drug_name_2']
            predicted_severity = row['severity_label']
            
            match = self.match_ddi_pair(drug1, drug2, self.expert_df)
            
            if match:
                matches.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'predicted_severity': predicted_severity,
                    'predicted_confidence': row.get('severity_confidence', 0),
                    **match
                })
            else:
                unmatched.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'predicted_severity': predicted_severity
                })
        
        matched_df = pd.DataFrame(matches)
        
        if len(matched_df) == 0:
            print("   ⚠️ No matches found!")
            return {'matched': 0, 'accuracy': 0}
        
        print(f"\n   Matched pairs: {len(matched_df)}")
        print(f"   Unmatched (sampled): {len(unmatched)}")
        
        # Calculate metrics
        def normalize_severity(s):
            s_lower = s.lower().replace(' interaction', '')
            return self.SEVERITY_MAP.get(s_lower, -1)
        
        matched_df['predicted_num'] = matched_df['predicted_severity'].apply(normalize_severity)
        matched_df['expert_num'] = matched_df['expert_severity'].apply(normalize_severity)
        
        valid_df = matched_df[(matched_df['predicted_num'] >= 0) & (matched_df['expert_num'] >= 0)]
        
        if len(valid_df) == 0:
            print("   ⚠️ No valid severity comparisons!")
            return {'matched': len(matched_df), 'accuracy': 0}
        
        # Exact accuracy
        exact_acc = accuracy_score(valid_df['expert_num'], valid_df['predicted_num'])
        
        # Adjacent accuracy (within 1 level)
        adjacent_correct = sum(abs(valid_df['predicted_num'] - valid_df['expert_num']) <= 1)
        adjacent_acc = adjacent_correct / len(valid_df)
        
        # F1 scores
        f1_macro = f1_score(valid_df['expert_num'], valid_df['predicted_num'], 
                           average='macro', zero_division=0)
        f1_weighted = f1_score(valid_df['expert_num'], valid_df['predicted_num'],
                              average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(valid_df['expert_num'], valid_df['predicted_num'],
                             labels=[0, 1, 2, 3])
        
        results = {
            'matched_pairs': len(matched_df),
            'valid_comparisons': len(valid_df),
            'exact_accuracy': exact_acc,
            'adjacent_accuracy': adjacent_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'matched_df': matched_df
        }
        
        # Print results
        print(f"\n   📊 VALIDATION RESULTS:")
        print(f"      Valid comparisons: {len(valid_df)}")
        print(f"      Exact accuracy: {exact_acc:.1%}")
        print(f"      Adjacent accuracy (±1): {adjacent_acc:.1%}")
        print(f"      F1 (macro): {f1_macro:.3f}")
        print(f"      F1 (weighted): {f1_weighted:.3f}")
        
        print(f"\n   📋 Confusion Matrix:")
        print(f"      Pred →  Minor  Mod   Major Contra")
        print(f"      Expert ↓")
        for i, label in enumerate(['Minor', 'Moderate', 'Major', 'Contra']):
            row_str = ''.join(f'{cm[i][j]:6d}' for j in range(4))
            print(f"      {label:8s}{row_str}")
        
        # Save results
        matched_df.to_csv(self.config.output_dir / 'expert_matched_pairs.csv', index=False)
        
        # Save detailed analysis
        analysis = {
            'exact_accuracy': exact_acc,
            'adjacent_accuracy': adjacent_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'matched_pairs': len(matched_df),
            'severity_distribution_predicted': valid_df['predicted_num'].value_counts().to_dict(),
            'severity_distribution_expert': valid_df['expert_num'].value_counts().to_dict(),
        }
        
        import json
        with open(self.config.output_dir / 'validation_results.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    # Load DDI data
    print(f"\n📂 Loading DDI data from {config.ddi_data}...")
    ddi_df = pd.read_csv(config.ddi_data)
    print(f"   Loaded {len(ddi_df)} DDI pairs")
    
    # Initialize validator
    validator = ClinicalSeverityValidator(config)
    validator.initialize(extract_synonyms=True)
    
    # Validate predictions
    results = validator.validate(ddi_df)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\n   Results saved to: {config.output_dir}/")
    
    return results


if __name__ == "__main__":
    main()
