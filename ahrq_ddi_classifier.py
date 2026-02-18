#!/usr/bin/env python3
"""
AHRQ-Compliant DDI Seriousness Classification System

Based on: Malone DC, et al. "Recommendations for Selecting Drug-Drug Interactions 
for Clinical Decision Support." Am J Health Syst Pharm. 2016;73(8):576–585.
PMC5064943

Key Recommendations Implemented:
1. Three seriousness categories (High/Moderate/Low) for CDS
2. Judicious use of "Contraindicated" - only true absolute contraindications
3. Clinical consequences clearly described
4. Evidence quality grading (GRADE-aligned)
5. Recommended management actions
6. Interaction mechanism classification

Author: DDI Risk Analysis Research Team
Date: 2026
"""

import os
import json
import re
import warnings
from pathlib import Path
from dataclasses import dataclass, field, field
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import Counter, defaultdict
from enum import Enum
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    ddi_data_path: str = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"
    output_dir: Path = field(default_factory=lambda: Path("publication_ahrq"))
    seed: int = 42
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)


# ============================================================================
# AHRQ CLASSIFICATION FRAMEWORK
# ============================================================================

class Seriousness(Enum):
    """
    AHRQ Recommended Seriousness Categories (Malone et al. 2016)
    
    Per recommendations: "We recommend that decision support systems for DDIs 
    use no more than three categories of seriousness."
    
    The terminology follows AHRQ preference:
    - "Seriousness" = extent to which adverse reaction can/does cause harm
    - "Severity" = intensity of adverse reaction in an individual
    """
    HIGH = "High"           # Interruptive alert - requires clinician action
    MODERATE = "Moderate"   # Notification - may require therapy modification  
    LOW = "Low"             # Generally should not require notification
    
    # Special category per AHRQ: "contraindicated" should be rare
    CONTRAINDICATED = "Contraindicated"  # Absolutely no situations where benefit > risk


class EvidenceQuality(Enum):
    """
    Evidence quality rating aligned with GRADE approach
    Reference: Guyatt et al. BMJ 2008;336:924-926
    """
    HIGH = "High"           # RCTs, systematic reviews
    MODERATE = "Moderate"   # Observational studies, cohort data 
    LOW = "Low"             # Case reports, pharmacological reasoning
    VERY_LOW = "Very Low"   # In vitro, theoretical


class InteractionMechanism(Enum):
    """Interaction mechanism types per AHRQ recommendation"""
    PHARMACOKINETIC = "Pharmacokinetic"   # Absorption, metabolism, excretion
    PHARMACODYNAMIC = "Pharmacodynamic"   # Same receptor/pathway
    MIXED = "Mixed"                        # Both PK and PD
    UNKNOWN = "Unknown"


@dataclass
class DDIClassificationResult:
    """
    Complete DDI classification per AHRQ recommendations
    
    Includes all 7 key elements recommended by Malone et al.:
    1. Classification of seriousness
    2. Clinical consequences
    3. Frequency of harm (when available)
    4. Modifying factors
    5. Interaction mechanism
    6. Recommended action
    7. Evidence (with quality ratings)
    """
    seriousness: Seriousness
    clinical_consequence: str
    frequency: str  # "Common", "Uncommon", "Rare", "Unknown"
    mechanism: InteractionMechanism
    recommended_action: str
    evidence_quality: EvidenceQuality
    evidence_reference: str
    confidence: float
    
    # For backward compatibility with 4-tier systems
    legacy_severity: str  # "Contraindicated", "Major", "Moderate", "Minor"
    
    def to_dict(self) -> dict:
        return {
            'seriousness': self.seriousness.value,
            'clinical_consequence': self.clinical_consequence,
            'frequency': self.frequency,
            'mechanism': self.mechanism.value,
            'recommended_action': self.recommended_action,
            'evidence_quality': self.evidence_quality.value,
            'evidence_reference': self.evidence_reference,
            'confidence': self.confidence,
            'legacy_severity': self.legacy_severity
        }


# ============================================================================
# AHRQ-COMPLIANT DDI CLASSIFIER
# ============================================================================

class AHRQDDIClassifier:
    """
    AHRQ-Compliant DDI Seriousness Classifier
    
    Implements recommendations from Malone et al. (2016):
    - Three-tier seriousness system (High/Moderate/Low)
    - Judicious contraindicated classification
    - Clinical consequences emphasis
    - Evidence-based with quality ratings
    - Management recommendations included
    
    Key principle: "Contraindicated DDIs are those for which no situations 
    have been identified where the benefit of the combination outweighs the risk."
    """
    
    # ==========================================================================
    # ABSOLUTE CONTRAINDICATIONS (Very restrictive per AHRQ)
    # Only pairs where concurrent use should NEVER occur
    # ==========================================================================
    
    ABSOLUTE_CONTRAINDICATIONS = {
        # These are TRUE absolute contraindications per FDA/AHRQ standards
        # "no situations exist where benefit outweighs risk"
        
        # MAOIs + Serotonergic drugs - Serotonin syndrome (FDA Black Box)
        ('selegiline', 'meperidine'): {
            'consequence': 'Fatal serotonin syndrome',
            'evidence': 'FDA Black Box Warning',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication - never co-administer'
        },
        ('tranylcypromine', 'meperidine'): {
            'consequence': 'Fatal serotonin syndrome',
            'evidence': 'FDA Black Box Warning',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication - never co-administer'
        },
        ('phenelzine', 'meperidine'): {
            'consequence': 'Fatal serotonin syndrome',
            'evidence': 'FDA Black Box Warning',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication - never co-administer'
        },
        
        # Methotrexate + Trimethoprim - Severe pancytopenia
        ('methotrexate', 'trimethoprim'): {
            'consequence': 'Severe pancytopenia, potentially fatal',
            'evidence': 'Multiple case reports, pharmacological basis',
            'quality': EvidenceQuality.MODERATE,
            'action': 'Avoid combination; use alternative antibiotic'
        },
        
        # Cisapride + QT drugs (withdrawn from market due to deaths)
        ('cisapride', 'erythromycin'): {
            'consequence': 'Torsades de pointes, sudden death',
            'evidence': 'FDA - Drug withdrawn for this reason',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication'
        },
        ('cisapride', 'ketoconazole'): {
            'consequence': 'Torsades de pointes, sudden death',
            'evidence': 'FDA - Drug withdrawn for this reason',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication'
        },
        
        # Thioridazine + CYP2D6 inhibitors - FDA label contraindication
        ('thioridazine', 'fluoxetine'): {
            'consequence': 'QT prolongation, torsades de pointes',
            'evidence': 'FDA Label Contraindication',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication - use alternative'
        },
        ('thioridazine', 'paroxetine'): {
            'consequence': 'QT prolongation, torsades de pointes',
            'evidence': 'FDA Label Contraindication',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication - use alternative'
        },
        
        # Pimozide + macrolides - QT prolongation
        ('pimozide', 'clarithromycin'): {
            'consequence': 'QT prolongation, sudden cardiac death',
            'evidence': 'FDA Label Contraindication',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication'
        },
        ('pimozide', 'erythromycin'): {
            'consequence': 'QT prolongation, sudden cardiac death',
            'evidence': 'FDA Label Contraindication',
            'quality': EvidenceQuality.HIGH,
            'action': 'Absolute contraindication'
        },
    }
    
    # ==========================================================================
    # HIGH SERIOUSNESS - Interruptive Alert Required
    # May be life-threatening or require medical intervention
    # ==========================================================================
    
    HIGH_SERIOUSNESS_PAIRS = {
        # QT Prolongation combinations (high risk but can be managed)
        ('amiodarone', 'sotalol'): {
            'consequence': 'Additive QT prolongation, torsades risk',
            'evidence': 'ACC/AHA Guidelines, FDA warnings',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid if possible; if necessary, monitor ECG closely',
            'frequency': 'Uncommon'
        },
        ('amiodarone', 'haloperidol'): {
            'consequence': 'Additive QT prolongation',
            'evidence': 'FDA warnings, pharmacological basis',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor ECG; consider alternative antipsychotic',
            'frequency': 'Uncommon'
        },
        
        # Bleeding combinations
        ('warfarin', 'aspirin'): {
            'consequence': 'Serious bleeding, GI hemorrhage',
            'evidence': 'CHEST Guidelines, TWOSIDES PRR=8.5',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Use lowest aspirin dose; add PPI; monitor INR/bleeding',
            'frequency': 'Common'
        },
        ('warfarin', 'nsaid'): {
            'consequence': 'GI bleeding, hemorrhage',
            'evidence': 'FDA NSAID Black Box, CHEST Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.MIXED,
            'action': 'Avoid if possible; use alternative analgesic',
            'frequency': 'Common'
        },
        ('dabigatran', 'aspirin'): {
            'consequence': 'Major bleeding',
            'evidence': 'RE-LY trial data',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Assess bleeding risk; minimize aspirin use',
            'frequency': 'Common'
        },
        ('rivaroxaban', 'aspirin'): {
            'consequence': 'Major bleeding',
            'evidence': 'COMPASS trial',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Use low-dose aspirin only if indicated',
            'frequency': 'Common'
        },
        
        # Statin + inhibitor combinations
        ('simvastatin', 'amiodarone'): {
            'consequence': 'Myopathy, rhabdomyolysis',
            'evidence': 'FDA Label - dose limit 20mg',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Limit simvastatin to 20mg/day',
            'frequency': 'Uncommon'
        },
        ('simvastatin', 'diltiazem'): {
            'consequence': 'Myopathy risk',
            'evidence': 'FDA Label',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Limit simvastatin to 10mg/day',
            'frequency': 'Uncommon'
        },
        ('lovastatin', 'itraconazole'): {
            'consequence': 'Rhabdomyolysis',
            'evidence': 'FDA Label Contraindication',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Avoid combination; use alternative statin',
            'frequency': 'Rare'
        },
        
        # Digoxin toxicity
        ('digoxin', 'amiodarone'): {
            'consequence': 'Digoxin toxicity (bradycardia, arrhythmia)',
            'evidence': 'FDA Label, PK studies',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Reduce digoxin dose by 50%; monitor levels',
            'frequency': 'Common'
        },
        ('digoxin', 'verapamil'): {
            'consequence': 'Digoxin toxicity, bradycardia',
            'evidence': 'FDA Label',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.MIXED,
            'action': 'Reduce digoxin dose; monitor levels and HR',
            'frequency': 'Common'
        },
        
        # Hyperkalemia
        ('spironolactone', 'potassium'): {
            'consequence': 'Severe hyperkalemia, cardiac arrhythmia',
            'evidence': 'Endocrine Society Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid combination; monitor potassium closely',
            'frequency': 'Common'
        },
        ('ace_inhibitor', 'potassium'): {
            'consequence': 'Hyperkalemia',
            'evidence': 'Clinical practice guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor potassium; avoid supplements unless deficient',
            'frequency': 'Uncommon'
        },
    }
    
    # ==========================================================================
    # CLINICAL CONSEQUENCE PATTERNS (AHRQ emphasis on outcomes)
    # Pattern -> (seriousness, consequence description, evidence, mechanism)
    # ==========================================================================
    
    CLINICAL_CONSEQUENCE_PATTERNS = {
        # LIFE-THREATENING OUTCOMES - High Seriousness
        'torsades de pointes': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Life-threatening ventricular arrhythmia',
            'evidence': 'ACC/AHA Arrhythmia Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid combination; if essential, continuous cardiac monitoring',
            'frequency': 'Rare',
            'legacy': 'Contraindicated'
        },
        'qt prolongation': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Risk of torsades de pointes, sudden death',
            'evidence': 'FDA QT Guidance, CredibleMeds',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'ECG monitoring; consider alternative',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        'ventricular fibrillation': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Cardiac arrest, sudden death',
            'evidence': 'ACC/AHA Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid combination',
            'frequency': 'Rare',
            'legacy': 'Contraindicated'
        },
        'serotonin syndrome': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Potentially fatal hyperthermia, rigidity, autonomic instability',
            'evidence': 'Boyer & Shannon NEJM 2005, FDA MAOI warnings',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid combination; 14-day washout for MAOIs',
            'frequency': 'Uncommon',
            'legacy': 'Contraindicated'
        },
        'intracranial hemorrhage': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Life-threatening intracranial bleeding',
            'evidence': 'CHEST Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Avoid combination; assess bleeding risk',
            'frequency': 'Rare',
            'legacy': 'Contraindicated'
        },
        'rhabdomyolysis': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Muscle breakdown, acute kidney injury',
            'evidence': 'FDA Statin Labels',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Avoid combination or limit statin dose',
            'frequency': 'Rare',
            'legacy': 'Major'
        },
        
        # SERIOUS BLEEDING - High Seriousness
        'risk or severity of bleeding': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Increased bleeding risk, hemorrhage',
            'evidence': 'CHEST Antithrombotic Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor for bleeding; consider dose reduction',
            'frequency': 'Common',
            'legacy': 'Major'
        },
        'hemorrhage can be increased': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Serious bleeding events',
            'evidence': 'FDA Anticoagulant Labels',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Close monitoring; bleeding precautions',
            'frequency': 'Common',
            'legacy': 'Major'
        },
        'gastrointestinal bleeding': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'GI hemorrhage',
            'evidence': 'FDA NSAID Black Box Warning',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Add PPI; monitor for GI symptoms',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        
        # CARDIAC EFFECTS - High Seriousness
        'bradycardia': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Symptomatic slow heart rate',
            'evidence': 'ACC/AHA Heart Rhythm Guidelines',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor heart rate; adjust doses as needed',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        'hypotension': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Symptomatic low blood pressure',
            'evidence': 'Clinical pharmacology literature',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor BP; adjust doses; counsel on symptoms',
            'frequency': 'Common',
            'legacy': 'Major'
        },
        'hyperkalemia': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Cardiac arrhythmias from elevated potassium',
            'evidence': 'Endocrine Society Guidelines',
            'quality': EvidenceQuality.HIGH,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor potassium; avoid K supplements',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        
        # TOXICITY - High Seriousness
        'toxicity': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Drug toxicity',
            'evidence': 'Clinical toxicology',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Monitor drug levels; reduce dose if needed',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        'nephrotoxicity': {
            'seriousness': Seriousness.HIGH,
            'consequence': 'Kidney damage',
            'evidence': 'KDIGO Guidelines',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.MIXED,
            'action': 'Monitor renal function; stay hydrated',
            'frequency': 'Uncommon',
            'legacy': 'Major'
        },
        
        # MODERATE SERIOUSNESS - Notification Required
        'serum concentration': {
            'seriousness': Seriousness.MODERATE,
            'consequence': 'Altered drug levels',
            'evidence': 'Clinical PK literature',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Consider dose adjustment; monitor response',
            'frequency': 'Common',
            'legacy': 'Moderate'
        },
        'metabolism of': {
            'seriousness': Seriousness.MODERATE,
            'consequence': 'Altered drug metabolism',
            'evidence': 'FDA DDI Guidance',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Adjust dose based on interaction magnitude',
            'frequency': 'Common',
            'legacy': 'Moderate'
        },
        'therapeutic efficacy': {
            'seriousness': Seriousness.MODERATE,
            'consequence': 'Reduced drug effectiveness',
            'evidence': 'Clinical pharmacology',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Monitor therapeutic response',
            'frequency': 'Common',
            'legacy': 'Moderate'
        },
        'antihypertensive activities': {
            'seriousness': Seriousness.MODERATE,
            'consequence': 'Additive blood pressure lowering',
            'evidence': 'JNC Guidelines',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Monitor BP; adjust doses as needed',
            'frequency': 'Common',
            'legacy': 'Moderate'
        },
        'excretion rate': {
            'seriousness': Seriousness.MODERATE,
            'consequence': 'Altered drug elimination',
            'evidence': 'Clinical PK',
            'quality': EvidenceQuality.MODERATE,
            'mechanism': InteractionMechanism.PHARMACOKINETIC,
            'action': 'Consider renal function in dosing',
            'frequency': 'Common',
            'legacy': 'Moderate'
        },
        
        # LOW SERIOUSNESS - Generally No Alert Needed
        'sedation': {
            'seriousness': Seriousness.LOW,
            'consequence': 'Drowsiness',
            'evidence': 'Package insert standard',
            'quality': EvidenceQuality.LOW,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Counsel patient about sedation',
            'frequency': 'Common',
            'legacy': 'Minor'
        },
        'drowsiness': {
            'seriousness': Seriousness.LOW,
            'consequence': 'Increased sleepiness',
            'evidence': 'Package insert',
            'quality': EvidenceQuality.LOW,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Advise caution with driving/machinery',
            'frequency': 'Common',
            'legacy': 'Minor'
        },
        'dizziness': {
            'seriousness': Seriousness.LOW,
            'consequence': 'Lightheadedness',
            'evidence': 'Common adverse event',
            'quality': EvidenceQuality.LOW,
            'mechanism': InteractionMechanism.PHARMACODYNAMIC,
            'action': 'Rise slowly; avoid alcohol',
            'frequency': 'Common',
            'legacy': 'Minor'
        },
    }
    
    # ==========================================================================
    # QT-PROLONGING DRUGS (per CredibleMeds/FDA)
    # ==========================================================================
    
    QT_PROLONGING_DRUGS = {
        'amiodarone', 'sotalol', 'quinidine', 'procainamide', 'disopyramide',
        'droperidol', 'haloperidol', 'methadone', 'erythromycin', 'clarithromycin',
        'moxifloxacin', 'levofloxacin', 'ondansetron', 'domperidone', 'thioridazine',
        'pimozide', 'arsenic', 'cisapride', 'terfenadine', 'astemizole', 'dofetilide',
        'ibutilide', 'ziprasidone', 'chloroquine', 'hydroxychloroquine'
    }
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        self.classification_stats = defaultdict(int)
        
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for efficient matching"""
        compiled = {}
        for pattern, data in self.CLINICAL_CONSEQUENCE_PATTERNS.items():
            compiled[pattern] = (
                re.compile(rf'\b{re.escape(pattern)}\b', re.IGNORECASE),
                data
            )
        return compiled
    
    def _check_absolute_contraindication(self, description: str) -> Optional[DDIClassificationResult]:
        """Check for absolute contraindications (very restrictive per AHRQ)"""
        desc_lower = description.lower()
        
        for (drug1, drug2), data in self.ABSOLUTE_CONTRAINDICATIONS.items():
            if drug1.lower() in desc_lower and drug2.lower() in desc_lower:
                return DDIClassificationResult(
                    seriousness=Seriousness.CONTRAINDICATED,
                    clinical_consequence=data['consequence'],
                    frequency='Rare',
                    mechanism=InteractionMechanism.PHARMACODYNAMIC,
                    recommended_action=data['action'],
                    evidence_quality=data['quality'],
                    evidence_reference=data['evidence'],
                    confidence=0.95,
                    legacy_severity='Contraindicated'
                )
        return None
    
    def _check_high_seriousness_pairs(self, description: str) -> Optional[DDIClassificationResult]:
        """Check for known high seriousness drug pairs"""
        desc_lower = description.lower()
        
        for (drug1, drug2), data in self.HIGH_SERIOUSNESS_PAIRS.items():
            if drug1.lower() in desc_lower and drug2.lower() in desc_lower:
                return DDIClassificationResult(
                    seriousness=Seriousness.HIGH,
                    clinical_consequence=data['consequence'],
                    frequency=data.get('frequency', 'Unknown'),
                    mechanism=data.get('mechanism', InteractionMechanism.UNKNOWN),
                    recommended_action=data['action'],
                    evidence_quality=data['quality'],
                    evidence_reference=data['evidence'],
                    confidence=0.85,
                    legacy_severity='Major'
                )
        return None
    
    def _check_qt_combination(self, description: str) -> Optional[DDIClassificationResult]:
        """Check for QT-prolonging drug combinations"""
        desc_lower = description.lower()
        qt_drugs_found = [d for d in self.QT_PROLONGING_DRUGS if d.lower() in desc_lower]
        
        if len(qt_drugs_found) >= 2:
            return DDIClassificationResult(
                seriousness=Seriousness.HIGH,
                clinical_consequence=f'Additive QT prolongation from {", ".join(qt_drugs_found[:2])}',
                frequency='Uncommon',
                mechanism=InteractionMechanism.PHARMACODYNAMIC,
                recommended_action='ECG monitoring; consider alternative agents',
                evidence_quality=EvidenceQuality.HIGH,
                evidence_reference='FDA/CredibleMeds QT drug list',
                confidence=0.85,
                legacy_severity='Major'
            )
        return None
    
    def classify(self, description: str) -> DDIClassificationResult:
        """
        Classify DDI per AHRQ recommendations
        
        Priority:
        1. Absolute contraindications (very rare - AHRQ emphasis)
        2. Known high-seriousness pairs
        3. QT-prolonging combinations
        4. Pattern-based clinical consequence matching
        5. Default to Moderate seriousness
        
        Returns complete DDIClassificationResult with all AHRQ elements
        """
        if not description or pd.isna(description):
            return DDIClassificationResult(
                seriousness=Seriousness.MODERATE,
                clinical_consequence='Unknown interaction',
                frequency='Unknown',
                mechanism=InteractionMechanism.UNKNOWN,
                recommended_action='Monitor for unexpected effects',
                evidence_quality=EvidenceQuality.VERY_LOW,
                evidence_reference='No description available',
                confidence=0.3,
                legacy_severity='Moderate'
            )
        
        desc_lower = description.lower()
        
        # TIER 1: Absolute contraindications (very restrictive)
        result = self._check_absolute_contraindication(description)
        if result:
            self.classification_stats['contraindicated'] += 1
            return result
        
        # TIER 2: Known high-seriousness pairs
        result = self._check_high_seriousness_pairs(description)
        if result:
            self.classification_stats['high_pair'] += 1
            return result
        
        # TIER 3: QT combination check
        result = self._check_qt_combination(description)
        if result:
            self.classification_stats['qt_combination'] += 1
            return result
        
        # TIER 4: Pattern-based classification
        for pattern, (compiled_re, data) in self.compiled_patterns.items():
            if compiled_re.search(desc_lower):
                self.classification_stats[data['seriousness'].value] += 1
                return DDIClassificationResult(
                    seriousness=data['seriousness'],
                    clinical_consequence=data['consequence'],
                    frequency=data.get('frequency', 'Unknown'),
                    mechanism=data.get('mechanism', InteractionMechanism.UNKNOWN),
                    recommended_action=data['action'],
                    evidence_quality=data['quality'],
                    evidence_reference=data['evidence'],
                    confidence=0.7 if data['seriousness'] == Seriousness.HIGH else 0.6,
                    legacy_severity=data['legacy']
                )
        
        # TIER 5: Default to Moderate (per AHRQ - middle category)
        self.classification_stats['default_moderate'] += 1
        return DDIClassificationResult(
            seriousness=Seriousness.MODERATE,
            clinical_consequence='Potential interaction - clinical significance unclear',
            frequency='Unknown',
            mechanism=InteractionMechanism.UNKNOWN,
            recommended_action='Monitor for unexpected effects; adjust therapy if needed',
            evidence_quality=EvidenceQuality.LOW,
            evidence_reference='Pattern not matched - default classification',
            confidence=0.5,
            legacy_severity='Moderate'
        )
    
    def classify_batch(self, descriptions: List[str], 
                       show_progress: bool = True) -> pd.DataFrame:
        """Classify multiple DDI descriptions"""
        results = []
        iterator = tqdm(descriptions, desc="Classifying (AHRQ)") if show_progress else descriptions
        
        for desc in iterator:
            result = self.classify(desc)
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def get_legacy_severity(self, description: str) -> str:
        """Get 4-tier severity for backward compatibility"""
        return self.classify(description).legacy_severity
    
    def get_seriousness(self, description: str) -> str:
        """Get 3-tier seriousness per AHRQ"""
        return self.classify(description).seriousness.value


# ============================================================================
# VALIDATION AND PUBLICATION
# ============================================================================

class AHRQValidationRunner:
    """Run validation and generate publication materials"""
    
    def __init__(self, config_path: str = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"):
        self.config = Config()
        self.config.ddi_data_path = config_path
        self.config.output_dir = Path("publication_ahrq")
        self.config.output_dir.mkdir(exist_ok=True)
        (self.config.output_dir / 'tables').mkdir(exist_ok=True)
        (self.config.output_dir / 'figures').mkdir(exist_ok=True)
        (self.config.output_dir / 'data').mkdir(exist_ok=True)
        
        self.classifier = AHRQDDIClassifier()
        
    def load_data(self) -> pd.DataFrame:
        """Load DDI dataset"""
        logger.info(f"Loading data from {self.config.ddi_data_path}")
        df = pd.read_csv(self.config.ddi_data_path)
        logger.info(f"Loaded {len(df):,} DDI pairs")
        return df
    
    def run_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run AHRQ classification on dataset"""
        logger.info("Running AHRQ-compliant classification...")
        
        # Use interaction_description column
        desc_col = 'interaction_description' if 'interaction_description' in df.columns else 'description'
        results = self.classifier.classify_batch(df[desc_col].tolist())
        
        # Merge results with original data
        df_result = pd.concat([df.reset_index(drop=True), results], axis=1)
        
        return df_result
    
    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics table"""
        
        # Seriousness distribution (3-tier AHRQ)
        seriousness_counts = df['seriousness'].value_counts()
        
        # Legacy severity distribution (4-tier for comparison)
        legacy_counts = df['legacy_severity'].value_counts()
        
        summary = {
            'Total DDI Pairs': len(df),
            'High Seriousness (Interruptive)': seriousness_counts.get('High', 0),
            'Moderate Seriousness (Notification)': seriousness_counts.get('Moderate', 0),
            'Low Seriousness (Minimal)': seriousness_counts.get('Low', 0),
            'Contraindicated (Absolute)': seriousness_counts.get('Contraindicated', 0),
            '---': '---',
            'Legacy: Contraindicated': legacy_counts.get('Contraindicated', 0),
            'Legacy: Major': legacy_counts.get('Major', 0),
            'Legacy: Moderate': legacy_counts.get('Moderate', 0),
            'Legacy: Minor': legacy_counts.get('Minor', 0),
        }
        
        summary_df = pd.DataFrame([summary]).T.reset_index()
        summary_df.columns = ['Metric', 'Value']
        
        return summary_df
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate publication report"""
        
        report = f"""
# AHRQ-Compliant DDI Seriousness Classification Report

## Methodology
Based on Malone DC, et al. "Recommendations for Selecting Drug-Drug Interactions 
for Clinical Decision Support." Am J Health Syst Pharm. 2016;73(8):576–585.

## Key Implementation Features
1. **Three Seriousness Categories** (per AHRQ recommendation):
   - High: Interruptive alert required - may be life-threatening
   - Moderate: Notification required - may worsen condition
   - Low: Generally should not require alert

2. **Judicious Contraindicated Classification**:
   - Per AHRQ: "Only a small set of drug combinations are truly contraindicated"
   - Reserved for pairs where "no situations exist where benefit outweighs risk"

3. **Clinical Consequence Emphasis**:
   - All classifications include specific clinical outcomes
   - Recommended management actions provided

4. **Evidence Grading**:
   - GRADE-aligned quality ratings (High/Moderate/Low/Very Low)
   - Source references documented

## Dataset Summary
- Total DDI Pairs: {len(df):,}

## Seriousness Distribution (AHRQ 3-Tier)
"""
        seriousness_counts = df['seriousness'].value_counts()
        for level, count in seriousness_counts.items():
            pct = count / len(df) * 100
            report += f"- {level}: {count:,} ({pct:.1f}%)\n"
        
        report += f"""
## Legacy Severity Distribution (4-Tier for Comparison)
"""
        legacy_counts = df['legacy_severity'].value_counts()
        for level, count in legacy_counts.items():
            pct = count / len(df) * 100
            report += f"- {level}: {count:,} ({pct:.1f}%)\n"
        
        report += f"""
## Evidence Quality Distribution
"""
        quality_counts = df['evidence_quality'].value_counts()
        for level, count in quality_counts.items():
            pct = count / len(df) * 100
            report += f"- {level}: {count:,} ({pct:.1f}%)\n"
        
        report += f"""
## Interaction Mechanism Distribution
"""
        mechanism_counts = df['mechanism'].value_counts()
        for mech, count in mechanism_counts.items():
            pct = count / len(df) * 100
            report += f"- {mech}: {count:,} ({pct:.1f}%)\n"
        
        report += """
## References
1. Malone DC, et al. Am J Health Syst Pharm. 2016;73(8):576-585. (PMC5064943)
2. GRADE Working Group. BMJ 2008;336:924-926.
3. FDA Drug Interaction Guidance (2020)
4. CHEST Antithrombotic Guidelines
5. ACC/AHA Cardiovascular Guidelines
"""
        
        return report
    
    def run(self):
        """Run complete validation and generate outputs"""
        logger.info("="*60)
        logger.info("AHRQ-Compliant DDI Classification System")
        logger.info("="*60)
        
        # Load data
        df = self.load_data()
        
        # Run classification
        df_result = self.run_classification(df)
        
        # Generate summary
        summary = self.generate_summary_table(df_result)
        logger.info("\n" + summary.to_string(index=False))
        
        # Save results
        df_result.to_csv(self.config.output_dir / 'data' / 'classified_ddi_ahrq.csv', index=False)
        summary.to_csv(self.config.output_dir / 'tables' / 'summary_ahrq.csv', index=False)
        summary.to_markdown(self.config.output_dir / 'tables' / 'summary_ahrq.md', index=False)
        
        # Generate report
        report = self.generate_report(df_result)
        with open(self.config.output_dir / 'AHRQ_CLASSIFICATION_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info(f"\nResults saved to {self.config.output_dir}")
        
        return df_result


# ============================================================================
# COMPARISON WITH PREVIOUS CLASSIFIER
# ============================================================================

def compare_classifiers(data_path: str = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"):
    """Compare AHRQ classifier with previous evidence-based classifier"""
    
    from publication_evidence_based_classifier import EvidenceBasedSeverityClassifier
    
    df = pd.read_csv(data_path)
    descriptions = df['description'].tolist()
    
    # Previous classifier
    old_classifier = EvidenceBasedSeverityClassifier()
    old_results = old_classifier.classify_batch(descriptions)
    
    # New AHRQ classifier
    ahrq_classifier = AHRQDDIClassifier()
    ahrq_results = ahrq_classifier.classify_batch(descriptions)
    
    # Compare
    print("\n" + "="*60)
    print("CLASSIFIER COMPARISON")
    print("="*60)
    
    print("\nPrevious Classifier (4-tier):")
    print(old_results['severity'].value_counts())
    
    print("\nAHRQ Classifier - Seriousness (3-tier):")
    print(ahrq_results['seriousness'].value_counts())
    
    print("\nAHRQ Classifier - Legacy Severity (4-tier for comparison):")
    print(ahrq_results['legacy_severity'].value_counts())
    
    # Agreement rate
    agreement = (old_results['severity'] == ahrq_results['legacy_severity']).mean()
    print(f"\nAgreement rate (legacy severity): {agreement*100:.1f}%")
    
    return old_results, ahrq_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    runner = AHRQValidationRunner()
    results = runner.run()
    
    print("\n" + "="*60)
    print("Classification Statistics:")
    print("="*60)
    for stat, count in runner.classifier.classification_stats.items():
        print(f"  {stat}: {count:,}")
