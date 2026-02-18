#!/usr/bin/env python3
"""
DDI Risk Analysis - Standalone Demo
All data from local Knowledge Graph CSV files (no external APIs)
"""
import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Configuration
KG_DIR = Path(__file__).parent / 'knowledge_graph_fact_based' / 'neo4j_export'

class DDIAnalyzer:
    """Self-contained DDI analyzer using local CSV data"""
    
    def __init__(self):
        self.drugs: Dict[str, dict] = {}
        self.drug_name_to_id: Dict[str, str] = {}
        self.ddis: Dict[Tuple[str, str], dict] = {}
        self.drug_side_effects: Dict[str, Set[str]] = defaultdict(set)
        self.side_effect_names: Dict[str, str] = {}
        self.drug_proteins: Dict[str, Set[str]] = defaultdict(set)
        self.protein_info: Dict[str, dict] = {}
        
    def load(self):
        """Load all data from CSV files"""
        print("\n📂 Loading Knowledge Graph from local CSV files...")
        print(f"   Path: {KG_DIR}")
        
        self._load_drugs()
        self._load_ddis()
        self._load_side_effects()
        self._load_proteins()
        
        print(f"\n✅ Knowledge Graph loaded successfully!")
        print(f"   • {len(self.drugs):,} drugs with full profiles")
        print(f"   • {len(self.ddis):,} drug-drug interactions")
        print(f"   • {len(self.side_effect_names):,} side effects")
        print(f"   • {len(self.protein_info):,} proteins")
        
    def _load_drugs(self):
        """Load drug profiles from DrugBank data"""
        path = KG_DIR / "drugs.csv"
        if not path.exists():
            print(f"   ❌ File not found: {path}")
            return
            
        df = pd.read_csv(path, low_memory=False)
        for _, row in df.iterrows():
            drug_id = row.get('drugbank_id', '')
            name = str(row.get('name', '')).lower()
            
            self.drugs[drug_id] = {
                'drugbank_id': drug_id,
                'name': row.get('name', ''),
                'indication': row.get('indication', ''),
                'mechanism': row.get('mechanism_of_action', ''),
                'pharmacodynamics': row.get('pharmacodynamics', ''),
                'atc_codes': row.get('atc_codes', ''),
                'groups': row.get('groups', ''),
                'categories': row.get('categories', ''),
                'description': row.get('description', '')
            }
            self.drug_name_to_id[name] = drug_id
            self.drug_name_to_id[drug_id.lower()] = drug_id
        
        # Add common drug name aliases
        aliases = {
            'aspirin': 'acetylsalicylic acid',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'aleve': 'naproxen',
            'coumadin': 'warfarin',
            'lipitor': 'atorvastatin',
            'zocor': 'simvastatin',
            'crestor': 'rosuvastatin',
            'nexium': 'esomeprazole',
            'prilosec': 'omeprazole',
            'zoloft': 'sertraline',
            'prozac': 'fluoxetine',
            'lexapro': 'escitalopram',
            'xanax': 'alprazolam',
            'valium': 'diazepam',
            'ativan': 'lorazepam',
            'ambien': 'zolpidem',
            'viagra': 'sildenafil',
            'cialis': 'tadalafil',
            'glucophage': 'metformin',
            'norvasc': 'amlodipine',
            'prinivil': 'lisinopril',
            'zestril': 'lisinopril',
            'lasix': 'furosemide',
            'plavix': 'clopidogrel',
            'synthroid': 'levothyroxine',
            'vicodin': 'hydrocodone',
            'oxycontin': 'oxycodone',
            'ultram': 'tramadol',
            'neurontin': 'gabapentin',
            'lyrica': 'pregabalin',
            'singulair': 'montelukast',
            'zyrtec': 'cetirizine',
            'claritin': 'loratadine',
            'benadryl': 'diphenhydramine',
            'pepcid': 'famotidine',
            'zantac': 'ranitidine',
        }
        for alias, real_name in aliases.items():
            if real_name in self.drug_name_to_id and alias not in self.drug_name_to_id:
                self.drug_name_to_id[alias] = self.drug_name_to_id[real_name]
            
        print(f"   ✅ Loaded {len(self.drugs):,} drug profiles")
        
    def _load_ddis(self):
        """Load drug-drug interactions"""
        path = KG_DIR / "ddi_edges.csv"
        if not path.exists():
            print(f"   ❌ File not found: {path}")
            return
            
        df = pd.read_csv(path, low_memory=False)
        for _, row in df.iterrows():
            drug1 = row.get('drug1_id', '')
            drug2 = row.get('drug2_id', '')
            
            ddi = {
                'drug1_id': drug1,
                'drug2_id': drug2,
                'description': row.get('description', ''),
                'severity': row.get('severity', 'Unknown'),
                'source': row.get('source', 'DrugBank')
            }
            
            self.ddis[(drug1, drug2)] = ddi
            self.ddis[(drug2, drug1)] = ddi
            
        print(f"   ✅ Loaded {len(df):,} DDI edges")
        
    def _load_side_effects(self):
        """Load side effects from SIDER data"""
        # Load drug-side effect relationships
        edge_path = KG_DIR / "drug_side_effect_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path, low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                se_id = row.get('side_effect_id', '')
                self.drug_side_effects[drug_id].add(se_id)
            print(f"   ✅ Loaded {len(df):,} drug-side effect links")
        
        # Load side effect names
        se_path = KG_DIR / "side_effects.csv"
        if se_path.exists():
            df = pd.read_csv(se_path, low_memory=False)
            for _, row in df.iterrows():
                se_id = row.get('umls_cui', row.get('id', ''))
                name = row.get('name', row.get('side_effect_name', ''))
                self.side_effect_names[se_id] = name
            print(f"   ✅ Loaded {len(self.side_effect_names):,} side effect names")
                
    def _load_proteins(self):
        """Load protein targets from UniProt data"""
        # Load drug-protein relationships
        edge_path = KG_DIR / "drug_protein_edges.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path, low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                protein_id = row.get('protein_id', '')
                self.drug_proteins[drug_id].add(protein_id)
            print(f"   ✅ Loaded {len(df):,} drug-protein links")
        
        # Load protein info
        protein_path = KG_DIR / "proteins.csv"
        if protein_path.exists():
            df = pd.read_csv(protein_path, low_memory=False)
            for _, row in df.iterrows():
                protein_id = row.get('protein_id', row.get('id', ''))
                self.protein_info[protein_id] = {
                    'name': row.get('name', ''),
                    'uniprot_id': row.get('uniprot_id', ''),
                    'gene_name': row.get('gene_name', ''),
                    'general_function': row.get('general_function', ''),
                    'specific_function': row.get('specific_function', '')
                }
            print(f"   ✅ Loaded {len(self.protein_info):,} protein profiles")
    
    def resolve_drug(self, name: str) -> str:
        """Resolve drug name to DrugBank ID"""
        name_lower = name.lower().strip()
        return self.drug_name_to_id.get(name_lower, '')
    
    def analyze(self, drug_names: List[str]) -> dict:
        """Analyze drug interactions"""
        # Resolve drug names
        resolved = []
        unresolved = []
        for name in drug_names:
            drug_id = self.resolve_drug(name)
            if drug_id:
                resolved.append((name, drug_id))
            else:
                unresolved.append(name)
        
        if unresolved:
            print(f"\n⚠️  Could not find: {', '.join(unresolved)}")
            
        if len(resolved) < 2:
            return {'error': 'Need at least 2 recognized drugs'}
        
        # Get drug profiles
        profiles = []
        for name, drug_id in resolved:
            profiles.append(self.drugs.get(drug_id, {}))
        
        # Find interactions
        interactions = []
        drug_ids = [d[1] for d in resolved]
        for i in range(len(drug_ids)):
            for j in range(i + 1, len(drug_ids)):
                key = (drug_ids[i], drug_ids[j])
                if key in self.ddis:
                    ddi = self.ddis[key]
                    interactions.append({
                        'drug1': resolved[i][0],
                        'drug2': resolved[j][0],
                        'severity': ddi.get('severity', 'Unknown'),
                        'description': ddi.get('description', '')
                    })
        
        # Find shared side effects
        shared_se = None
        if len(drug_ids) >= 2:
            se_sets = [self.drug_side_effects.get(did, set()) for did in drug_ids]
            if all(se_sets):
                shared_se = se_sets[0].intersection(*se_sets[1:])
        
        # Calculate risk (simple heuristic)
        risk_score = 0
        for ddi in interactions:
            sev = ddi.get('severity', '').lower()
            if 'major' in sev or 'contraindicated' in sev:
                risk_score += 0.4
            elif 'moderate' in sev:
                risk_score += 0.2
            else:
                risk_score += 0.1
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.6:
            risk_level = 'HIGH'
        elif risk_score >= 0.3:
            risk_level = 'MODERATE'
        elif risk_score > 0:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'drugs': resolved,
            'profiles': profiles,
            'interactions': interactions,
            'shared_side_effects': shared_se,
            'risk_score': risk_score,
            'risk_level': risk_level
        }


def print_analysis(analyzer: DDIAnalyzer, result: dict):
    """Pretty print analysis results"""
    if 'error' in result:
        print(f"\n❌ {result['error']}")
        return
    
    risk_emoji = {
        'MINIMAL': '🟢', 'LOW': '🟡', 
        'MODERATE': '🟠', 'HIGH': '🔴'
    }.get(result['risk_level'], '⚪')
    
    print(f"\n{'='*70}")
    print(f"  {risk_emoji} RISK LEVEL: {result['risk_level']} ({result['risk_score']:.0%})")
    print(f"{'='*70}")
    
    # Drug Profiles
    print(f"\n📋 DRUG PROFILES (Source: DrugBank)")
    print("-" * 60)
    
    for name, drug_id in result['drugs']:
        profile = analyzer.drugs.get(drug_id, {})
        print(f"\n💊 {profile.get('name', name).upper()}")
        print(f"   DrugBank ID: {drug_id}")
        print(f"   URL: https://go.drugbank.com/drugs/{drug_id}")
        
        indication = profile.get('indication', '')
        if indication and str(indication) != 'nan':
            ind_text = str(indication)[:400]
            if len(str(indication)) > 400:
                ind_text += "..."
            print(f"\n   📌 INDICATION:")
            for line in ind_text.split('\n')[:5]:
                if line.strip():
                    print(f"      {line.strip()[:100]}")
        
        mechanism = profile.get('mechanism', '')
        if mechanism and str(mechanism) != 'nan':
            mech_text = str(mechanism)[:400]
            if len(str(mechanism)) > 400:
                mech_text += "..."
            print(f"\n   ⚙️  MECHANISM OF ACTION:")
            for line in mech_text.split('\n')[:5]:
                if line.strip():
                    print(f"      {line.strip()[:100]}")
    
    # Interactions
    print(f"\n\n⚠️  DRUG-DRUG INTERACTIONS ({len(result['interactions'])} found)")
    print("-" * 60)
    
    if result['interactions']:
        for i, ddi in enumerate(result['interactions'][:15], 1):
            sev = ddi.get('severity', 'Unknown')
            sev_emoji = '🔴' if 'major' in sev.lower() else ('🟡' if 'moderate' in sev.lower() else '🟢')
            
            print(f"\n{i}. {ddi['drug1'].upper()} ↔ {ddi['drug2'].upper()}")
            print(f"   {sev_emoji} Severity: {sev}")
            
            desc = ddi.get('description', 'No description available')
            if desc and str(desc) != 'nan':
                desc_text = str(desc)[:500]
                if len(str(desc)) > 500:
                    desc_text += "..."
                print(f"   📝 {desc_text}")
        
        if len(result['interactions']) > 15:
            print(f"\n   ... and {len(result['interactions']) - 15} more interactions")
    else:
        print("\n   ✅ No direct interactions found in database")
    
    # Shared Side Effects
    if result['shared_side_effects']:
        print(f"\n\n⚡ SHARED SIDE EFFECTS (Source: SIDER)")
        print("-" * 60)
        for se_id in list(result['shared_side_effects'])[:10]:
            se_name = analyzer.side_effect_names.get(se_id, se_id)
            print(f"   • {se_name}")
        if len(result['shared_side_effects']) > 10:
            print(f"   ... and {len(result['shared_side_effects']) - 10} more")


def main():
    print("\n" + "=" * 70)
    print("  💊 DDI RISK ANALYSIS - OFFLINE DEMO")
    print("  All data from local Knowledge Graph (DrugBank, SIDER, UniProt)")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = DDIAnalyzer()
    analyzer.load()
    
    # Demo queries
    demo_drugs = [
        "warfarin, aspirin, ibuprofen",
        "metformin, lisinopril, amlodipine",
        "sertraline, tramadol"
    ]
    
    print("\n" + "=" * 70)
    print("  RUNNING DEMO ANALYSES")
    print("=" * 70)
    
    for drug_input in demo_drugs:
        print(f"\n\n{'#' * 70}")
        print(f"  QUERY: {drug_input}")
        print(f"{'#' * 70}")
        
        drugs = [d.strip() for d in drug_input.split(',')]
        result = analyzer.analyze(drugs)
        print_analysis(analyzer, result)
    
    # Interactive mode
    print("\n\n" + "=" * 70)
    print("  INTERACTIVE MODE")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 50)
        print("Enter drug names (comma-separated) or 'quit' to exit")
        print("-" * 50)
        
        try:
            user_input = input("\n🔍 Drugs: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if user_input.lower() in ['quit', 'exit', 'q', '']:
            print("\nGoodbye! 👋")
            break
        
        drugs = [d.strip() for d in user_input.split(',')]
        result = analyzer.analyze(drugs)
        print_analysis(analyzer, result)


if __name__ == '__main__':
    main()
