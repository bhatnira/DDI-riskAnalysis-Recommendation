#!/usr/bin/env python3
"""
Generate Sample Outputs for DDI Risk Analysis Application

This script runs sample drug combinations through the analysis pipeline
and saves the results for publication materials.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from ddi_app
from ddi_app import kg, current_analysis

def run_sample_analysis(drugs_input, case_name):
    """Run analysis on a drug combination and return results"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {case_name}")
    print(f"Drugs: {drugs_input}")
    print('='*60)
    
    # Load KG if needed
    if not kg.loaded:
        print("Loading Knowledge Graph...")
        kg.load()
    
    # Identify drugs
    found_drugs, suggestions, not_found = kg.identify_drugs(drugs_input)
    
    results = {
        'case_name': case_name,
        'input': drugs_input,
        'timestamp': datetime.now().isoformat(),
        'found_drugs': [],
        'not_found': not_found,
        'suggestions': {k: [(m[0], float(m[1])) for m in v] for k, v in suggestions.items()},
        'interactions': [],
        'risk_score': 0,
        'risk_level': 'N/A',
        'severity_counts': {},
        'regimen_pri': {},
        'alternatives': {}
    }
    
    if len(found_drugs) < 2:
        print(f"Not enough drugs found for analysis. Found: {len(found_drugs)}")
        results['error'] = "Insufficient drugs for analysis"
        return results
    
    # Get drug info
    resolved = [d[2] for d in found_drugs]  # drug_data is third element
    results['found_drugs'] = [
        {
            'input_name': d[0],
            'resolved_name': d[1],
            'drugbank_id': d[2].get('drugbank_id', ''),
            'atc_codes': d[2].get('atc_codes', [])
        }
        for d in found_drugs
    ]
    
    # Get drug IDs
    drug_ids = [d.get('drugbank_id', '') for d in resolved if d.get('drugbank_id')]
    
    # Calculate risk score
    risk_score, interactions, counts = kg.calculate_risk_score(drug_ids)
    
    results['risk_score'] = float(risk_score)
    results['risk_level'] = "CRITICAL" if risk_score >= 0.8 else "HIGH" if risk_score >= 0.5 else "MODERATE" if risk_score >= 0.2 else "LOW"
    results['severity_counts'] = counts
    
    # Add interaction details
    for interaction in interactions:
        d1_id = interaction.get('drug1_id', '')
        d2_id = interaction.get('drug2_id', '')
        d1_data = kg.drugs_by_id.get(d1_id, {})
        d2_data = kg.drugs_by_id.get(d2_id, {})
        
        results['interactions'].append({
            'drug1': d1_data.get('name', d1_id),
            'drug2': d2_data.get('name', d2_id),
            'severity': interaction.get('severity', 'unknown'),
            'description': interaction.get('description', 'No description available')[:200]
        })
    
    # Calculate PRI
    regimen_pri = kg.calculate_regimen_pri(drug_ids)
    if regimen_pri:
        results['regimen_pri'] = {
            'drug_pris': {
                name: {'pri': float(data.get('pri', 0)), 'risk_level': data.get('risk_level', 'Unknown')}
                for name, data in regimen_pri.get('drug_pris', {}).items()
            },
            'regimen_pri': float(regimen_pri.get('regimen_pri', 0))
        }
    
    # Find alternatives for high-risk drugs
    candidate_drugs = set()
    for i in interactions:
        sev = i.get('severity', '').lower()
        if 'contraindicated' in sev or 'major' in sev:
            candidate_drugs.add(i.get('drug1_id', ''))
            candidate_drugs.add(i.get('drug2_id', ''))
    
    # Also add high-PRI drugs
    if regimen_pri and 'drug_pris' in regimen_pri:
        for drug_name, pri_data in regimen_pri.get('drug_pris', {}).items():
            if pri_data.get('pri', 0) > 0.5:
                for did in drug_ids:
                    ddata = kg.drugs_by_id.get(did, {})
                    if ddata.get('name', '').lower() == drug_name.lower():
                        candidate_drugs.add(did)
                        break
    
    for drug_id in candidate_drugs:
        if drug_id in drug_ids:
            other_ids = [d for d in drug_ids if d != drug_id]
            alternatives = kg.find_alternatives_with_ars(drug_id, other_ids)
            if alternatives:
                drug_name = kg.drugs_by_id.get(drug_id, {}).get('name', drug_id)
                results['alternatives'][drug_name] = [
                    {
                        'name': alt.get('name', ''),
                        'ars': float(alt.get('ars', 0)),
                        'atc_code': alt.get('atc_code', ''),
                        'interaction_count': alt.get('interaction_count', 0),
                        'max_severity': alt.get('max_severity', 'unknown')
                    }
                    for alt in alternatives[:5]  # Top 5 alternatives
                ]
    
    # Print summary
    print(f"\nResults:")
    print(f"  Risk Level: {results['risk_level']} ({results['risk_score']:.2%})")
    print(f"  Interactions Found: {len(results['interactions'])}")
    print(f"  Severity Distribution: {results['severity_counts']}")
    print(f"  Drugs with Alternatives: {list(results['alternatives'].keys())}")
    
    return results


def main():
    """Run sample analyses and save results"""
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample cases
    sample_cases = [
        {
            'name': 'High-Risk Cardiovascular Regimen',
            'drugs': 'warfarin, aspirin, atorvastatin, lisinopril, metoprolol'
        },
        {
            'name': 'Anticoagulation Risk Scenario',
            'drugs': 'warfarin, aspirin, ibuprofen'
        },
        {
            'name': 'Diabetes Management',
            'drugs': 'metformin, glipizide, lisinopril, atorvastatin'
        },
        {
            'name': 'Pain Management with CNS Risk',
            'drugs': 'tramadol, sertraline, alprazolam'
        },
        {
            'name': 'Cardiac with PDE5 Inhibitor',
            'drugs': 'sildenafil, nitroglycerin, metoprolol'
        },
        {
            'name': 'Polypharmacy Elderly Patient',
            'drugs': 'warfarin, digoxin, furosemide, potassium, lisinopril, atorvastatin'
        }
    ]
    
    all_results = []
    
    print("\n" + "="*70)
    print("  DDI Risk Analysis - Sample Output Generation")
    print("="*70)
    
    for case in sample_cases:
        try:
            result = run_sample_analysis(case['drugs'], case['name'])
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {case['name']}: {e}")
            all_results.append({
                'case_name': case['name'],
                'input': case['drugs'],
                'error': str(e)
            })
    
    # Save all results
    output_file = os.path.join(output_dir, 'sample_analyses.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, 'analysis_summary.md')
    with open(summary_file, 'w') as f:
        f.write("# DDI Risk Analysis - Sample Case Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in all_results:
            f.write(f"## {result.get('case_name', 'Unknown')}\n\n")
            f.write(f"**Input Drugs:** {result.get('input', 'N/A')}\n\n")
            
            if 'error' in result:
                f.write(f"**Error:** {result['error']}\n\n")
                continue
            
            f.write(f"**Risk Level:** {result.get('risk_level', 'N/A')} ({result.get('risk_score', 0):.1%})\n\n")
            
            # Found drugs
            if result.get('found_drugs'):
                f.write("### Identified Drugs\n\n")
                f.write("| Input | Resolved | DrugBank ID |\n")
                f.write("|-------|----------|-------------|\n")
                for drug in result['found_drugs']:
                    f.write(f"| {drug['input_name']} | {drug['resolved_name']} | {drug['drugbank_id']} |\n")
                f.write("\n")
            
            # Interactions
            if result.get('interactions'):
                f.write("### Drug-Drug Interactions\n\n")
                f.write("| Drug 1 | Drug 2 | Severity |\n")
                f.write("|--------|--------|----------|\n")
                for inter in result['interactions']:
                    f.write(f"| {inter['drug1']} | {inter['drug2']} | {inter['severity']} |\n")
                f.write("\n")
            
            # PRI scores
            if result.get('regimen_pri', {}).get('drug_pris'):
                f.write("### Polypharmacy Risk Index (PRI)\n\n")
                f.write("| Drug | PRI Score | Risk Level |\n")
                f.write("|------|-----------|------------|\n")
                for drug, data in result['regimen_pri']['drug_pris'].items():
                    f.write(f"| {drug.title()} | {data['pri']:.3f} | {data['risk_level']} |\n")
                f.write("\n")
            
            # Alternatives
            if result.get('alternatives'):
                f.write("### Recommended Alternatives\n\n")
                for drug, alts in result['alternatives'].items():
                    f.write(f"**For {drug.title()}:**\n\n")
                    f.write("| Alternative | ARS Score | Max Severity with Regimen |\n")
                    f.write("|-------------|-----------|---------------------------|\n")
                    for alt in alts[:3]:
                        f.write(f"| {alt['name'].title()} | {alt['ars']:.3f} | {alt['max_severity']} |\n")
                    f.write("\n")
            
            f.write("---\n\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    main()
