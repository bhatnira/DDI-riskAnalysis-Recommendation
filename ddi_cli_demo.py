#!/usr/bin/env python3
"""
DDI Risk Analysis - Command Line Demo
All data is pre-loaded from local knowledge graph (no external API calls)
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*70)
    print("  DDI Risk Analysis - Command Line Demo")
    print("  (All data from local Knowledge Graph - no external APIs)")
    print("="*70)
    
    # Load knowledge graph
    print("\n📂 Loading Knowledge Graph from local CSV files...")
    
    try:
        from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
        kg_loader = KnowledgeGraphLoader()
        assessor = PolypharmacyRiskAssessor(kg_loader)
        print(f"   ✅ Loaded {len(kg_loader.drug_name_to_id):,} drugs")
        print(f"   ✅ Loaded {len(kg_loader.ddi_index):,} drug-drug interactions")
    except Exception as e:
        print(f"   ❌ Error loading KG: {e}")
        return
    
    # Load extended drug data
    print("\n📂 Loading extended drug information...")
    import pandas as pd
    kg_path = os.path.join(os.path.dirname(__file__), 'knowledge_graph_fact_based', 'neo4j_export')
    
    drug_data = {}
    try:
        drugs_df = pd.read_csv(os.path.join(kg_path, 'drugs.csv'))
        for _, row in drugs_df.iterrows():
            name = str(row.get('name', '')).lower()
            drug_data[name] = {
                'drugbank_id': row.get('drugbank_id', ''),
                'name': row.get('name', ''),
                'indication': row.get('indication', ''),
                'mechanism': row.get('mechanism_of_action', ''),
                'pharmacodynamics': row.get('pharmacodynamics', ''),
                'atc_codes': row.get('atc_codes', ''),
                'groups': row.get('groups', '')
            }
        print(f"   ✅ Extended info for {len(drug_data):,} drugs")
    except Exception as e:
        print(f"   ⚠️ Could not load extended data: {e}")
    
    # Interactive mode
    while True:
        print("\n" + "-"*70)
        print("Enter drug names (comma-separated) or 'quit' to exit")
        print("Examples: warfarin, aspirin, ibuprofen")
        print("          metformin, lisinopril, amlodipine")
        print("-"*70)
        
        user_input = input("\n🔍 Drugs to analyze: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        drugs = [d.strip().lower() for d in user_input.replace(';', ',').split(',') if d.strip()]
        
        if len(drugs) < 2:
            print("❌ Please enter at least 2 drugs to check interactions")
            continue
        
        print(f"\n{'='*70}")
        print(f"  ANALYSIS FOR: {', '.join(drugs).upper()}")
        print(f"{'='*70}")
        
        # Get risk assessment
        try:
            result = assessor.assess_polypharmacy_risk(drugs)
            
            # Risk Summary
            risk_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🔴', 'CRITICAL': '🚨'}.get(result.risk_level, '⚪')
            print(f"\n{risk_emoji} RISK LEVEL: {result.risk_level}")
            print(f"   Risk Score: {result.overall_risk_score:.2%}")
            
            # Drug Profiles
            print(f"\n📋 DRUG PROFILES (from DrugBank):")
            print("-"*50)
            for drug in drugs:
                info = drug_data.get(drug, {})
                if info:
                    print(f"\n  💊 {info.get('name', drug).upper()}")
                    print(f"     DrugBank ID: {info.get('drugbank_id', 'N/A')}")
                    indication = info.get('indication', '')
                    if indication and str(indication) != 'nan':
                        print(f"     Indication: {str(indication)[:200]}...")
                    mechanism = info.get('mechanism', '')
                    if mechanism and str(mechanism) != 'nan':
                        print(f"     Mechanism: {str(mechanism)[:200]}...")
                else:
                    print(f"\n  💊 {drug.upper()} - No extended info available")
            
            # Interactions
            if result.ddi_pairs:
                print(f"\n⚠️  DRUG-DRUG INTERACTIONS FOUND: {len(result.ddi_pairs)}")
                print("-"*50)
                for i, ddi in enumerate(result.ddi_pairs[:10], 1):
                    drug1 = ddi.get('drug1', 'Unknown')
                    drug2 = ddi.get('drug2', 'Unknown')
                    severity = ddi.get('severity', 'Unknown')
                    description = ddi.get('description', 'No description')
                    
                    sev_emoji = {'major': '🔴', 'moderate': '🟡', 'minor': '🟢'}.get(
                        severity.lower().split()[0] if severity else '', '⚪')
                    
                    print(f"\n  {i}. {drug1.upper()} ↔ {drug2.upper()}")
                    print(f"     {sev_emoji} Severity: {severity}")
                    print(f"     📝 {description[:300]}..." if len(str(description)) > 300 else f"     📝 {description}")
                
                if len(result.ddi_pairs) > 10:
                    print(f"\n  ... and {len(result.ddi_pairs) - 10} more interactions")
            else:
                print("\n✅ No direct drug-drug interactions found in database")
            
            # Side effects context
            if hasattr(result, 'shared_side_effects') and result.shared_side_effects:
                print(f"\n⚡ SHARED SIDE EFFECTS (from SIDER):")
                print("-"*50)
                for se in list(result.shared_side_effects)[:5]:
                    print(f"  • {se}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
