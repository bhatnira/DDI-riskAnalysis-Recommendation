#!/usr/bin/env python3
"""
Combine FAERS and DDInter Evidence for Fair Severity Target Values

This script combines evidence from two gold-standard sources:
1. FAERS (FDA Adverse Event Reporting System) - Real-world pharmacovigilance data
2. DDInter (Drug-Drug Interaction Database) - Literature-curated severity labels

Output: Combined "fair value" target distribution for DDI severity classification
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# Data Sources
# =============================================================================

# DDInter severity distribution (from ddinter_all.csv analysis)
# DDInter uses 3 levels: Major, Moderate, Minor (no Contraindicated category)
DDINTER_RAW = {
    'Major': 67234,
    'Moderate': 258942,
    'Minor': 21722,
    'Unknown': 94364  # Will exclude from percentage calculation
}

# Calculate DDInter percentages (excluding Unknown)
ddinter_total = sum([DDINTER_RAW['Major'], DDINTER_RAW['Moderate'], DDINTER_RAW['Minor']])
DDINTER_DISTRIBUTION = {
    'Major': DDINTER_RAW['Major'] / ddinter_total * 100,
    'Moderate': DDINTER_RAW['Moderate'] / ddinter_total * 100,
    'Minor': DDINTER_RAW['Minor'] / ddinter_total * 100
}

# FAERS severity distribution (from OpenFDA API comprehensive analysis)
# Based on MedWatch 3500 outcome categories per 21 CFR 314.80
FAERS_DISTRIBUTION = {
    'Death_or_LifeThreatening': 20.6,    # Maps to → Contraindicated
    'Hospitalization_or_Disability': 37.1,  # Maps to → Major
    'Other_Serious': 30.0,                  # Maps to → Moderate
    'Non_Serious': 12.3                     # Maps to → Minor
}

# =============================================================================
# Mapping Functions
# =============================================================================

def map_ddinter_to_4level():
    """
    Map DDInter 3-level scale to 4-level scale.
    
    DDInter doesn't have a Contraindicated category.
    Strategy: Estimate Contraindicated from FAERS death/life-threatening ratio
    """
    # DDInter Major includes what we would call both Contraindicated and Major
    # Use FAERS ratio of Death/Life-Threatening vs Hospitalization to split
    faers_severe_ratio = FAERS_DISTRIBUTION['Death_or_LifeThreatening'] / (
        FAERS_DISTRIBUTION['Death_or_LifeThreatening'] + 
        FAERS_DISTRIBUTION['Hospitalization_or_Disability']
    )
    
    # Split DDInter Major into Contraindicated and Major
    ddinter_contraindicated = DDINTER_DISTRIBUTION['Major'] * faers_severe_ratio
    ddinter_major = DDINTER_DISTRIBUTION['Major'] * (1 - faers_severe_ratio)
    
    return {
        'Contraindicated': ddinter_contraindicated,
        'Major': ddinter_major,
        'Moderate': DDINTER_DISTRIBUTION['Moderate'],
        'Minor': DDINTER_DISTRIBUTION['Minor']
    }

def map_faers_to_4level():
    """
    Map FAERS outcomes to DDI 4-level severity scale.
    
    Mapping rationale based on FDA MedWatch 3500 outcome definitions:
    - Death/Life-Threatening → Contraindicated (should never co-prescribe)
    - Hospitalization/Disability → Major (significant clinical intervention)
    - Other Serious → Moderate (medical attention needed)
    - Non-Serious → Minor (monitor/no intervention)
    """
    return {
        'Contraindicated': FAERS_DISTRIBUTION['Death_or_LifeThreatening'],
        'Major': FAERS_DISTRIBUTION['Hospitalization_or_Disability'],
        'Moderate': FAERS_DISTRIBUTION['Other_Serious'],
        'Minor': FAERS_DISTRIBUTION['Non_Serious']
    }

# =============================================================================
# Combination Methods
# =============================================================================

def simple_average(ddinter_4level, faers_4level):
    """Simple arithmetic average of both sources."""
    return {
        level: (ddinter_4level[level] + faers_4level[level]) / 2
        for level in ['Contraindicated', 'Major', 'Moderate', 'Minor']
    }

def weighted_average(ddinter_4level, faers_4level, ddinter_weight=0.5, faers_weight=0.5):
    """Weighted average with customizable weights."""
    return {
        level: ddinter_4level[level] * ddinter_weight + faers_4level[level] * faers_weight
        for level in ['Contraindicated', 'Major', 'Moderate', 'Minor']
    }

def conservative_estimate(ddinter_4level, faers_4level):
    """
    Conservative estimate: Use lower severe estimates, higher minor estimates.
    
    Rationale: FAERS has reporting bias toward serious events, so we 
    downweight severe categories from FAERS and trust DDInter more for 
    moderate/minor categories.
    """
    return {
        # For Contraindicated/Major: Use lower of the two (less aggressive)
        'Contraindicated': min(ddinter_4level['Contraindicated'], faers_4level['Contraindicated']),
        'Major': min(ddinter_4level['Major'], faers_4level['Major']),
        # For Moderate/Minor: Use simple average
        'Moderate': (ddinter_4level['Moderate'] + faers_4level['Moderate']) / 2,
        'Minor': (ddinter_4level['Minor'] + faers_4level['Minor']) / 2
    }

def clinical_judgment_combined(ddinter_4level, faers_4level):
    """
    Clinical judgment approach: Account for biases in each source.
    
    - FAERS overestimates severe events (reporting bias)
    - DDInter may underestimate highest-risk (no Contraindicated category)
    
    Strategy: 
    - Contraindicated: Use FAERS death rate scaled by 0.25 (reporting bias correction)
    - Major: Average of both, scaled to account for Contraindicated
    - Moderate: Trust DDInter more (literature-validated)
    - Minor: Trust DDInter more (FAERS under-reports minor events)
    """
    # FAERS death/life-threatening (20.6%) is inflated by ~4x due to reporting bias
    # Typical real-world Contraindicated DDI rate is ~5%
    contraindicated = faers_4level['Contraindicated'] * 0.25
    
    # Major combines DDInter Major with scaled FAERS hospitalization
    # FAERS hospitalization (37.1%) is inflated by ~1.5x
    major_ddinter = ddinter_4level['Major']
    major_faers = faers_4level['Major'] * 0.67
    major = (major_ddinter + major_faers) / 2
    
    # Moderate: DDInter is more reliable here
    moderate_ddinter = ddinter_4level['Moderate']
    moderate_faers = faers_4level['Moderate']
    moderate = moderate_ddinter * 0.7 + moderate_faers * 0.3
    
    # Minor: DDInter is much more reliable (FAERS severely under-reports)
    minor = ddinter_4level['Minor']
    
    # Normalize to 100%
    total = contraindicated + major + moderate + minor
    return {
        'Contraindicated': contraindicated / total * 100,
        'Major': major / total * 100,
        'Moderate': moderate / total * 100,
        'Minor': minor / total * 100
    }

def normalize_distribution(dist):
    """Normalize a distribution to sum to 100%."""
    total = sum(dist.values())
    return {k: v / total * 100 for k, v in dist.items()}

# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 80)
    print("COMBINING FAERS AND DDInter EVIDENCE FOR FAIR SEVERITY TARGET VALUES")
    print("=" * 80)
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # -------------------------------------------------------------------------
    # Source Distributions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SOURCE DATA")
    print("-" * 80)
    
    print("\n📊 DDInter Database (Original 3-level):")
    print(f"   Total pairs analyzed: {ddinter_total:,} (excluding {DDINTER_RAW['Unknown']:,} Unknown)")
    for level, pct in DDINTER_DISTRIBUTION.items():
        print(f"   {level}: {pct:.1f}%")
    
    print("\n📊 FAERS Pharmacovigilance (4-level outcomes):")
    for outcome, pct in FAERS_DISTRIBUTION.items():
        print(f"   {outcome}: {pct}%")
    
    # -------------------------------------------------------------------------
    # Mapped to 4-Level Scale
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("MAPPED TO 4-LEVEL DDI SEVERITY SCALE")
    print("-" * 80)
    
    ddinter_4level = map_ddinter_to_4level()
    faers_4level = map_faers_to_4level()
    
    print("\n📌 DDInter → 4-Level Mapping:")
    for level, pct in ddinter_4level.items():
        print(f"   {level}: {pct:.1f}%")
    
    print("\n📌 FAERS → 4-Level Mapping:")
    for level, pct in faers_4level.items():
        print(f"   {level}: {pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # Combined Estimates
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("COMBINED FAIR VALUE ESTIMATES")
    print("-" * 80)
    
    # Method 1: Simple Average
    simple_avg = normalize_distribution(simple_average(ddinter_4level, faers_4level))
    print("\n1️⃣ Simple Average (50/50 weight):")
    for level, pct in simple_avg.items():
        print(f"   {level}: {pct:.1f}%")
    
    # Method 2: DDInter-weighted (trust literature more)
    ddinter_weighted = normalize_distribution(weighted_average(ddinter_4level, faers_4level, 0.6, 0.4))
    print("\n2️⃣ DDInter-Weighted (60/40):")
    for level, pct in ddinter_weighted.items():
        print(f"   {level}: {pct:.1f}%")
    
    # Method 3: Conservative
    conservative = normalize_distribution(conservative_estimate(ddinter_4level, faers_4level))
    print("\n3️⃣ Conservative Estimate:")
    for level, pct in conservative.items():
        print(f"   {level}: {pct:.1f}%")
    
    # Method 4: Clinical Judgment
    clinical = clinical_judgment_combined(ddinter_4level, faers_4level)
    print("\n4️⃣ Clinical Judgment (Bias-Corrected):")
    for level, pct in clinical.items():
        print(f"   {level}: {pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # Final Recommended Fair Value
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDED FAIR VALUE TARGETS")
    print("=" * 80)
    
    # Average of all methods to get consensus
    all_methods = [simple_avg, ddinter_weighted, conservative, clinical]
    consensus = {}
    for level in ['Contraindicated', 'Major', 'Moderate', 'Minor']:
        values = [m[level] for m in all_methods]
        consensus[level] = np.mean(values)
    
    # Normalize consensus
    consensus = normalize_distribution(consensus)
    
    print("\n🎯 CONSENSUS FAIR VALUE (Average of all methods):")
    print("-" * 40)
    for level, pct in consensus.items():
        print(f"   {level}: {pct:.1f}%")
    
    # Practical rounded targets
    print("\n📋 PRACTICAL ROUNDED TARGETS:")
    print("-" * 40)
    rounded_targets = {
        'Contraindicated': round(consensus['Contraindicated']),
        'Major': round(consensus['Major']),
        'Moderate': round(consensus['Moderate']),
        'Minor': round(consensus['Minor'])
    }
    
    # Adjust to sum to 100
    diff = 100 - sum(rounded_targets.values())
    rounded_targets['Moderate'] += diff  # Adjust moderate as it's the largest category
    
    for level, pct in rounded_targets.items():
        print(f"   {level}: {pct}%")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    results = {
        'analysis_date': datetime.now().isoformat(),
        'source_data': {
            'ddinter': {
                'total_pairs': ddinter_total,
                'original_distribution': DDINTER_DISTRIBUTION,
                'mapped_4level': ddinter_4level
            },
            'faers': {
                'original_distribution': FAERS_DISTRIBUTION,
                'mapped_4level': faers_4level
            }
        },
        'combination_methods': {
            'simple_average': simple_avg,
            'ddinter_weighted_60_40': ddinter_weighted,
            'conservative': conservative,
            'clinical_judgment': clinical
        },
        'fair_value_consensus': consensus,
        'recommended_targets': rounded_targets
    }
    
    output_path = Path('/home/nbhatta1/Desktop/copyOfOriginal/external_data/faers_ddinter_combined.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # -------------------------------------------------------------------------
    # Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: All Methods")
    print("=" * 80)
    
    table_data = {
        'Source/Method': ['DDInter (mapped)', 'FAERS (mapped)', 
                          'Simple Avg', 'DDInter-Weighted', 
                          'Conservative', 'Clinical', 'FAIR VALUE'],
        'Contraindicated': [f"{ddinter_4level['Contraindicated']:.1f}", 
                           f"{faers_4level['Contraindicated']:.1f}",
                           f"{simple_avg['Contraindicated']:.1f}",
                           f"{ddinter_weighted['Contraindicated']:.1f}",
                           f"{conservative['Contraindicated']:.1f}",
                           f"{clinical['Contraindicated']:.1f}",
                           f"{rounded_targets['Contraindicated']}"],
        'Major': [f"{ddinter_4level['Major']:.1f}", 
                  f"{faers_4level['Major']:.1f}",
                  f"{simple_avg['Major']:.1f}",
                  f"{ddinter_weighted['Major']:.1f}",
                  f"{conservative['Major']:.1f}",
                  f"{clinical['Major']:.1f}",
                  f"{rounded_targets['Major']}"],
        'Moderate': [f"{ddinter_4level['Moderate']:.1f}", 
                    f"{faers_4level['Moderate']:.1f}",
                    f"{simple_avg['Moderate']:.1f}",
                    f"{ddinter_weighted['Moderate']:.1f}",
                    f"{conservative['Moderate']:.1f}",
                    f"{clinical['Moderate']:.1f}",
                    f"{rounded_targets['Moderate']}"],
        'Minor': [f"{ddinter_4level['Minor']:.1f}", 
                  f"{faers_4level['Minor']:.1f}",
                  f"{simple_avg['Minor']:.1f}",
                  f"{ddinter_weighted['Minor']:.1f}",
                  f"{conservative['Minor']:.1f}",
                  f"{clinical['Minor']:.1f}",
                  f"{rounded_targets['Minor']}"]
    }
    
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    
    # Save table as CSV
    csv_path = Path('/home/nbhatta1/Desktop/copyOfOriginal/external_data/faers_ddinter_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table saved to: {csv_path}")
    
    return results

if __name__ == "__main__":
    results = main()
