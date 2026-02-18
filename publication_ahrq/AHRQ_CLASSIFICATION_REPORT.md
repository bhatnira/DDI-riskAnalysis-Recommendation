
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
- Total DDI Pairs: 759,774

## Seriousness Distribution (AHRQ 3-Tier)
- Moderate: 663,182 (87.3%)
- High: 95,046 (12.5%)
- Low: 1,546 (0.2%)

## Legacy Severity Distribution (4-Tier for Comparison)
- Moderate: 663,182 (87.3%)
- Major: 94,416 (12.4%)
- Minor: 1,546 (0.2%)
- Contraindicated: 630 (0.1%)

## Evidence Quality Distribution
- Moderate: 402,004 (52.9%)
- Low: 274,694 (36.2%)
- High: 83,076 (10.9%)

## Interaction Mechanism Distribution
- Pharmacokinetic: 330,320 (43.5%)
- Unknown: 273,148 (36.0%)
- Pharmacodynamic: 154,742 (20.4%)
- Mixed: 1,564 (0.2%)

## References
1. Malone DC, et al. Am J Health Syst Pharm. 2016;73(8):576-585. (PMC5064943)
2. GRADE Working Group. BMJ 2008;336:924-926.
3. FDA Drug Interaction Guidance (2020)
4. CHEST Antithrombotic Guidelines
5. ACC/AHA Cardiovascular Guidelines
