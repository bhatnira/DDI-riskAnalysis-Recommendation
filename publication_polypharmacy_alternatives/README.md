# Polypharmacy Risk Assessment and Alternative Therapy Suggestion

Publication materials for the Polypharmacy Risk Index (PRI) and Multi-Objective Drug Recommendation System.

## Directory Structure

```
publication_polypharmacy_alternatives/
├── README.md                      # This file
├── methods.tex                    # Methods section (separate)
├── methods.pdf                    # Compiled methods (195 KB)
├── results.tex                    # Results section (separate)
├── results.pdf                    # Compiled results (5.8 MB)
├── methods_brief.tex              # Combined methods+results document
├── methods_brief.pdf              # Compiled combined (5.9 MB)
├── supplementary_materials.tex    # Supplementary materials
├── supplementary_materials.pdf    # Compiled supplementary (211 KB)
├── figures/                       # Publication figures (7 figures)
│   ├── polypharmacy_risk_escalation.*
│   ├── drug_risk_matrix.*
│   ├── severity_heatmap.*
│   ├── drug_alternatives_heatmap.*
│   ├── drug_substitution_network.*
│   ├── network_safe_alternatives.*
│   └── class_alternatives_summary.*
├── data/                          # Supporting data files
│   └── severity_distribution.csv
└── tables/                        # LaTeX table files
```

## Overview

This publication module presents two integrated components for managing drug-drug interactions in polypharmacy scenarios:

### 1. Polypharmacy Risk Index (PRI)

A network-based metric quantifying individual drug risk within multi-drug regimens:

$$\text{PRI}(d) = 0.25 \cdot C_{\text{degree}}(d) + 0.30 \cdot C_{\text{weighted}}(d) + 0.20 \cdot C_{\text{betweenness}}(d) + 0.25 \cdot S(d)$$

**Component Metrics:**
| Metric | Weight | Description |
|--------|--------|-------------|
| Degree Centrality | 0.25 | Number of interacting drugs |
| Weighted Degree | 0.30 | Severity-weighted interaction sum |
| Betweenness Centrality | 0.20 | Role in risk propagation pathways |
| Severity Profile Score | 0.25 | Proportion of severe interactions |

**Risk Classification:**
| PRI Score | Risk Level | Clinical Action |
|-----------|------------|-----------------|
| > 0.5 | High Risk | Immediate clinical review required |
| 0.3 - 0.5 | Medium Risk | Close monitoring warranted |
| < 0.3 | Lower Risk | Standard monitoring protocols |

### 2. Multi-Objective Drug Recommendation System

A framework for identifying therapeutic alternatives balancing efficacy and safety:

$$\text{RecScore}(d_{\text{alt}}) = 0.40 \cdot T(d_{\text{alt}}) + 0.35 \cdot \text{SAF}(d_{\text{alt}}) + 0.25 \cdot R(d_{\text{alt}})$$

**Score Components:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Therapeutic Similarity (T) | 0.40 | ATC matching, shared targets, disease overlap, pathway similarity |
| Safety Improvement (SAF) | 0.35 | Reduction in DDI risk with current regimen |
| Risk Reduction (R) | 0.25 | Net decrease in severe interactions and PRI |

## Validation Case Study

**Original High-Risk Cardiovascular Regimen:**
- Warfarin (Anticoagulant)
- Amiodarone (Antiarrhythmic)
- Digoxin (Cardiac Glycoside)
- Quinidine (Antiarrhythmic)
- Propranolol (Beta-blocker)

**Initial Risk Profile:** 10 severe interactions (2 contraindicated, 8 major)

**After Warfarin → Dabigatran Substitution:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Contraindicated | 2 | 1 | 50% reduction |
| Major | 8 | 6 | 25% reduction |
| Total Severe | 10 | 7 | 30% reduction |
| Average PRI | 0.569 | 0.504 | 11.5% improvement |

## Figures

1. **polypharmacy_risk_escalation** - Risk escalation with increasing drug count
2. **drug_risk_matrix** - High-risk drug combination matrix
3. **severity_heatmap** - Severity distribution across drug classes
4. **drug_alternatives_heatmap** - Recommendation scores for therapeutic substitution
5. **drug_substitution_network** - Network of recommended alternatives
6. **network_safe_alternatives** - Safe alternatives highlighting improved safety profiles
7. **class_alternatives_summary** - Class-level therapeutic substitution patterns

## Compilation

```bash
cd /home/nbhatta1/Desktop/copyOfOriginal-knowledge-graph/publication_polypharmacy_alternatives
pdflatex methods_brief.tex
pdflatex supplementary_materials.tex
```

## References

1. Wishart DS, et al. DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Res*. 2018;46(D1):D1074-D1082.
2. Xiong G, et al. DDInter: an online drug-drug interaction database. *Nucleic Acids Res*. 2022;50(D1):D1200-D1207.
3. American Geriatrics Society 2019 Beers Criteria Update Expert Panel. American Geriatrics Society 2019 Updated AGS Beers Criteria. *J Am Geriatr Soc*. 2019;67(4):674-694.
4. O'Mahony D, et al. STOPP/START criteria for potentially inappropriate prescribing in older people: version 2. *Age Ageing*. 2015;44(2):213-218.
