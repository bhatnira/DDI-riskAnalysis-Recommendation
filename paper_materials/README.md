# Risk Assessment and AI-Driven Risk-Aware Alternative Drug Recommendation System

## Paper Materials

This directory contains all generated materials for the paper:

**"Risk Assessment and AI-Driven Risk-Aware Alternative Drug Recommendation System"**

---

## Directory Structure

```
paper_materials/
├── generate_paper_analysis.py    # Main analysis generator
├── outputs/
│   ├── figures/                  # Publication-quality figures
│   ├── tables/                   # Data tables (CSV + Markdown)
│   └── data/                     # Raw analysis data (JSON)
└── README.md                     # This file
```

## Generated Figures

| Figure | Title | Description |
|--------|-------|-------------|
| **Fig 1** | Risk Assessment | PRI distribution, components, Pareto analysis, CV comparison |
| **Fig 2** | Severity Patterns | Severity distribution, breakdown, correlation, confidence |
| **Fig 3** | Recommendation Performance | Risk reduction, MOO scores, PRI delta, rank analysis |
| **Fig 4** | Alternative Discovery | ATC classes, availability, improvement potential |
| **Fig 5** | System Architecture | Complete system workflow diagram |

## Generated Tables

| Table | Title | Contents |
|-------|-------|----------|
| **Table 1** | Dataset Summary | DDI records, drug counts, severity breakdown |
| **Table 2** | Top 20 Risk Drugs | Highest PRI drugs with metrics |
| **Table 3** | Risk Zone Statistics | Drugs per risk zone with statistics |
| **Table 4** | MOO Weights | Multi-objective optimization configuration |

## Running the Generator

```bash
cd /Users/nb/Desktop/PhRMA-Paper
python paper_materials/generate_paper_analysis.py
```

## Paper Outline

### Abstract
We present a novel AI-driven drug recommendation system that combines network-based 
risk assessment with multi-objective optimization to provide safer medication alternatives 
for polypharmacy patients. Our Polypharmacy Risk Index (PRI) quantifies drug interaction 
risk using graph-theoretic metrics, while our recommender engine balances risk reduction, 
centrality improvement, and therapeutic equivalence.

### 1. Introduction
- Polypharmacy challenge in cardiovascular care
- Limitations of pairwise DDI checking
- Need for systemic risk assessment

### 2. Methods
- 2.1 Drug Risk Network Construction
- 2.2 Polypharmacy Risk Index (PRI)
- 2.3 Multi-Objective Recommendation Engine
- 2.4 ATC-Based Alternative Discovery

### 3. Results
- 3.1 Risk Distribution Analysis
- 3.2 Severity Pattern Characterization
- 3.3 Recommendation System Performance
- 3.4 Case Studies

### 4. Discussion
- Clinical utility
- Comparison with existing approaches
- Limitations

### 5. Conclusion
- Key contributions
- Future directions

---

## Key Metrics

### PRI Formula
```
PRI = 0.25×DC + 0.30×WD + 0.20×BC + 0.25×SP

Where:
- DC = Normalized Degree Centrality
- WD = Normalized Weighted Degree
- BC = Normalized Betweenness Centrality
- SP = Normalized Severity Profile
```

### Multi-Objective Score
```
Score = 0.35×RR + 0.20×CR + 0.25×PA - 0.20×NIP

Where:
- RR = Risk Reduction
- CR = Centrality Reduction
- PA = Phenotype Avoidance
- NIP = New Interaction Penalty
```
