# Project Summary for AI Agent Prompt

## Project Overview
**AI-based Polypharmacy Risk-aware Drug Recommender System** - A publication-ready system for analyzing drug-drug interactions (DDIs) and recommending safer therapeutic alternatives for cardiovascular/antithrombotic medications.

**Target Journal:** Journal of Managed Care & Specialty Pharmacy (JMCP) - IF ~3.5

**GitHub Repository:** https://github.com/bhatnira/DDI-riskAnalysis-Recommendation

---

## Dataset
- **File:** `ddi_cardio_or_antithrombotic_labeled (1).csv` (183MB, excluded from git)
- **Size:** 759,774 DDI records
- **Columns:** `drugbank_id_1`, `drug_name_1`, `atc_1`, `is_cardiovascular_1`, `is_antithrombotic_1`, `drugbank_id_2`, `drug_name_2`, `atc_2`, `is_cardiovascular_2`, `is_antithrombotic_2`, `interaction_description`, `severity_label`, `severity_confidence`, `severity_numeric`
- **4,314 unique drugs**, **43,523 unique interaction mechanisms**

---

## Critical Issue Identified
**Severity labels are AI-generated** using Facebook BART zero-shot classification (`severity_label`, `severity_confidence`). Using these labels to train/validate ML models creates **circular validation** - not suitable for publication.

**Solution:** Network-based risk assessment using **graph topology only** (degree centrality, betweenness, weighted connections) - no dependency on AI-generated labels.

---

## Architecture

### Core Agents (`/agents/`)
| File | Purpose |
|------|---------|
| `orchestrator.py` | Central pipeline controller |
| `interaction_agent.py` | Detects DDIs from medication lists |
| `severity_agent.py` | ML-based severity classification |
| `alternative_agent.py` | Finds safer alternatives via ATC |
| `explanation_agent.py` | Generates clinical reports |
| `drug_risk_network.py` | Graph-based risk network |
| `recommender.py` | Multi-objective optimization |

### New Modules (Created for JMCP Publication)
| File | Purpose |
|------|---------|
| `faers_integration.py` | FDA FAERS API integration for external validation |
| `run_faers_validation.py` | Complete FAERS validation runner |
| `comprehensive_comparison.py` | Three-approach comparison (Algorithmic vs GNN-Severity vs GNN-Embedding) |
| `gnn_risk_assessment.py` | Graph Neural Network models (GAT) |

---

## Key Methodologies

### 1. Network-Based Risk Assessment (Publication-Ready)
- Uses **graph topology only** - no AI labels
- Metrics: Degree centrality, betweenness centrality, weighted degree
- Polypharmacy Risk Index (PRI) = weighted combination of centrality metrics

### 2. FAERS External Validation
- **API:** OpenFDA (https://api.fda.gov/drug/event.json)
- **Rate Limit:** 240 requests/minute
- Queries: Total reports, serious events, death reports, concomitant drug events
- Provides **independent external validation** from real-world adverse events

### 3. Three-Approach Comparison
1. **Algorithmic-Greedy:** Rule-based using ATC codes + network centrality
2. **GNN-Severity:** Graph Attention Network using severity labels (circular - for comparison only)
3. **GNN-Embedding:** PubMedBERT embeddings + GAT (avoids label dependency)

---

## Technical Stack
- **Python 3.14** on macOS (Apple Silicon/MPS)
- **PyTorch 2.10.0** with PyTorch Geometric 2.7.0
- **Embedding Model:** PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
- **GNN:** Graph Attention Network (GAT) with 2 layers

---

## Current Status

### Completed ✅
- Three-approach comparison framework
- Network-based risk assessment (no AI labels)
- FAERS integration module
- PubMedBERT embedding integration
- GAT model implementation
- GitHub repository setup

### In Progress ⏳
- FAERS validation correlation (API queries working, need diverse risk samples)
- Publication figures/tables generation

### Known Issues
- FAERS concomitant drug queries return sparse data for uncommon drug pairs
- Network risk scores cluster at high values (need threshold adjustment for diversity)
- Serious event ratio calculation fixed (using report counts, not reaction counts)

---

## File Structure
```
PhRMA-Paper/
├── agents/                    # Core agent modules
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── interaction_agent.py
│   ├── severity_agent.py
│   ├── alternative_agent.py
│   ├── explanation_agent.py
│   ├── drug_risk_network.py
│   ├── recommender.py
│   ├── faers_integration.py   # NEW: FAERS API client
│   ├── run_faers_validation.py # NEW: Validation runner
│   └── comprehensive_comparison.py
├── main.py                    # CLI entry point
├── paper_materials/           # Publication drafts
├── publication/               # Publication outputs (figures removed)
├── .gitignore                 # Excludes large files (>100MB)
└── README.md
```

---

## Git Ignore (Large Files Excluded)
```
ddi_cardio_or_antithrombotic_labeled*.csv
publication/data/all_interactions_with_metrics.csv
outputs/embedding_cache.pkl
cache/
.cache/
results/
outputs/
```

---

## Key Commands
```bash
# Run main analysis
python3 main.py --drugs "Warfarin,Aspirin,Metoprolol" --no-llm

# Run FAERS validation
python3 agents/run_faers_validation.py --sample-size 40

# Run comprehensive comparison
python3 agents/comprehensive_comparison.py
```

---

## Publication Requirements for JMCP
1. **No circular validation** - avoid using AI-generated severity labels
2. **External validation** - FAERS adverse event correlation
3. **Transparent methodology** - network topology + ATC-based recommendations
4. **Clinical applicability** - focus on cardiovascular/antithrombotic drugs
5. **Reproducible results** - all code in GitHub repository

---

## FAERS API Details

### Endpoints Used
```python
# Base URL
BASE_URL = "https://api.fda.gov/drug/event.json"

# Drug adverse events
search = f'patient.drug.medicinalproduct:"{drug_name}"'
count = 'patient.reaction.reactionmeddrapt.exact'

# Serious events
search = f'patient.drug.medicinalproduct:"{drug_name}" AND serious:1'

# Death reports
search = f'patient.drug.medicinalproduct:"{drug_name}" AND seriousnessdeath:1'

# Concomitant drugs (interaction detection)
search = f'patient.drug.medicinalproduct:"{drug1}" AND patient.drug.medicinalproduct:"{drug2}"'
```

### Data Classes
```python
@dataclass
class FAERSDrugProfile:
    drug_name: str
    total_reports: int
    serious_reports: int
    death_reports: int
    adverse_events: List[FAERSAdverseEvent]
    top_signals: List[str]
    faers_risk_score: float
    serious_event_ratio: float

@dataclass
class FAERSInteractionSignal:
    drug1: str
    drug2: str
    concomitant_reports: int
    drug1_alone_reports: int
    drug2_alone_reports: int
    interaction_signal_score: float
    common_adverse_events: List[str]
```

---

## Network Risk Metrics

### Polypharmacy Risk Index (PRI)
```python
PRI = 0.4 * degree_centrality + 0.3 * betweenness_centrality + 0.3 * weighted_degree
```

### High-Risk Pair Detection
```python
# Pair risk score based on network topology
pair_risk = (node1_centrality + node2_centrality) / 2 * edge_weight_normalized
```

---

## Sample Drug Lists for Testing
```python
SAMPLE_DRUG_LISTS = {
    'cardiovascular_basic': ['Warfarin', 'Aspirin', 'Metoprolol', 'Lisinopril', 'Atorvastatin'],
    'cardiovascular_combo': ['Warfarin', 'Clopidogrel', 'Aspirin', 'Heparin'],
    'heart_failure': ['Digoxin', 'Furosemide', 'Spironolactone', 'Carvedilol', 'Lisinopril'],
    'hypertension': ['Amlodipine', 'Lisinopril', 'Hydrochlorothiazide', 'Metoprolol'],
    'diabetes_cardiac': ['Metformin', 'Glipizide', 'Atorvastatin', 'Lisinopril', 'Aspirin']
}
```

---

## Next Steps for Publication
1. Complete FAERS validation with diverse risk samples
2. Generate publication-quality figures
3. Write methods section describing network-based approach
4. Prepare supplementary materials with full validation results
5. Submit to JMCP

---

*Last Updated: February 16, 2026*
