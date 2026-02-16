# 🏥 AI-based Polypharmacy Risk-aware Drug Recommender System

An agentic, modular architecture for drug-drug interaction (DDI) analysis and polypharmacy risk assessment, implementing the methodology from the paper "AI-based Polypharmacy Risk-aware Drug Recommender System".

## 📋 Table of Contents

- [Overview](#overview)
- [Paper Methodology](#paper-methodology)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Agent Details](#agent-details)
- [API Reference](#api-reference)

## 🎯 Overview

This system analyzes drug-drug interactions and provides:
- **Drug Risk Network**: Graph-based DDI network with centrality metrics
- **Polypharmacy Risk Index (PRI)**: Paper methodology for risk quantification
- **Multi-Objective Recommender**: Ranks alternatives by risk reduction + centrality + phenotype avoidance
- **Interaction Detection**: Identifies all pairwise DDIs from a medication list
- **Severity Classification**: ML-based severity prediction (Contraindicated, Major, Moderate, Minor)
- **Risk Assessment**: Overall polypharmacy risk scoring
- **Alternative Recommendations**: ATC-based safer drug alternatives
- **Clinical Reports**: Human-readable reports with optional BioMistral-7B LLM

## 📊 Paper Methodology

### 1. Drug Risk Network Construction
- Nodes = Drugs, Edges = Interactions weighted by severity
- Severity weights: Contraindicated=10, Major=7, Moderate=4, Minor=1
- Computes degree centrality, weighted degree, and betweenness centrality

### 2. Polypharmacy Risk Index (PRI)
```
PRI = 0.25×(Degree Centrality) + 0.30×(Weighted Degree) + 0.20×(Betweenness) + 0.25×(Severity Profile)
```

### 3. Multi-Objective Recommender Algorithm
Ranks alternatives using four weighted objectives:
- **Risk Reduction (35%)**: PRI delta between original and alternative drug
- **Centrality Reduction (20%)**: Network centrality improvement
- **Phenotype Avoidance (25%)**: Avoiding harmful interaction phenotypes
- **New Interaction Penalty (20%)**: Minimizing new severe interactions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                           │
│              (Central Pipeline Controller)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Input (Drug List)                                        │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────┐                                       │
│   │ 🔍 InteractionAgent │ ──► Detect all DDIs                   │
│   │   • Drug validation │                                       │
│   │   • Pairwise lookup │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│   ┌─────────────────────┐                                       │
│   │ ⚠️  SeverityAgent   │ ──► Classify severity, compute risk   │
│   │   • ML prediction   │                                       │
│   │   • Risk scoring    │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│   ┌─────────────────────┐                                       │
│   │ 💊 AlternativeAgent │ ──► Find safer alternatives           │
│   │   • ATC matching    │                                       │
│   │   • Safety scoring  │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│   ┌─────────────────────┐                                       │
│   │ 📝 ExplanationAgent │ ──► Generate reports                  │
│   │   • Clinical report │                                       │
│   │   • Patient summary │                                       │
│   └─────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│   ┌─────────────────────┐                                       │
│   │   Final Output      │                                       │
│   └─────────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone or navigate to the project directory
cd /path/to/PhRMA-Paper

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Command Line

```bash
# Analyze specific drugs
python main.py --drugs "Warfarin,Aspirin,Metoprolol,Lisinopril"

# Interactive mode
python main.py --interactive

# Use sample drug list
python main.py --sample cardiovascular_basic

# Train ML model and analyze
python main.py --drugs "Warfarin,Aspirin" --train-model

# Save reports to files
python main.py --drugs "Warfarin,Aspirin" --save-report analysis_report
```

### Python API

```python
from agents import OrchestratorAgent
import pandas as pd

# Load DDI database
df = pd.read_csv('ddi_cardio_or_antithrombotic_labeled (1).csv')

# Initialize the orchestrator
orchestrator = OrchestratorAgent(verbose=True)
orchestrator.initialize(df, train_severity_model=False)

# Analyze medications
result = orchestrator.analyze_drugs([
    'Warfarin', 
    'Aspirin', 
    'Metoprolol', 
    'Lisinopril'
])

# Get reports
print(result['reports'])           # Clinical report
print(result['patient_summary'])   # Patient-friendly summary
print(result['structured_output']) # JSON for integration
```

## 📖 Usage

### Interactive Mode

```bash
python main.py --interactive
```

Commands:
- `analyze <drug1>, <drug2>, ...` - Analyze drug interactions
- `sample <name>` - Use predefined sample drug list
- `samples` - List available sample lists
- `help` - Show help
- `quit` - Exit

### Available Sample Drug Lists

| Name | Drugs |
|------|-------|
| `cardiovascular_basic` | Warfarin, Aspirin, Metoprolol, Lisinopril, Atorvastatin |
| `cardiovascular_combo` | Warfarin, Clopidogrel, Aspirin, Heparin |
| `heart_failure` | Digoxin, Furosemide, Spironolactone, Carvedilol, Lisinopril |
| `hypertension` | Amlodipine, Lisinopril, Hydrochlorothiazide, Metoprolol |
| `diabetes_cardiac` | Metformin, Glipizide, Atorvastatin, Lisinopril, Aspirin |

### Output Formats

- **clinical** - Full clinical report with severity details
- **patient** - Patient-friendly summary
- **json** - Structured JSON for integration
- **all** - All formats (default)

## 🤖 Agent Details

### BaseAgent
Abstract base class providing:
- Status tracking (IDLE, RUNNING, SUCCESS, FAILED, WAITING)
- Message passing between agents
- Execution timing and error handling

### InteractionAgent 🔍
- **Input**: List of drug names
- **Output**: All detected DDIs with details
- **Features**:
  - Drug name validation and fuzzy matching
  - Bidirectional interaction lookup
  - Cardiovascular drug flagging (includes antithrombotic drugs)

### SeverityAgent ⚠️
- **Input**: Interactions from InteractionAgent
- **Output**: Severity classifications and risk scores
- **Features**:
  - ML-based severity prediction (GradientBoosting)
  - Rule-based fallback classification
  - Overall risk score calculation (0-100)

### AlternativeAgent 💊
- **Input**: Problematic drugs and current medication list
- **Output**: Safer alternative recommendations
- **Features**:
  - ATC-code based therapeutic matching
  - Multi-level class search (anatomical → chemical)
  - Safety scoring based on interaction profiles

### ExplanationAgent 📝
- **Input**: All analysis results
- **Output**: Human-readable reports
- **Features**:
  - Clinical report generation
  - Patient-friendly summaries
  - Structured JSON for system integration

### OrchestratorAgent 🎯
- Central pipeline coordinator
- Manages execution flow between agents
- Aggregates results and handles errors

## 📚 API Reference

### OrchestratorAgent

```python
# Initialize
orchestrator = OrchestratorAgent(verbose=True)
orchestrator.initialize(ddi_dataframe, train_severity_model=False)

# Analyze
result = orchestrator.analyze_drugs(['Drug1', 'Drug2', ...])

# Quick summary
summary = orchestrator.get_quick_summary(['Drug1', 'Drug2'])

# Get execution log
log = orchestrator.get_execution_log()

# Reset for new analysis
orchestrator.reset()
```

### Result Structure

```python
result = {
    'success': True/False,
    'data': {
        'pipeline_results': {
            'validation': {...},
            'interactions': [...],
            'analyzed_interactions': [...],
            'risk_assessment': {...},
            'alternatives': {...},
            'clinical_report': '...',
            'patient_summary': '...',
            'structured_output': {...}
        },
        'execution_summary': {
            'total_duration_seconds': float,
            'drugs_analyzed': int,
            'interactions_found': int,
            'risk_level': str
        }
    },
    'errors': [...] or None
}
```

## 📊 Data Format

The system expects a DDI CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `drugbank_id_1` | DrugBank ID of first drug |
| `drug_name_1` | Name of first drug |
| `atc_1` | ATC code of first drug |
| `is_cardiovascular_1` | Cardiovascular flag (0/1) - includes antithrombotic drugs |
| `is_antithrombotic_1` | Legacy antithrombotic flag (now merged into cardiovascular) |
| `drugbank_id_2` | DrugBank ID of second drug |
| `drug_name_2` | Name of second drug |
| `atc_2` | ATC code of second drug |
| `is_cardiovascular_2` | Cardiovascular flag (0/1) - includes antithrombotic drugs |
| `is_antithrombotic_2` | Legacy antithrombotic flag (now merged into cardiovascular) |
| `interaction_description` | Description of the interaction |
| `severity_label` | Severity classification |
| `severity_confidence` | Confidence score |
| `severity_numeric` | Numeric severity (1-4) |

**Note**: All antithrombotic drugs are now classified as cardiovascular drugs in the analysis.

## ⚠️ Disclaimer

This AI-generated analysis is for informational purposes only and should not replace professional clinical judgment. Always consult with qualified healthcare providers before making changes to medication regimens.

## 📄 License

This project is for educational and research purposes. Please refer to the original paper for citation requirements.

## 📚 Reference

Based on: "AI-based Polypharmacy Risk-aware Drug Recommender System"
