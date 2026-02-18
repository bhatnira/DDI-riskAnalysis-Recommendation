"""
Agents Package - Modular Agentic Architecture for Polypharmacy Risk Analysis

This package provides an agentic, modular architecture for drug-drug interaction
analysis and polypharmacy risk assessment.

Architecture Overview:
======================

┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                           │
│              (Central Pipeline Controller)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐    ┌─────────────────┐                   │
│   │ 🔍 Interaction  │ ──▶│ ⚠️  Severity    │                   │
│   │    Agent        │    │    Agent        │                   │
│   └─────────────────┘    └────────┬────────┘                   │
│                                   │                            │
│                                   ▼                            │
│   ┌─────────────────┐    ┌─────────────────┐                   │
│   │ 📝 Explanation  │ ◀──│ 💊 Alternative  │                   │
│   │    Agent        │    │    Agent        │                   │
│   └─────────────────┘    └─────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Agents:
-------
- BaseAgent: Abstract base class for all agents
- InteractionAgent: Detects drug-drug interactions from a medication list
- SeverityAgent: Classifies and scores interaction severity using ML
- AlternativeAgent: Finds safer therapeutic alternatives via ATC classification
- ExplanationAgent: Generates human-readable clinical reports
- OrchestratorAgent: Coordinates the full analysis pipeline

Usage:
------
>>> from agents import OrchestratorAgent
>>> import pandas as pd
>>> 
>>> # Load DDI data
>>> df = pd.read_csv('ddi_data.csv')
>>> 
>>> # Initialize orchestrator
>>> orchestrator = OrchestratorAgent()
>>> orchestrator.initialize(df)
>>> 
>>> # Analyze medications
>>> result = orchestrator.analyze_drugs(['Warfarin', 'Aspirin', 'Metoprolol'])
>>> print(result['reports'])
"""

from .base_agent import BaseAgent, AgentResult, AgentStatus, AgentMessage
from .interaction_agent import InteractionAgent
from .severity_agent import SeverityAgent
from .alternative_agent import AlternativeAgent
from .explanation_agent import ExplanationAgent
from .orchestrator import OrchestratorAgent
from .llm_client import BioMistralClient, OllamaClient, get_llm_client
from .drug_risk_network import DrugRiskNetwork, DrugNode, DDIEdge
from .recommender import MultiObjectiveRecommender, AlternativeCandidate

# FAERS External Validation
from .faers_integration import FAERSClient, FAERSValidator, FAERSDrugProfile

# GNN Risk Assessment (optional - requires torch_geometric)
try:
    from .gnn_risk_assessment import (
        GNNSeverityPredictor, 
        GNNEmbeddingPredictor, 
        DrugEmbedder,
        run_gnn_comparison
    )
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# Comprehensive Comparison
from .comprehensive_comparison import ComprehensiveComparison, AlgorithmicRiskAssessor

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentResult', 
    'AgentStatus',
    'AgentMessage',
    # Specialized agents
    'InteractionAgent',
    'SeverityAgent',
    'AlternativeAgent',
    'ExplanationAgent',
    # Main orchestrator
    'OrchestratorAgent',
    # LLM clients
    'BioMistralClient',
    'OllamaClient',
    'get_llm_client',
    # Drug Risk Network (Paper Implementation)
    'DrugRiskNetwork',
    'DrugNode',
    'DDIEdge',
    # Multi-Objective Recommender (Paper Implementation)
    'MultiObjectiveRecommender',
    'AlternativeCandidate',
    # FAERS External Validation
    'FAERSClient',
    'FAERSValidator',
    'FAERSDrugProfile',
    # GNN Risk Assessment
    'GNN_AVAILABLE',
    'GNNSeverityPredictor',
    'GNNEmbeddingPredictor',
    'DrugEmbedder',
    'run_gnn_comparison',
    # Comprehensive Comparison
    'ComprehensiveComparison',
    'AlgorithmicRiskAssessor'
]

__version__ = '1.0.0'
__author__ = 'AI Polypharmacy Risk-aware Recommender System'
