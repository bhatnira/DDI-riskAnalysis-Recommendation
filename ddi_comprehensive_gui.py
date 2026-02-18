#!/usr/bin/env python3
"""
DDI Comprehensive Analyzer GUI
==============================
Features:
- Quick Interaction Check
- Comprehensive Report with Citations (Agentic LLM)
- Natural Language Chat

Run with: python ddi_comprehensive_gui.py
Then open: http://localhost:8080
"""

import http.server
import socketserver
import json
import threading
import os
import re
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the DDI analysis components
try:
    from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

import pandas as pd
from pathlib import Path

# Extended data storage for rich content
drug_full_data = {}  # drugbank_id -> full drug info
protein_full_data = {}  # protein_id -> full protein info

# Global state
assessor = None
kg_loader = None
loading_status = {"loading": True, "message": "Initializing..."}
chat_history = []

# LLM Configuration
LLM_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "default_model": "meditron:7b-q4_K_M",
    "temperature": 0.7,
    "max_tokens": 2048
}

AVAILABLE_MODELS = {
    "meditron:7b-q4_K_M": {"name": "Meditron 7B (Medical)", "description": "Medical/clinical focus", "params": "7B", "size": "4.1 GB"},
    "medllama2:7b-q4_K_M": {"name": "MedLlama2 7B (Medical)", "description": "Medical-tuned LLaMA", "params": "7B", "size": "4.1 GB"},
    "llama3:latest": {"name": "Llama 3 8B (General)", "description": "Strong general reasoning", "params": "8B", "size": "4.7 GB"},
    "mistral:7b-instruct-q4_K_M": {"name": "Mistral 7B Instruct", "description": "Fast Q&A", "params": "7B", "size": "4.4 GB"}
}

SYSTEM_PROMPT = """You are DDI Assistant, an expert clinical pharmacist AI specializing in drug-drug interactions.
You have access to a comprehensive knowledge graph from DrugBank containing drug interactions, side effects, protein targets, and pathways.
Always cite sources when providing information. Be accurate, evidence-based, and clinically relevant."""

REPORT_SYNTHESIS_PROMPT = """You are a clinical pharmacist writing a comprehensive drug interaction report for a healthcare team.

You have been provided with ACTUAL DATA extracted from authoritative pharmaceutical databases:
- DrugBank: Drug information, indications, mechanisms of action, DDI descriptions
- SIDER: Side effect data with UMLS identifiers
- UniProt: Protein target information with functions
- KEGG/SMPDB: Metabolic pathway data

Using this REAL database content, write a professional clinical report. 

IMPORTANT GUIDELINES:
1. Use the ACTUAL drug indications and mechanisms provided - do not make up information
2. Quote the EXACT DDI descriptions from DrugBank when discussing interactions
3. Cite specific database identifiers (e.g., DrugBank ID DB00001, UMLS CUI C0000729)
4. Include hyperlinks to source databases where provided
5. Highlight overlapping side effects that may be amplified
6. Discuss shared protein targets and their clinical implications
7. Provide actionable recommendations based on the evidence

Format your report with these sections:

## Executive Summary
Brief risk assessment with key concerns.

## Drug Profiles
For each drug, summarize from the provided indication and mechanism data.

## Drug-Drug Interactions
List each interaction with the FULL description from DrugBank, severity, and clinical significance.

## Side Effect Risk Analysis 
Identify overlapping side effects from SIDER data that may be amplified.

## Mechanistic Analysis
Discuss shared protein targets from UniProt data and pathway conflicts.

## Clinical Recommendations
Evidence-based suggestions for the care team.

## Database References
List all cited database identifiers with URLs.

---
KNOWLEDGE GRAPH DATA:
{kg_data}
---

Generate the comprehensive clinical report using the ACTUAL content above:"""


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDI Comprehensive Analyzer</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,-apple-system,sans-serif;background:#f0f2f5;min-height:100vh}
        .header{background:linear-gradient(135deg,#1a365d,#2563eb);color:#fff;padding:20px;text-align:center}
        .header h1{font-size:24px;margin-bottom:6px}
        .header p{font-size:13px;opacity:0.9}
        
        .main-container{max-width:1600px;margin:0 auto;padding:20px;display:grid;grid-template-columns:350px 1fr;gap:20px}
        
        .panel{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);overflow:hidden}
        .panel-header{background:linear-gradient(135deg,#f8fafc,#e2e8f0);padding:14px 18px;border-bottom:1px solid #e2e8f0;font-weight:600;font-size:15px;display:flex;justify-content:space-between;align-items:center}
        .panel-content{padding:18px}
        
        /* Left Sidebar */
        .sidebar{display:flex;flex-direction:column;gap:20px}
        
        /* Input Section */
        .drug-input{margin-bottom:15px}
        .drug-input label{display:block;font-size:13px;font-weight:600;margin-bottom:8px;color:#374151}
        .drug-input input{width:100%;padding:12px 14px;border:2px solid #e5e7eb;border-radius:10px;font-size:14px;transition:border-color 0.2s}
        .drug-input input:focus{outline:none;border-color:#2563eb}
        .hint{color:#6b7280;font-size:11px;margin-top:6px}
        
        /* Buttons */
        .btn{padding:12px 20px;border:none;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.2s;display:inline-flex;align-items:center;gap:8px}
        .btn:hover{transform:translateY(-1px)}
        .btn:active{transform:translateY(0)}
        .btn-primary{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;box-shadow:0 4px 14px rgba(37,99,235,0.3)}
        .btn-primary:hover{box-shadow:0 6px 20px rgba(37,99,235,0.4)}
        .btn-report{background:linear-gradient(135deg,#059669,#047857);color:#fff;box-shadow:0 4px 14px rgba(5,150,105,0.3);width:100%;margin-top:10px}
        .btn-report:hover{box-shadow:0 6px 20px rgba(5,150,105,0.4)}
        .btn-secondary{background:#f3f4f6;color:#374151}
        .btn-secondary:hover{background:#e5e7eb}
        .btn-ex{background:#eff6ff;color:#2563eb;padding:8px 12px;font-size:11px;margin:4px;border-radius:6px}
        .btn-ex:hover{background:#2563eb;color:#fff}
        
        .btn-row{display:flex;gap:10px;margin-top:12px}
        
        /* Examples */
        .examples{margin-top:18px;padding-top:18px;border-top:1px solid #e5e7eb}
        .examples-label{font-size:11px;color:#6b7280;margin-bottom:8px;font-weight:600}
        
        /* Quick Results */
        .quick-results{margin-top:18px}
        .risk-card{padding:16px;border-radius:10px;margin-bottom:12px}
        .risk-critical{background:linear-gradient(135deg,#fef2f2,#fee2e2);border-left:4px solid #dc2626}
        .risk-high{background:linear-gradient(135deg,#fff7ed,#ffedd5);border-left:4px solid #ea580c}
        .risk-moderate{background:linear-gradient(135deg,#fffbeb,#fef3c7);border-left:4px solid #d97706}
        .risk-low{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-left:4px solid #16a34a}
        .risk-score{font-size:28px;font-weight:700;margin-bottom:4px}
        .risk-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
        
        .int-list{max-height:300px;overflow-y:auto}
        .int-item{padding:12px;margin:8px 0;border-radius:8px;border-left:4px solid;font-size:13px}
        .int-item.contraindicated{background:#fef2f2;border-color:#dc2626}
        .int-item.major{background:#fff7ed;border-color:#ea580c}
        .int-item.moderate{background:#fffbeb;border-color:#d97706}
        .int-item.minor{background:#f0fdf4;border-color:#16a34a}
        .int-drugs{font-weight:600;color:#1f2937}
        .int-sev{font-size:10px;font-weight:700;text-transform:uppercase;color:#6b7280;margin-top:4px}
        
        /* Main Content Area */
        .content-area{display:flex;flex-direction:column;gap:20px}
        
        /* Tabs */
        .tabs{display:flex;gap:4px;padding:4px;background:#f3f4f6;border-radius:10px;width:fit-content}
        .tab{padding:10px 20px;border:none;background:transparent;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;color:#6b7280;transition:all 0.2s}
        .tab.active{background:#fff;color:#1f2937;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
        .tab:hover:not(.active){color:#374151}
        
        /* Report View */
        .report-container{display:none;flex:1}
        .report-container.active{display:flex;flex-direction:column}
        .report-content{flex:1;padding:24px;overflow-y:auto;max-height:calc(100vh - 280px);line-height:1.7;font-size:14px}
        .report-content h2{color:#1e40af;margin:24px 0 12px;padding-bottom:8px;border-bottom:2px solid #dbeafe;font-size:18px}
        .report-content h3{color:#1f2937;margin:18px 0 10px;font-size:15px}
        .report-content ul{margin-left:20px;margin-bottom:12px}
        .report-content li{margin-bottom:6px}
        .report-content .citation{background:#eff6ff;padding:2px 8px;border-radius:4px;font-size:12px;font-family:monospace;color:#1d4ed8}
        .report-content .severity-tag{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;text-transform:uppercase}
        .report-content .sev-contraindicated{background:#fecaca;color:#991b1b}
        .report-content .sev-major{background:#fed7aa;color:#9a3412}
        .report-content .sev-moderate{background:#fde68a;color:#92400e}
        .report-content .sev-minor{background:#bbf7d0;color:#166534}
        .report-placeholder{text-align:center;padding:60px 20px;color:#9ca3af}
        .report-placeholder h3{color:#6b7280;margin-bottom:12px}
        .report-loading{display:flex;flex-direction:column;align-items:center;gap:16px;padding:60px}
        .report-loading .spinner{width:48px;height:48px;border:4px solid #e5e7eb;border-top-color:#2563eb;border-radius:50%;animation:spin 1s linear infinite}
        @keyframes spin{to{transform:rotate(360deg)}}
        .agent-steps{text-align:left;max-width:400px;margin-top:16px}
        .agent-step{padding:8px 12px;margin:4px 0;border-radius:6px;font-size:12px;display:flex;align-items:center;gap:8px}
        .agent-step.done{background:#dcfce7;color:#166534}
        .agent-step.active{background:#dbeafe;color:#1d4ed8;animation:pulse 1.5s infinite}
        .agent-step.pending{background:#f3f4f6;color:#9ca3af}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}}
        
        /* Chat View */
        .chat-container{display:none;flex:1;flex-direction:column}
        .chat-container.active{display:flex}
        .chat-messages{flex:1;padding:20px;overflow-y:auto;max-height:calc(100vh - 340px);min-height:400px}
        .message{margin-bottom:16px;display:flex}
        .message.user{justify-content:flex-end}
        .message.assistant{justify-content:flex-start}
        .message-content{max-width:75%;padding:14px 18px;border-radius:18px;font-size:14px;line-height:1.6}
        .user .message-content{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-bottom-right-radius:4px}
        .assistant .message-content{background:#f3f4f6;color:#1f2937;border-bottom-left-radius:4px}
        .chat-input{padding:16px;border-top:1px solid #e5e7eb;display:flex;gap:12px}
        .chat-input input{flex:1;padding:14px 18px;border:2px solid #e5e7eb;border-radius:24px;font-size:14px}
        .chat-input input:focus{outline:none;border-color:#2563eb}
        .welcome{text-align:center;padding:50px 20px;color:#6b7280}
        .welcome h3{color:#374151;margin-bottom:12px;font-size:18px}
        .suggestions{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:20px}
        .suggestion{background:#eff6ff;color:#2563eb;padding:10px 16px;border-radius:20px;font-size:12px;cursor:pointer;transition:0.2s}
        .suggestion:hover{background:#2563eb;color:#fff}
        
        /* Status Bar */
        .status-bar{text-align:center;padding:10px;font-size:11px;color:#6b7280;background:#f8fafc;border-top:1px solid #e5e7eb}
        .status-ready{color:#059669}
        .status-loading{color:#d97706}
        
        /* Modal */
        .modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:1000;justify-content:center;align-items:center}
        .modal.show{display:flex}
        .modal-content{background:#fff;border-radius:16px;padding:24px;max-width:500px;width:90%;max-height:80vh;overflow-y:auto}
        .modal-header{font-size:18px;font-weight:600;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center}
        .modal-close{background:none;border:none;font-size:24px;cursor:pointer;color:#9ca3af}
        .model-option{padding:14px;margin:8px 0;border-radius:10px;cursor:pointer;border:2px solid #e5e7eb;transition:all 0.2s}
        .model-option:hover{border-color:#2563eb;background:#eff6ff}
        .model-option.active{border-color:#2563eb;background:#eff6ff}
        .model-name{font-weight:600;margin-bottom:4px}
        .model-desc{font-size:12px;color:#6b7280}
        .model-meta{font-size:11px;color:#9ca3af;margin-top:4px}
        
        /* Print styles */
        @media print{.sidebar,.tabs,.chat-input,.btn,.header{display:none!important}.report-content{max-height:none!important}}
        @media(max-width:1024px){.main-container{grid-template-columns:1fr}}
    </style>
</head>
<body>

<div class="header">
    <h1>💊 DDI Comprehensive Analyzer</h1>
    <p>Knowledge Graph-Powered Drug Interaction Analysis with AI Synthesis</p>
</div>

<div class="main-container">
    <!-- Left Sidebar -->
    <div class="sidebar">
        <div class="panel">
            <div class="panel-header">
                <span>🔍 Drug Regimen Analysis</span>
            </div>
            <div class="panel-content">
                <div class="drug-input">
                    <label>Enter Drug Names</label>
                    <input type="text" id="drug-input" placeholder="warfarin, aspirin, metoprolol, lisinopril" onkeypress="if(event.key==='Enter')quickCheck()">
                    <p class="hint">Separate multiple drugs with commas</p>
                </div>
                
                <div class="btn-row">
                    <button class="btn btn-primary" onclick="quickCheck()">⚡ Quick Check</button>
                    <button class="btn btn-secondary" onclick="clearInputs()">Clear</button>
                </div>
                
                <button class="btn btn-report" onclick="generateReport()">📋 Generate Comprehensive Report</button>
                
                <div class="examples">
                    <div class="examples-label">Try these examples:</div>
                    <button class="btn btn-ex" onclick="setDrugs('warfarin, aspirin, ibuprofen')">🔴 High Risk</button>
                    <button class="btn btn-ex" onclick="setDrugs('metformin, lisinopril, amlodipine, metoprolol')">🟠 CV Combo</button>
                    <button class="btn btn-ex" onclick="setDrugs('sertraline, tramadol, ondansetron')">🟡 Serotonin Risk</button>
                    <button class="btn btn-ex" onclick="setDrugs('acetaminophen, omeprazole')">🟢 Low Risk</button>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">⚡ Quick Results</div>
            <div class="panel-content">
                <div id="quick-results">
                    <div class="report-placeholder">
                        <p>Enter drugs above and click Quick Check</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Main Content Area -->
    <div class="content-area">
        <div class="panel" style="flex:1;display:flex;flex-direction:column">
            <div class="panel-header">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('report')">📋 Comprehensive Report</button>
                    <button class="tab" onclick="switchTab('chat')">💬 Chat Assistant</button>
                </div>
                <button class="btn btn-secondary" style="padding:6px 12px;font-size:11px" onclick="showModelModal()" id="model-btn">🤖 Meditron 7B</button>
            </div>
            
            <!-- Report Tab -->
            <div class="report-container active" id="report-tab">
                <div class="report-content" id="report-content">
                    <div class="report-placeholder">
                        <h3>📋 Comprehensive Report</h3>
                        <p>Enter a drug regimen and click "Generate Comprehensive Report"</p>
                        <p style="margin-top:12px;font-size:12px">The report will include:</p>
                        <ul style="text-align:left;display:inline-block;margin-top:8px;font-size:13px">
                            <li>Drug-drug interactions with severity and mechanisms</li>
                            <li>Side effect overlap analysis</li>
                            <li>Protein target conflicts</li>
                            <li>Pathway analysis</li>
                            <li>Citations to DrugBank, SIDER, and other sources</li>
                            <li>AI-synthesized clinical recommendations</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <!-- Chat Tab -->
            <div class="chat-container" id="chat-tab">
                <div class="chat-messages" id="chat-messages">
                    <div class="welcome">
                        <h3>💬 DDI Chat Assistant</h3>
                        <p>Ask questions about drug interactions, mechanisms, and safety</p>
                        <div class="suggestions">
                            <div class="suggestion" onclick="askQuestion('What are the risks of combining warfarin with NSAIDs?')">Warfarin + NSAIDs</div>
                            <div class="suggestion" onclick="askQuestion('Explain serotonin syndrome risk factors')">Serotonin Syndrome</div>
                            <div class="suggestion" onclick="askQuestion('What are safer alternatives to aspirin for antiplatelet therapy?')">Aspirin Alternatives</div>
                        </div>
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Ask about drug interactions..." onkeypress="if(event.key==='Enter')sendChat()">
                    <button class="btn btn-primary" onclick="sendChat()">Send</button>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="status-bar" id="status">Connecting...</div>

<!-- Model Selection Modal -->
<div class="modal" id="model-modal">
    <div class="modal-content">
        <div class="modal-header">
            <span>Select LLM Model</span>
            <button class="modal-close" onclick="closeModelModal()">&times;</button>
        </div>
        <div id="model-list"></div>
    </div>
</div>

<script>
let currentModel = '';
let availableModels = {};
let chatHistory = [];

// Initialize immediately
(function() {
    console.log('Script loaded, checking status in 100ms...');
    setTimeout(checkStatus, 100);
})();

// Also on load
window.onload = function() {
    console.log('Page loaded, checking status...');
    checkStatus();
};

function checkStatus() {
    const s = document.getElementById('status');
    if (!s) { setTimeout(checkStatus, 100); return; }
    
    console.log('Fetching /status...');
    fetch('/status', {cache: 'no-store'})
        .then(r => {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(d => {
            console.log('Status data:', d);
            if(d.loading) {
                s.className = 'status-bar status-loading';
                s.textContent = '⏳ ' + d.message;
                setTimeout(checkStatus, 1000);
            } else {
                s.className = 'status-bar status-ready';
                currentModel = d.current_model || '';
                availableModels = d.available_models || {};
                const modelInfo = availableModels[currentModel] || {};
                s.textContent = '✅ ' + d.message + ' | 🤖 ' + (modelInfo.name || 'Template');
                const modelBtn = document.getElementById('model-btn');
                if (modelBtn) modelBtn.textContent = '🤖 ' + (modelInfo.name?.split(' ')[0] || 'Model');
            }
        })
        .catch(e => {
            console.error('Status error:', e);
            s.textContent = '⚠️ Retrying... (' + e.message + ')';
            setTimeout(checkStatus, 2000);
        });
}

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.report-container, .chat-container').forEach(c => c.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(tab + '-tab').classList.add('active');
}

function setDrugs(drugs) {
    document.getElementById('drug-input').value = drugs;
    quickCheck();
}

function clearInputs() {
    document.getElementById('drug-input').value = '';
    document.getElementById('quick-results').innerHTML = '<div class="report-placeholder"><p>Enter drugs above</p></div>';
}

function quickCheck() {
    const drugs = document.getElementById('drug-input').value.trim();
    if(!drugs) { alert('Please enter drug names'); return; }
    
    const results = document.getElementById('quick-results');
    results.innerHTML = '<div style="text-align:center;padding:20px"><div class="spinner" style="width:32px;height:32px;border:3px solid #e5e7eb;border-top-color:#2563eb;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto"></div><p style="margin-top:12px;color:#6b7280;font-size:12px">Analyzing...</p></div>';
    
    fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({drugs: drugs})
    })
    .then(r => r.json())
    .then(d => {
        if(d.error) {
            results.innerHTML = '<div style="color:#dc2626;padding:12px">⚠️ ' + d.error + '</div>';
            return;
        }
        displayQuickResults(d);
    })
    .catch(e => results.innerHTML = '<div style="color:#dc2626;padding:12px">⚠️ Error: ' + e + '</div>');
}

function displayQuickResults(d) {
    const results = document.getElementById('quick-results');
    const rc = d.risk_level.toLowerCase();
    const riskClass = rc === 'critical' ? 'risk-critical' : rc === 'high' ? 'risk-high' : rc === 'moderate' ? 'risk-moderate' : 'risk-low';
    
    let html = '<div class="risk-card ' + riskClass + '">';
    html += '<div class="risk-score">' + d.risk_score.toFixed(2) + '</div>';
    html += '<div class="risk-label">' + d.risk_level + ' Risk</div>';
    html += '<div style="margin-top:8px;font-size:12px;color:#6b7280">Drugs: ' + d.drugs.join(', ') + '</div>';
    html += '</div>';
    
    if(d.interactions.length === 0) {
        html += '<div style="text-align:center;padding:16px;color:#059669">✅ No significant interactions detected</div>';
    } else {
        html += '<div style="font-size:12px;font-weight:600;margin-bottom:8px">Found ' + d.interactions.length + ' interaction(s):</div>';
        html += '<div class="int-list">';
        for(const i of d.interactions) {
            const sc = i.severity.toLowerCase().replace(/\\s+/g, '-').replace('-interaction', '');
            html += '<div class="int-item ' + sc + '">';
            html += '<div class="int-drugs">' + i.drug1 + ' ↔ ' + i.drug2 + '</div>';
            html += '<div class="int-sev">' + i.severity + '</div>';
            html += '</div>';
        }
        html += '</div>';
    }
    
    results.innerHTML = html;
}

function generateReport() {
    const drugs = document.getElementById('drug-input').value.trim();
    if(!drugs) { alert('Please enter drug names first'); return; }
    
    // Switch to report tab
    switchTab('report');
    document.querySelector('.tab').click();
    
    const content = document.getElementById('report-content');
    content.innerHTML = `
        <div class="report-loading">
            <div class="spinner"></div>
            <div style="font-weight:600;color:#374151">Generating Comprehensive Report...</div>
            <div class="agent-steps">
                <div class="agent-step active" id="step-1">📊 Querying Knowledge Graph...</div>
                <div class="agent-step pending" id="step-2">🔍 Analyzing Drug Interactions...</div>
                <div class="agent-step pending" id="step-3">💊 Checking Side Effect Overlap...</div>
                <div class="agent-step pending" id="step-4">🧬 Examining Protein Targets...</div>
                <div class="agent-step pending" id="step-5">🤖 Synthesizing with LLM...</div>
                <div class="agent-step pending" id="step-6">📋 Formatting Report...</div>
            </div>
        </div>
    `;
    
    // Simulate agent steps
    setTimeout(() => updateStep(1, 2), 500);
    setTimeout(() => updateStep(2, 3), 1200);
    setTimeout(() => updateStep(3, 4), 2000);
    setTimeout(() => updateStep(4, 5), 2800);
    
    fetch('/comprehensive_report', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({drugs: drugs})
    })
    .then(r => r.json())
    .then(d => {
        if(d.error) {
            content.innerHTML = '<div style="color:#dc2626;padding:24px">⚠️ Error: ' + d.error + '</div>';
            return;
        }
        updateStep(5, 6);
        setTimeout(() => {
            displayReport(d);
        }, 500);
    })
    .catch(e => content.innerHTML = '<div style="color:#dc2626;padding:24px">⚠️ Error: ' + e + '</div>');
}

function updateStep(done, active) {
    document.getElementById('step-' + done)?.classList.replace('active', 'done');
    document.getElementById('step-' + done)?.innerHTML = '✅ ' + document.getElementById('step-' + done)?.textContent.slice(2);
    document.getElementById('step-' + active)?.classList.replace('pending', 'active');
}

function displayReport(d) {
    const content = document.getElementById('report-content');
    
    let html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">';
    html += '<div><h2 style="margin:0;border:none;padding:0">Comprehensive Drug Interaction Report</h2>';
    html += '<div style="color:#6b7280;font-size:12px;margin-top:4px">Generated: ' + new Date().toLocaleString() + '</div></div>';
    html += '<button class="btn btn-secondary" onclick="window.print()" style="padding:8px 16px">🖨️ Print Report</button>';
    html += '</div>';
    
    // Risk Summary Card
    const rc = d.risk_level?.toLowerCase() || 'moderate';
    const riskClass = rc === 'critical' ? 'risk-critical' : rc === 'high' ? 'risk-high' : rc === 'moderate' ? 'risk-moderate' : 'risk-low';
    html += '<div class="risk-card ' + riskClass + '" style="margin-bottom:24px">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center">';
    html += '<div><div class="risk-label">Overall Risk Assessment</div><div class="risk-score">' + d.risk_level + '</div></div>';
    html += '<div style="text-align:right"><div style="font-size:12px;color:#6b7280">Risk Score</div><div style="font-size:32px;font-weight:700">' + (d.risk_score?.toFixed(2) || 'N/A') + '</div></div>';
    html += '</div>';
    html += '<div style="margin-top:12px;font-size:13px"><strong>Regimen:</strong> ' + (d.drugs?.join(', ') || 'N/A') + '</div>';
    
    // Metadata
    if(d.metadata) {
        html += '<div style="margin-top:8px;font-size:11px;color:#6b7280">';
        html += 'Interactions: ' + (d.metadata.interaction_count || 0) + ' | ';
        html += 'Shared Side Effects: ' + (d.metadata.shared_side_effects || 0) + ' | ';
        html += 'Shared Proteins: ' + (d.metadata.shared_proteins || 0);
        html += '</div>';
    }
    html += '</div>';
    
    // LLM Synthesized Report (if available)
    if(d.llm_report && !d.llm_report.startsWith('[')) {
        html += '<div style="background:#eff6ff;padding:16px;border-radius:8px;margin-bottom:24px;border-left:4px solid #2563eb">';
        html += '<h3 style="margin:0 0 12px 0;color:#1e40af">🤖 AI-Synthesized Clinical Summary</h3>';
        html += '<div style="font-size:13px;color:#374151">' + formatLLMReport(d.llm_report) + '</div>';
        html += '</div>';
    }
    
    // Always show raw KG data with full source material
    if(d.kg_data) {
        html += '<div style="border-top:2px solid #e5e7eb;margin-top:24px;padding-top:24px">';
        html += '<h2 style="margin-bottom:16px;color:#1f2937">📊 Source Database Evidence</h2>';
        html += formatRawKGData(d.kg_data);
        html += '</div>';
    }
    
    content.innerHTML = html;
}

function formatLLMReport(report) {
    // Convert markdown-style formatting to HTML
    let html = report
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/\\n\\n/g, '</p><p>')
        .replace(/\\n/g, '<br>')
        .replace(/\\[(DB\\d+)\\]/g, '<span class="citation">$1</span>')
        .replace(/\\[(UMLS:[A-Z0-9]+)\\]/g, '<span class="citation">$1</span>')
        .replace(/Contraindicated/gi, '<span class="severity-tag sev-contraindicated">Contraindicated</span>')
        .replace(/Major interaction/gi, '<span class="severity-tag sev-major">Major</span>')
        .replace(/Moderate interaction/gi, '<span class="severity-tag sev-moderate">Moderate</span>')
        .replace(/Minor interaction/gi, '<span class="severity-tag sev-minor">Minor</span>');
    
    return '<div class="llm-report">' + html + '</div>';
}

function formatRawKGData(kg) {
    let html = '';
    
    // Drug Profiles Section
    if(kg.drug_profiles?.length > 0) {
        html += '<h2>Drug Profiles</h2>';
        for(const drug of kg.drug_profiles) {
            html += '<div style="background:#f8fafc;padding:16px;margin:12px 0;border-radius:8px;border-left:4px solid #2563eb">';
            html += '<div style="display:flex;justify-content:space-between;align-items:start">';
            html += '<h3 style="margin:0 0 8px 0;color:#1e40af">' + drug.name + '</h3>';
            html += '<span class="citation">' + drug.drugbank_id + '</span>';
            html += '</div>';
            if(drug.type) html += '<div style="font-size:12px;color:#6b7280;margin-bottom:8px"><strong>Type:</strong> ' + drug.type + ' | <strong>Groups:</strong> ' + (drug.groups || 'N/A') + '</div>';
            if(drug.indication) {
                html += '<div style="margin:8px 0"><strong>Indication:</strong></div>';
                html += '<div style="font-size:13px;color:#374151;background:#fff;padding:10px;border-radius:4px;max-height:150px;overflow-y:auto">' + drug.indication.substring(0, 800) + (drug.indication.length > 800 ? '...' : '') + '</div>';
            }
            if(drug.mechanism_of_action) {
                html += '<div style="margin:8px 0"><strong>Mechanism of Action:</strong></div>';
                html += '<div style="font-size:13px;color:#374151;background:#fff;padding:10px;border-radius:4px;max-height:150px;overflow-y:auto">' + drug.mechanism_of_action.substring(0, 800) + (drug.mechanism_of_action.length > 800 ? '...' : '') + '</div>';
            }
            if(drug.drugbank_url) html += '<div style="margin-top:8px"><a href="' + drug.drugbank_url + '" target="_blank" style="color:#2563eb;font-size:12px">View on DrugBank →</a></div>';
            html += '</div>';
        }
    }
    
    // Drug Interactions Section
    html += '<h2>Drug-Drug Interactions</h2>';
    if(kg.interactions?.length > 0) {
        for(const i of kg.interactions) {
            const sevClass = i.severity.toLowerCase().includes('contraindicated') ? 'sev-contraindicated' : 
                            i.severity.toLowerCase().includes('major') ? 'sev-major' : 
                            i.severity.toLowerCase().includes('moderate') ? 'sev-moderate' : 'sev-minor';
            html += '<div style="background:#fff;padding:16px;margin:12px 0;border-radius:8px;border:1px solid #e5e7eb">';
            html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">';
            html += '<strong style="font-size:15px">' + i.drug1 + ' ↔ ' + i.drug2 + '</strong>';
            html += '<span class="severity-tag ' + sevClass + '">' + i.severity + '</span>';
            html += '</div>';
            if(i.description) {
                html += '<div style="font-size:13px;color:#374151;background:#f8fafc;padding:12px;border-radius:4px;margin:8px 0;line-height:1.6">' + i.description + '</div>';
            }
            html += '<div style="font-size:11px;color:#6b7280;margin-top:8px">';
            html += '<span class="citation">' + i.drug1_id + '</span> + <span class="citation">' + i.drug2_id + '</span>';
            html += ' | Source: ' + i.source;
            if(i.interaction_url) html += ' | <a href="' + i.interaction_url + '" target="_blank" style="color:#2563eb">View details</a>';
            html += '</div></div>';
        }
    } else {
        html += '<p style="color:#059669;padding:16px;background:#f0fdf4;border-radius:8px">✅ No direct drug-drug interactions found in the database.</p>';
    }
    
    // Side Effects Section
    if(kg.side_effects?.length > 0) {
        html += '<h2>Overlapping Side Effects (SIDER Database)</h2>';
        html += '<p style="font-size:13px;color:#6b7280;margin-bottom:12px">Side effects shared by multiple drugs in the regimen may be amplified:</p>';
        html += '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:12px">';
        for(const se of kg.side_effects.slice(0, 15)) {
            html += '<div style="background:#fff7ed;padding:12px;border-radius:8px;border-left:3px solid #ea580c">';
            html += '<div style="font-weight:600;margin-bottom:4px">' + se.name + '</div>';
            html += '<div style="font-size:12px;color:#6b7280">Shared by: ' + se.drugs.join(', ') + '</div>';
            html += '<div style="font-size:11px;color:#9a3412;margin-top:4px">' + se.risk_note + '</div>';
            html += '<div style="font-size:10px;color:#6b7280;margin-top:4px"><span class="citation">' + se.umls_cui + '</span></div>';
            html += '</div>';
        }
        html += '</div>';
        if(kg.side_effects.length > 15) {
            html += '<p style="font-size:12px;color:#6b7280;margin-top:12px">+ ' + (kg.side_effects.length - 15) + ' more shared side effects</p>';
        }
    }
    
    // Protein Targets Section
    if(kg.proteins?.length > 0) {
        html += '<h2>Shared Protein Targets (DrugBank/UniProt)</h2>';
        html += '<p style="font-size:13px;color:#6b7280;margin-bottom:12px">Drugs targeting the same proteins may have additive effects or compete:</p>';
        for(const p of kg.proteins.slice(0, 10)) {
            html += '<div style="background:#eff6ff;padding:14px;margin:10px 0;border-radius:8px;border-left:3px solid #2563eb">';
            html += '<div style="display:flex;justify-content:space-between;align-items:start">';
            html += '<div><strong>' + p.name + '</strong>';
            if(p.gene_name) html += ' <span style="font-size:12px;color:#6b7280">(' + p.gene_name + ')</span>';
            html += '</div>';
            html += '<span class="citation">' + p.id + '</span>';
            html += '</div>';
            html += '<div style="font-size:12px;color:#374151;margin:6px 0">Targeted by: <strong>' + p.drugs.join(', ') + '</strong></div>';
            if(p.general_function) {
                html += '<div style="font-size:12px;margin:6px 0"><strong>Function:</strong> ' + p.general_function.substring(0, 300) + '</div>';
            }
            if(p.specific_function) {
                html += '<div style="font-size:12px;color:#4b5563;margin:6px 0">' + p.specific_function.substring(0, 400) + '</div>';
            }
            if(p.cellular_location) html += '<div style="font-size:11px;color:#6b7280"><strong>Location:</strong> ' + p.cellular_location + '</div>';
            if(p.uniprot_url) html += '<div style="margin-top:6px"><a href="' + p.uniprot_url + '" target="_blank" style="color:#2563eb;font-size:11px">View on UniProt →</a></div>';
            html += '</div>';
        }
    }
    
    // Pathways Section
    if(kg.pathways?.length > 0) {
        html += '<h2>Shared Metabolic Pathways</h2>';
        html += '<ul>';
        for(const pw of kg.pathways.slice(0, 10)) {
            html += '<li><strong>' + pw.id + '</strong> - Drugs: ' + pw.drugs.join(', ');
            if(pw.kegg_url) html += ' <a href="' + pw.kegg_url + '" target="_blank" style="color:#2563eb;font-size:11px">[KEGG]</a>';
            html += '</li>';
        }
        html += '</ul>';
    }
    
    // Sources Section
    if(kg.metadata?.sources) {
        html += '<h2>Database Sources</h2>';
        html += '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));gap:10px">';
        for(const src of kg.metadata.sources) {
            html += '<div style="background:#f8fafc;padding:10px;border-radius:6px">';
            html += '<div style="font-weight:600;font-size:13px">' + src.name + '</div>';
            html += '<div style="font-size:11px;color:#6b7280">' + src.description + '</div>';
            html += '<a href="' + src.url + '" target="_blank" style="font-size:11px;color:#2563eb">' + src.url + '</a>';
            html += '</div>';
        }
        html += '</div>';
    }
    
    return html;
}

// Chat Functions
function askQuestion(q) {
    document.getElementById('chat-input').value = q;
    sendChat();
}

function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if(!msg) return;
    
    input.value = '';
    addMessage('user', msg);
    chatHistory.push({role: 'user', content: msg});
    
    // Typing indicator
    const typing = document.createElement('div');
    typing.className = 'message assistant';
    typing.id = 'typing';
    typing.innerHTML = '<div class="message-content"><div style="display:flex;gap:4px"><span style="width:8px;height:8px;background:#6b7280;border-radius:50%;animation:bounce 1.4s infinite"></span><span style="width:8px;height:8px;background:#6b7280;border-radius:50%;animation:bounce 1.4s infinite 0.16s"></span><span style="width:8px;height:8px;background:#6b7280;border-radius:50%;animation:bounce 1.4s infinite 0.32s"></span></div></div>';
    document.getElementById('chat-messages').appendChild(typing);
    scrollChat();
    
    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, history: chatHistory.slice(-10)})
    })
    .then(r => r.json())
    .then(d => {
        document.getElementById('typing')?.remove();
        if(d.error) {
            addMessage('assistant', '⚠️ ' + d.error);
        } else {
            addMessage('assistant', d.response);
            chatHistory.push({role: 'assistant', content: d.response});
        }
    })
    .catch(e => {
        document.getElementById('typing')?.remove();
        addMessage('assistant', '⚠️ Error: ' + e.message);
    });
}

function addMessage(role, content) {
    const messages = document.getElementById('chat-messages');
    const welcome = messages.querySelector('.welcome');
    if(welcome) welcome.remove();
    
    const div = document.createElement('div');
    div.className = 'message ' + role;
    
    let formatted = content
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/\\n/g, '<br>');
    
    div.innerHTML = '<div class="message-content">' + formatted + '</div>';
    messages.appendChild(div);
    scrollChat();
}

function scrollChat() {
    const messages = document.getElementById('chat-messages');
    messages.scrollTop = messages.scrollHeight;
}

// Model Modal
function showModelModal() {
    const modal = document.getElementById('model-modal');
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    
    for(const [key, info] of Object.entries(availableModels)) {
        const isActive = key === currentModel;
        const div = document.createElement('div');
        div.className = 'model-option' + (isActive ? ' active' : '');
        div.innerHTML = '<div class="model-name">' + (isActive ? '✓ ' : '') + info.name + '</div>' +
                        '<div class="model-desc">' + info.description + '</div>' +
                        '<div class="model-meta">Parameters: ' + info.params + ' | Size: ' + info.size + '</div>';
        div.onclick = () => selectModel(key);
        list.appendChild(div);
    }
    
    modal.classList.add('show');
}

function closeModelModal() {
    document.getElementById('model-modal').classList.remove('show');
}

function selectModel(key) {
    fetch('/set_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: key})
    })
    .then(r => r.json())
    .then(d => {
        if(d.success) {
            currentModel = key;
            closeModelModal();
            checkStatus();
        } else {
            alert('Error: ' + d.error);
        }
    });
}
</script>
</body>
</html>"""


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Multi-model LLM client for report synthesis"""
    
    def __init__(self):
        self.backend = None
        self.model_name = "template"
        self.current_model = None
        self.installed_models = {}
        self._detect_backend()
    
    def _detect_backend(self):
        try:
            response = requests.get(f"{LLM_CONFIG['ollama_url']}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.installed_models = {m['name']: m for m in models}
                
                if models:
                    # Prefer medical models
                    model = None
                    for preferred in ['meditron', 'medllama2', 'llama3', 'mistral']:
                        for m_name in self.installed_models.keys():
                            if preferred in m_name.lower():
                                model = m_name
                                break
                        if model:
                            break
                    
                    if not model:
                        model = models[0]['name']
                    
                    self.backend = 'ollama'
                    self.current_model = model
                    self.model_name = AVAILABLE_MODELS.get(model, {}).get('name', model)
                    return
        except:
            pass
        
        self.backend = 'template'
        self.model_name = 'template'
    
    def set_model(self, model_key: str) -> bool:
        if model_key in self.installed_models or model_key in AVAILABLE_MODELS:
            self.current_model = model_key
            self.model_name = AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)
            return True
        return False
    
    def get_available_models(self) -> Dict:
        return {k: v for k, v in AVAILABLE_MODELS.items() if k in self.installed_models}
    
    def generate(self, messages: List[Dict], context: str = "") -> str:
        if self.backend == 'ollama':
            return self._generate_ollama(messages)
        return self._generate_template(messages, context)
    
    def _generate_ollama(self, messages: List[Dict]) -> str:
        try:
            response = requests.post(
                f"{LLM_CONFIG['ollama_url']}/api/chat",
                json={
                    "model": self.current_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": LLM_CONFIG['temperature'], "num_predict": LLM_CONFIG['max_tokens']}
                },
                timeout=180
            )
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '[No response]')
        except Exception as e:
            return f"[LLM Error: {str(e)[:100]}]"
        return "[Request failed]"
    
    def _generate_template(self, messages: List[Dict], context: str = "") -> str:
        return """**Template Response** (LLM not available)

Based on the knowledge graph data provided, this drug regimen requires careful review.

Please install Ollama and a medical model (meditron, medllama2) for AI-synthesized reports:
```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull meditron:7b-q4_K_M
```"""
    
    def synthesize_report(self, kg_data: Dict) -> str:
        """Use LLM to synthesize a comprehensive report from KG data"""
        kg_data_str = json.dumps(kg_data, indent=2)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REPORT_SYNTHESIS_PROMPT.format(kg_data=kg_data_str)}
        ]
        
        return self.generate(messages)


# ============================================================================
# KNOWLEDGE GRAPH DATA EXTRACTION WITH RICH CONTENT
# ============================================================================

def load_extended_data():
    """Load extended drug and protein data with full content from source databases"""
    global drug_full_data, protein_full_data
    
    kg_dir = Path("knowledge_graph_fact_based/neo4j_export")
    
    # Load full drug data
    drugs_path = kg_dir / "drugs.csv"
    if drugs_path.exists():
        try:
            df = pd.read_csv(drugs_path)
            for _, row in df.iterrows():
                drug_id = row['drugbank_id']
                drug_full_data[drug_id] = {
                    "drugbank_id": drug_id,
                    "name": row['name'] if pd.notna(row.get('name')) else drug_id,
                    "type": row['type'] if pd.notna(row.get('type')) else "Unknown",
                    "cas_number": row['cas_number'] if pd.notna(row.get('cas_number')) else None,
                    "atc_codes": row['atc_codes'] if pd.notna(row.get('atc_codes')) else None,
                    "groups": row['groups'] if pd.notna(row.get('groups')) else None,
                    "indication": row['indication'] if pd.notna(row.get('indication')) else None,
                    "mechanism_of_action": row['mechanism_of_action'] if pd.notna(row.get('mechanism_of_action')) else None,
                    "molecular_weight": row['molecular_weight'] if pd.notna(row.get('molecular_weight')) else None,
                    "pubchem_cid": row['pubchem_cid'] if pd.notna(row.get('pubchem_cid')) else None,
                    "chembl_id": row['chembl_id'] if pd.notna(row.get('chembl_id')) else None,
                    "kegg_id": row['kegg_id'] if pd.notna(row.get('kegg_id')) else None
                }
            print(f"   Loaded extended data for {len(drug_full_data):,} drugs")
        except Exception as e:
            print(f"   Warning: Could not load extended drug data: {e}")
    
    # Load full protein data
    proteins_path = kg_dir / "proteins.csv"
    if proteins_path.exists():
        try:
            df = pd.read_csv(proteins_path)
            for _, row in df.iterrows():
                prot_id = row['protein_id']
                protein_full_data[prot_id] = {
                    "protein_id": prot_id,
                    "name": row['name'] if pd.notna(row.get('name')) else prot_id,
                    "uniprot_id": row['uniprot_id'] if pd.notna(row.get('uniprot_id')) else None,
                    "gene_name": row['gene_name'] if pd.notna(row.get('gene_name')) else None,
                    "organism": row['organism'] if pd.notna(row.get('organism')) else None,
                    "general_function": row['general_function'] if pd.notna(row.get('general_function')) else None,
                    "specific_function": row['specific_function'] if pd.notna(row.get('specific_function')) else None,
                    "cellular_location": row['cellular_location'] if pd.notna(row.get('cellular_location')) else None
                }
            print(f"   Loaded extended data for {len(protein_full_data):,} proteins")
        except Exception as e:
            print(f"   Warning: Could not load extended protein data: {e}")


def extract_comprehensive_kg_data(drugs: List[str]) -> Dict:
    """Extract comprehensive data from knowledge graph with FULL content from source databases"""
    global kg_loader, assessor, drug_full_data, protein_full_data
    
    if not kg_loader or not assessor:
        return {"error": "Knowledge graph not loaded"}
    
    # Drug name aliases (common names to database names)
    DRUG_ALIASES = {
        'aspirin': 'acetylsalicylic acid',
        'tylenol': 'acetaminophen', 'advil': 'ibuprofen', 'motrin': 'ibuprofen',
        'aleve': 'naproxen', 'coumadin': 'warfarin', 'lipitor': 'atorvastatin',
        'zocor': 'simvastatin', 'crestor': 'rosuvastatin', 'nexium': 'esomeprazole',
        'prilosec': 'omeprazole', 'zoloft': 'sertraline', 'prozac': 'fluoxetine',
        'lexapro': 'escitalopram', 'xanax': 'alprazolam', 'valium': 'diazepam',
        'ativan': 'lorazepam', 'ambien': 'zolpidem', 'viagra': 'sildenafil',
        'glucophage': 'metformin', 'norvasc': 'amlodipine', 'prinivil': 'lisinopril',
        'zestril': 'lisinopril', 'lasix': 'furosemide', 'plavix': 'clopidogrel',
        'synthroid': 'levothyroxine', 'ultram': 'tramadol', 'neurontin': 'gabapentin',
    }
    
    # Resolve drug names to IDs and get FULL drug information
    drug_ids = []
    drug_names = []
    drug_profiles = []  # Full drug profiles with actual content
    
    for drug in drugs:
        drug_lower = drug.lower().strip()
        # Apply alias if exists
        drug_lower = DRUG_ALIASES.get(drug_lower, drug_lower)
        drug_id = kg_loader.drug_name_to_id.get(drug_lower)
        if drug_id:
            drug_ids.append(drug_id)
            drug_names.append(drug_lower)
            
            # Get FULL drug profile from extended data
            full_info = drug_full_data.get(drug_id, {})
            drug_profiles.append({
                "name": full_info.get("name", drug_lower),
                "drugbank_id": drug_id,
                "drugbank_url": f"https://go.drugbank.com/drugs/{drug_id}",
                "type": full_info.get("type"),
                "atc_codes": full_info.get("atc_codes"),
                "groups": full_info.get("groups"),
                "indication": full_info.get("indication"),  # ACTUAL indication text
                "mechanism_of_action": full_info.get("mechanism_of_action"),  # ACTUAL mechanism
                "molecular_weight": full_info.get("molecular_weight"),
                "external_ids": {
                    "pubchem": full_info.get("pubchem_cid"),
                    "chembl": full_info.get("chembl_id"),
                    "kegg": full_info.get("kegg_id"),
                    "cas": full_info.get("cas_number")
                },
                "source": "DrugBank Database"
            })
    
    if len(drug_ids) < 2:
        return {"error": f"Need at least 2 valid drugs. Found: {drug_names}"}
    
    # Get DDIs with FULL descriptions from database
    interactions = []
    for i, d1 in enumerate(drug_ids):
        for d2 in drug_ids[i+1:]:
            ddi = kg_loader.ddi_index.get((d1, d2))
            if ddi:
                drug1_info = drug_full_data.get(d1, {})
                drug2_info = drug_full_data.get(d2, {})
                interactions.append({
                    "drug1": drug1_info.get("name", d1),
                    "drug1_id": d1,
                    "drug1_url": f"https://go.drugbank.com/drugs/{d1}",
                    "drug2": drug2_info.get("name", d2),
                    "drug2_id": d2,
                    "drug2_url": f"https://go.drugbank.com/drugs/{d2}",
                    "severity": ddi.severity,
                    "description": ddi.description,  # FULL description from DrugBank
                    "interaction_url": f"https://go.drugbank.com/drugs/{d1}#interactions",
                    "source": "DrugBank DDI Database",
                    "citation": f"DrugBank [{d1}] + [{d2}]"
                })
    
    # Get side effect overlap with FULL names from SIDER
    side_effect_overlap = []
    all_side_effects = {}
    
    for drug_id in drug_ids:
        drug_name = drug_full_data.get(drug_id, {}).get("name", drug_id)
        for se_id in kg_loader.drug_side_effects.get(drug_id, []):
            if se_id not in all_side_effects:
                se_name = kg_loader.side_effect_names.get(se_id, se_id)
                all_side_effects[se_id] = {
                    "id": se_id,
                    "umls_cui": se_id,  # UMLS Concept Unique Identifier
                    "name": se_name,
                    "drugs": [],
                    "drug_ids": [],
                    "source": "SIDER Database (Side Effect Resource)",
                    "sider_url": f"http://sideeffects.embl.de/se/{se_id}/",
                    "umls_url": f"https://uts.nlm.nih.gov/uts/umls/concept/{se_id}"
                }
            all_side_effects[se_id]["drugs"].append(drug_name)
            all_side_effects[se_id]["drug_ids"].append(drug_id)
    
    # Filter to shared side effects (appearing in 2+ drugs)
    for se_id, se_data in all_side_effects.items():
        if len(se_data["drugs"]) >= 2:
            se_data["risk_note"] = f"This side effect may be amplified when {len(se_data['drugs'])} drugs are combined"
            side_effect_overlap.append(se_data)
    
    side_effect_overlap.sort(key=lambda x: -len(x["drugs"]))
    
    # Get protein target overlap with FULL protein information
    protein_overlap = []
    all_proteins = {}
    
    for drug_id in drug_ids:
        drug_name = drug_full_data.get(drug_id, {}).get("name", drug_id)
        for prot_id in kg_loader.drug_proteins.get(drug_id, []):
            if prot_id not in all_proteins:
                prot_info = protein_full_data.get(prot_id, {})
                all_proteins[prot_id] = {
                    "id": prot_id,
                    "name": prot_info.get("name", prot_id),
                    "uniprot_id": prot_info.get("uniprot_id"),
                    "gene_name": prot_info.get("gene_name"),
                    "general_function": prot_info.get("general_function"),  # ACTUAL function
                    "specific_function": prot_info.get("specific_function"),  # ACTUAL specific function
                    "cellular_location": prot_info.get("cellular_location"),
                    "drugs": [],
                    "drug_ids": [],
                    "uniprot_url": f"https://www.uniprot.org/uniprotkb/{prot_info.get('uniprot_id')}" if prot_info.get('uniprot_id') else None,
                    "source": "DrugBank Protein Targets"
                }
            all_proteins[prot_id]["drugs"].append(drug_name)
            all_proteins[prot_id]["drug_ids"].append(drug_id)
    
    # Filter to shared protein targets
    for prot_id, prot_data in all_proteins.items():
        if len(prot_data["drugs"]) >= 2:
            prot_data["mechanism_note"] = f"Shared target may lead to additive/synergistic effects or competition"
            protein_overlap.append(prot_data)
    
    protein_overlap.sort(key=lambda x: -len(x["drugs"]))
    
    # Get pathway overlap
    pathway_overlap = []
    all_pathways = {}
    
    for drug_id in drug_ids:
        drug_name = drug_full_data.get(drug_id, {}).get("name", drug_id)
        for path_id in kg_loader.drug_pathways.get(drug_id, []):
            if path_id not in all_pathways:
                all_pathways[path_id] = {
                    "id": path_id,
                    "drugs": [],
                    "drug_ids": [],
                    "kegg_url": f"https://www.kegg.jp/pathway/{path_id}" if path_id.startswith("hsa") else None,
                    "source": "KEGG/SMPDB Pathway Database"
                }
            all_pathways[path_id]["drugs"].append(drug_name)
            all_pathways[path_id]["drug_ids"].append(drug_id)
    
    for path_id, path_data in all_pathways.items():
        if len(path_data["drugs"]) >= 2:
            pathway_overlap.append(path_data)
    
    pathway_overlap.sort(key=lambda x: -len(x["drugs"]))
    
    # Calculate risk score
    try:
        risk_result = assessor.assess_polypharmacy_risk(drugs)
        risk_score = risk_result.overall_risk_score
        risk_level = risk_result.risk_level
    except:
        risk_score = 0.0
        risk_level = "Unknown"
    
    return {
        "drugs": drug_names,
        "drug_profiles": drug_profiles,  # FULL drug profiles with indications and mechanisms
        "interactions": interactions,  # FULL interaction descriptions
        "side_effects": side_effect_overlap[:25],  # Top 25 with SIDER data
        "proteins": protein_overlap[:20],  # Top 20 with FULL protein function data
        "pathways": pathway_overlap[:15],  # Top 15 
        "risk_score": risk_score,
        "risk_level": risk_level,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "sources": [
                {"name": "DrugBank", "url": "https://go.drugbank.com/", "description": "Comprehensive drug database"},
                {"name": "SIDER", "url": "http://sideeffects.embl.de/", "description": "Side Effect Resource"},
                {"name": "UniProt", "url": "https://www.uniprot.org/", "description": "Protein sequence and function"},
                {"name": "KEGG", "url": "https://www.kegg.jp/", "description": "Biological pathway database"},
                {"name": "UMLS", "url": "https://www.nlm.nih.gov/research/umls/", "description": "Unified Medical Language System"}
            ],
            "drug_count": len(drug_ids),
            "interaction_count": len(interactions),
            "shared_side_effects": len(side_effect_overlap),
            "shared_proteins": len(protein_overlap),
            "shared_pathways": len(pathway_overlap)
        }
    }


# ============================================================================
# HTTP HANDLER
# ============================================================================

llm_client = None


class DDIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def _send_json(self, data, status=200):
        response = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)
    
    def _send_html(self, html):
        response = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(response))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(response)
    
    def do_GET(self):
        # Handle root and index paths
        if self.path in ['/', '/index.html', '/index.htm']:
            self._send_html(HTML_PAGE)
        elif self.path == '/status':
            status = dict(loading_status)
            if llm_client:
                status['llm'] = llm_client.model_name
                status['current_model'] = llm_client.current_model
                status['available_models'] = llm_client.get_available_models()
            else:
                status['llm'] = 'initializing'
                status['current_model'] = ''
                status['available_models'] = {}
            self._send_json(status)
        elif self.path == '/favicon.ico':
            # Return empty favicon to avoid 404
            self.send_response(204)
            self.end_headers()
        else:
            # For any other path, serve the main page
            self._send_html(HTML_PAGE)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body)
        except:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == '/analyze':
            result = analyze_drugs(data.get('drugs', ''))
            self._send_json(result)
        elif self.path == '/comprehensive_report':
            result = generate_comprehensive_report(data.get('drugs', ''))
            self._send_json(result)
        elif self.path == '/chat':
            result = handle_chat(data.get('message', ''), data.get('history', []))
            self._send_json(result)
        elif self.path == '/set_model':
            result = set_model(data.get('model', ''))
            self._send_json(result)
        else:
            self.send_error(404)


def analyze_drugs(drug_input: str) -> dict:
    """Quick analysis of drug interactions"""
    global assessor
    
    if loading_status["loading"]:
        return {"error": "Knowledge graph is still loading..."}
    
    if not assessor:
        return {"error": "Knowledge graph not available"}
    
    if not drug_input:
        return {"error": "Please enter drug names"}
    
    drugs = [d.strip().lower() for d in drug_input.replace(';', ',').split(',') if d.strip()]
    
    if len(drugs) < 2:
        return {"error": "Please enter at least 2 drugs"}
    
    try:
        result = assessor.assess_polypharmacy_risk(drugs)
        
        interactions = []
        for ddi in result.ddi_pairs:
            interactions.append({
                "drug1": ddi.get('drug1', 'Unknown'),
                "drug2": ddi.get('drug2', 'Unknown'),
                "severity": ddi.get('severity', 'Unknown'),
                "description": (ddi.get('description', '') or '')[:200]
            })
        
        return {
            "drugs": drugs,
            "risk_level": result.risk_level,
            "risk_score": result.overall_risk_score,
            "interactions": interactions
        }
    except Exception as e:
        return {"error": str(e)}


def generate_comprehensive_report(drug_input: str) -> dict:
    """Generate comprehensive report with KG data and LLM synthesis"""
    global llm_client
    
    if loading_status["loading"]:
        return {"error": "Knowledge graph is still loading..."}
    
    if not drug_input:
        return {"error": "Please enter drug names"}
    
    drugs = [d.strip().lower() for d in drug_input.replace(';', ',').split(',') if d.strip()]
    
    if len(drugs) < 2:
        return {"error": "Please enter at least 2 drugs"}
    
    # Extract comprehensive KG data
    kg_data = extract_comprehensive_kg_data(drugs)
    
    if "error" in kg_data:
        return kg_data
    
    # Synthesize report with LLM
    llm_report = None
    if llm_client and llm_client.backend == 'ollama':
        try:
            llm_report = llm_client.synthesize_report(kg_data)
        except Exception as e:
            llm_report = f"[LLM synthesis failed: {str(e)}]"
    
    return {
        "drugs": kg_data.get("drugs", drugs),
        "risk_level": kg_data.get("risk_level", "Unknown"),
        "risk_score": kg_data.get("risk_score", 0.0),
        "kg_data": kg_data,
        "llm_report": llm_report,
        "metadata": kg_data.get("metadata", {})
    }


def handle_chat(message: str, history: List[Dict]) -> dict:
    """Handle chat messages"""
    global llm_client
    
    if not message:
        return {"error": "Empty message"}
    
    if not llm_client:
        return {"error": "Chat not initialized"}
    
    # Extract drugs and get KG context
    drugs = extract_drugs_from_text(message)
    kg_context = ""
    
    if len(drugs) >= 2:
        kg_data = extract_comprehensive_kg_data(drugs)
        if "error" not in kg_data:
            kg_context = f"\n\n[Knowledge Graph Context: {len(kg_data.get('interactions', []))} interactions found for {', '.join(drugs)}]"
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-8:]:
        messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
    messages.append({"role": "user", "content": message + kg_context})
    
    try:
        response = llm_client.generate(messages)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


def extract_drugs_from_text(text: str) -> List[str]:
    """Extract drug names from text"""
    if not kg_loader:
        return []
    
    text_lower = text.lower()
    found = []
    
    for drug_name in kg_loader.drug_name_to_id.keys():
        if drug_name in text_lower and len(drug_name) > 3:
            found.append(drug_name)
    
    return list(set(found))


def set_model(model_key: str) -> dict:
    """Switch LLM model"""
    global llm_client
    
    if not llm_client:
        return {"success": False, "error": "LLM not initialized"}
    
    if llm_client.set_model(model_key):
        return {"success": True, "model": model_key}
    return {"success": False, "error": f"Model '{model_key}' not available"}


def load_knowledge_graph():
    """Load knowledge graph and extended data synchronously"""
    global assessor, kg_loader, loading_status, llm_client
    
    print("Initializing LLM client...")
    llm_client = LLMClient()
    
    if not KG_AVAILABLE:
        loading_status = {"loading": False, "message": "KG module not found"}
        return
    
    try:
        print("Loading knowledge graph...")
        kg_loader = KnowledgeGraphLoader()
        kg_loader.load()
        assessor = PolypharmacyRiskAssessor(kg_loader)
        
        print("Loading extended drug data...")
        load_extended_data()
        
        loading_status = {"loading": False, "message": f"{len(kg_loader.drugs):,} drugs, {len(kg_loader.ddis):,} DDIs"}
        print(f"✅ Ready: {len(kg_loader.drugs):,} drugs, {len(kg_loader.ddis):,} DDIs")
    except Exception as e:
        loading_status = {"loading": False, "message": f"Error: {str(e)[:50]}"}
        print(f"❌ Error: {e}")


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    PORT = 8080
    
    # Load data FIRST (synchronously, before starting server)
    load_knowledge_graph()
    
    print("\n" + "=" * 60)
    print("  DDI Comprehensive Analyzer")
    print("=" * 60)
    print(f"\n  🌐 Open: http://localhost:{PORT}")
    print("\n  Features:")
    print("    • Quick drug interaction check")
    print("    • Comprehensive report with citations")
    print("    • Agentic LLM synthesis")
    print("    • Natural language chat")
    print("\n  Press Ctrl+C to stop\n")
    
    with ReusableTCPServer(("0.0.0.0", PORT), DDIHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
