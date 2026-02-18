#!/usr/bin/env python3
"""
DDI Risk Analysis GUI - Simplified Version
Loads all data before starting server to avoid threading issues
"""
import http.server
import socketserver
import json
import urllib.parse
import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Configuration
PORT = 8080
KG_DIR = Path(__file__).parent / 'knowledge_graph_fact_based' / 'neo4j_export'

# Global data stores (loaded at startup)
DRUGS = {}
DRUG_NAME_TO_ID = {}
DDIS = {}
SIDE_EFFECTS = {}
SE_NAMES = {}
READY = False

# Drug name aliases
ALIASES = {
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

def load_data():
    """Load all data synchronously at startup"""
    global DRUGS, DRUG_NAME_TO_ID, DDIS, SIDE_EFFECTS, SE_NAMES, READY
    
    print("Loading Knowledge Graph...")
    
    # Load drugs
    drugs_path = KG_DIR / "drugs.csv"
    if drugs_path.exists():
        df = pd.read_csv(drugs_path, low_memory=False)
        for _, row in df.iterrows():
            drug_id = row.get('drugbank_id', '')
            name = str(row.get('name', '')).lower()
            DRUGS[drug_id] = {
                'id': drug_id,
                'name': row.get('name', ''),
                'indication': str(row.get('indication', ''))[:500],
                'mechanism': str(row.get('mechanism_of_action', ''))[:500],
            }
            DRUG_NAME_TO_ID[name] = drug_id
            DRUG_NAME_TO_ID[drug_id.lower()] = drug_id
        # Add aliases
        for alias, real in ALIASES.items():
            if real in DRUG_NAME_TO_ID and alias not in DRUG_NAME_TO_ID:
                DRUG_NAME_TO_ID[alias] = DRUG_NAME_TO_ID[real]
        print(f"  ✓ {len(DRUGS)} drugs")
    
    # Load DDIs
    ddi_path = KG_DIR / "ddi_edges.csv"
    if ddi_path.exists():
        df = pd.read_csv(ddi_path, low_memory=False)
        for _, row in df.iterrows():
            d1, d2 = row.get('drug1_id', ''), row.get('drug2_id', '')
            ddi = {
                'drug1': d1, 'drug2': d2,
                'description': str(row.get('description', ''))[:300],
                'severity': row.get('severity', 'Unknown')
            }
            DDIS[(d1, d2)] = ddi
            DDIS[(d2, d1)] = ddi
        print(f"  ✓ {len(df)} DDIs")
    
    # Load side effects
    se_edge_path = KG_DIR / "drug_side_effect_edges.csv"
    if se_edge_path.exists():
        df = pd.read_csv(se_edge_path, low_memory=False)
        for _, row in df.iterrows():
            drug_id = row.get('drug_id', '')
            se_id = row.get('side_effect_id', '')
            if drug_id not in SIDE_EFFECTS:
                SIDE_EFFECTS[drug_id] = set()
            SIDE_EFFECTS[drug_id].add(se_id)
        print(f"  ✓ {len(df)} side effect links")
    
    se_path = KG_DIR / "side_effects.csv"
    if se_path.exists():
        df = pd.read_csv(se_path, low_memory=False)
        for _, row in df.iterrows():
            se_id = row.get('umls_cui', row.get('id', ''))
            SE_NAMES[se_id] = row.get('name', row.get('side_effect_name', ''))
        print(f"  ✓ {len(SE_NAMES)} side effect names")
    
    READY = True
    print(f"\n✅ Ready! {len(DRUGS)} drugs, {len(DDIS)//2} DDIs\n")


def resolve_drug(name):
    """Resolve drug name to ID"""
    return DRUG_NAME_TO_ID.get(name.lower().strip(), '')


def analyze(drug_names):
    """Analyze drug interactions"""
    resolved = []
    for name in drug_names:
        drug_id = resolve_drug(name)
        if drug_id:
            resolved.append((name, drug_id))
    
    if len(resolved) < 2:
        return {'error': f'Need at least 2 recognized drugs (found {len(resolved)})'}
    
    # Get interactions
    interactions = []
    drug_ids = [d[1] for d in resolved]
    for i in range(len(drug_ids)):
        for j in range(i+1, len(drug_ids)):
            key = (drug_ids[i], drug_ids[j])
            if key in DDIS:
                ddi = DDIS[key]
                interactions.append({
                    'drug1': resolved[i][0],
                    'drug2': resolved[j][0],
                    'severity': ddi.get('severity', 'Unknown'),
                    'description': ddi.get('description', '')
                })
    
    # Calculate risk
    risk = 0
    for ddi in interactions:
        sev = ddi.get('severity', '').lower()
        if 'major' in sev or 'contraindicated' in sev:
            risk += 0.35
        elif 'moderate' in sev:
            risk += 0.2
        else:
            risk += 0.1
    risk = min(risk, 1.0)
    
    level = 'MINIMAL' if risk == 0 else ('LOW' if risk < 0.3 else ('MODERATE' if risk < 0.6 else 'HIGH'))
    
    # Get shared side effects
    shared_se = []
    se_sets = [SIDE_EFFECTS.get(did, set()) for did in drug_ids]
    if all(se_sets):
        common = se_sets[0].intersection(*se_sets[1:])
        shared_se = [SE_NAMES.get(se_id, se_id) for se_id in list(common)[:10]]
    
    return {
        'drugs': [{'name': n, 'id': did, 'info': DRUGS.get(did, {})} for n, did in resolved],
        'interactions': interactions,
        'risk_score': risk,
        'risk_level': level,
        'shared_side_effects': shared_se
    }


HTML_PAGE = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>DDI Risk Analyzer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; padding: 20px; }
.container { max-width: 900px; margin: 0 auto; }
h1 { color: #1e40af; margin-bottom: 20px; }
.card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.input-group { display: flex; gap: 10px; margin-bottom: 15px; }
input[type="text"] { flex: 1; padding: 12px 16px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 16px; }
input[type="text"]:focus { outline: none; border-color: #2563eb; }
button { padding: 12px 24px; background: #2563eb; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
button:hover { background: #1d4ed8; }
.examples { margin-top: 10px; font-size: 14px; color: #6b7280; }
.examples a { color: #2563eb; cursor: pointer; text-decoration: underline; }
.status { padding: 10px 16px; border-radius: 8px; margin-bottom: 15px; font-weight: 500; }
.status.ready { background: #dcfce7; color: #166534; }
.status.loading { background: #fef3c7; color: #92400e; }
.results { display: none; }
.risk-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 18px; margin-bottom: 15px; }
.risk-HIGH { background: #fee2e2; color: #dc2626; }
.risk-MODERATE { background: #ffedd5; color: #ea580c; }
.risk-LOW { background: #fef9c3; color: #ca8a04; }
.risk-MINIMAL { background: #dcfce7; color: #16a34a; }
.drug-card { background: #f8fafc; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #2563eb; }
.drug-card h3 { color: #1e40af; margin-bottom: 8px; }
.drug-card p { font-size: 14px; color: #4b5563; line-height: 1.5; }
.drug-card a { color: #2563eb; font-size: 12px; }
.interaction { background: #fff7ed; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #f97316; }
.interaction h4 { margin-bottom: 8px; }
.interaction .severity { font-weight: 600; }
.sev-major { color: #dc2626; }
.sev-moderate { color: #ea580c; }
.sev-minor { color: #16a34a; }
.side-effects { background: #fdf4ff; border-radius: 8px; padding: 15px; }
.side-effects h3 { color: #7c3aed; margin-bottom: 10px; }
.side-effects ul { list-style: none; display: flex; flex-wrap: wrap; gap: 8px; }
.side-effects li { background: #f3e8ff; color: #6b21a8; padding: 4px 12px; border-radius: 20px; font-size: 13px; }
#loading { display: none; text-align: center; padding: 40px; }
.spinner { width: 40px; height: 40px; border: 4px solid #e5e7eb; border-top-color: #2563eb; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
<h1>💊 DDI Risk Analyzer</h1>

<div class="card">
<div id="status" class="status loading">⏳ Loading knowledge graph...</div>
<div class="input-group">
<input type="text" id="drugs" placeholder="Enter drug names (comma-separated)">
<button onclick="analyze()">Analyze</button>
</div>
<div class="examples">
Try: 
<a onclick="setDrugs('warfarin, aspirin, ibuprofen')">warfarin + aspirin + ibuprofen</a> |
<a onclick="setDrugs('metformin, lisinopril, amlodipine')">metformin + lisinopril + amlodipine</a> |
<a onclick="setDrugs('sertraline, tramadol')">sertraline + tramadol</a>
</div>
</div>

<div id="loading" class="card">
<div class="spinner"></div>
<p>Analyzing drug interactions...</p>
</div>

<div id="results" class="results"></div>
</div>

<script>
function setDrugs(d) { document.getElementById('drugs').value = d; }

function checkStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(d => {
            const s = document.getElementById('status');
            if (d.ready) {
                s.className = 'status ready';
                s.textContent = '✅ Ready: ' + d.drugs + ' drugs, ' + d.ddis + ' interactions';
            } else {
                s.className = 'status loading';
                s.textContent = '⏳ Loading...';
                setTimeout(checkStatus, 1000);
            }
        })
        .catch(() => setTimeout(checkStatus, 1000));
}

function analyze() {
    const drugs = document.getElementById('drugs').value.trim();
    if (!drugs) { alert('Please enter drug names'); return; }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({drugs: drugs})
    })
    .then(r => r.json())
    .then(d => {
        document.getElementById('loading').style.display = 'none';
        showResults(d);
    })
    .catch(e => {
        document.getElementById('loading').style.display = 'none';
        alert('Error: ' + e.message);
    });
}

function showResults(data) {
    const results = document.getElementById('results');
    
    if (data.error) {
        results.innerHTML = '<div class="card"><p style="color:#dc2626">⚠️ ' + data.error + '</p></div>';
        results.style.display = 'block';
        return;
    }
    
    let html = '<div class="card">';
    html += '<div class="risk-badge risk-' + data.risk_level + '">';
    html += data.risk_level + ' RISK (' + Math.round(data.risk_score * 100) + '%)</div>';
    
    // Drug profiles
    html += '<h3 style="margin:15px 0 10px">Drug Profiles</h3>';
    data.drugs.forEach(d => {
        const info = d.info || {};
        html += '<div class="drug-card">';
        html += '<h3>' + (info.name || d.name).toUpperCase() + '</h3>';
        html += '<a href="https://go.drugbank.com/drugs/' + d.id + '" target="_blank">DrugBank: ' + d.id + '</a>';
        if (info.indication && info.indication !== 'nan') {
            html += '<p><strong>Indication:</strong> ' + info.indication.substring(0, 300) + '...</p>';
        }
        if (info.mechanism && info.mechanism !== 'nan') {
            html += '<p><strong>Mechanism:</strong> ' + info.mechanism.substring(0, 300) + '...</p>';
        }
        html += '</div>';
    });
    
    // Interactions
    html += '<h3 style="margin:20px 0 10px">Interactions Found: ' + data.interactions.length + '</h3>';
    if (data.interactions.length === 0) {
        html += '<p style="color:#16a34a">✅ No direct interactions found in database</p>';
    } else {
        data.interactions.forEach(i => {
            const sevClass = i.severity.toLowerCase().includes('major') ? 'major' : 
                           (i.severity.toLowerCase().includes('moderate') ? 'moderate' : 'minor');
            html += '<div class="interaction">';
            html += '<h4>' + i.drug1.toUpperCase() + ' ↔ ' + i.drug2.toUpperCase() + '</h4>';
            html += '<p class="severity sev-' + sevClass + '">' + i.severity + '</p>';
            html += '<p>' + i.description + '</p>';
            html += '</div>';
        });
    }
    
    // Side effects
    if (data.shared_side_effects && data.shared_side_effects.length > 0) {
        html += '<div class="side-effects"><h3>⚡ Shared Side Effects</h3><ul>';
        data.shared_side_effects.forEach(se => {
            html += '<li>' + se + '</li>';
        });
        html += '</ul></div>';
    }
    
    html += '</div>';
    results.innerHTML = html;
    results.style.display = 'block';
}

window.onload = checkStatus;
</script>
</body>
</html>'''


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
    
    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        if self.path == '/status':
            self.send_json({
                'ready': READY,
                'drugs': len(DRUGS),
                'ddis': len(DDIS) // 2
            })
        else:
            self.send_html(HTML_PAGE)
    
    def do_POST(self):
        if self.path == '/analyze':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode()
            try:
                data = json.loads(body)
                drugs = [d.strip() for d in data.get('drugs', '').split(',') if d.strip()]
                result = analyze(drugs)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    # Load data FIRST (before starting server)
    load_data()
    
    print(f"Starting server on http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop\n")
    
    with ThreadedServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()


if __name__ == '__main__':
    main()
