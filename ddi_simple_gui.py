#!/usr/bin/env python3
"""
Simple Web GUI for DDI Analysis - No External Dependencies
Run with: python ddi_simple_gui.py
Then open: http://localhost:8080
"""

import http.server
import socketserver
import json
import threading
import urllib.parse
from io import BytesIO

# Import the DDI analysis components
try:
    from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

# Global state
assessor = None
loading_status = {"loading": True, "message": "Initializing..."}

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDI Analyzer</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,-apple-system,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;padding:20px}
        .container{max-width:850px;margin:0 auto}
        .card{background:#fff;border-radius:16px;box-shadow:0 10px 40px rgba(0,0,0,.2);padding:25px;margin-bottom:20px}
        h1{color:#333;text-align:center;margin-bottom:8px;font-size:26px}
        .subtitle{text-align:center;color:#666;margin-bottom:20px;font-size:14px}
        label{display:block;font-weight:600;margin-bottom:8px;color:#444}
        input[type=text]{width:100%;padding:14px;border:2px solid #e0e0e0;border-radius:10px;font-size:16px;transition:.3s}
        input:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 3px rgba(102,126,234,.2)}
        .hint{color:#888;font-size:12px;margin-top:5px}
        .btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:15px}
        .btn{padding:12px 24px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:.2s}
        .btn:hover{transform:translateY(-2px)}
        .btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 4px 15px rgba(102,126,234,.4)}
        .btn-secondary{background:#f0f0f0;color:#333}
        .btn-ex{background:#e8f4f8;color:#2196F3;padding:8px 14px;font-size:12px}
        .examples{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center}
        .examples span{color:#666;font-size:12px}
        #results{display:none}
        .result-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #f0f0f0}
        .risk-badge{padding:8px 18px;border-radius:20px;font-weight:700;font-size:13px;text-transform:uppercase}
        .risk-high{background:#ffebee;color:#c62828}
        .risk-moderate{background:#fff3e0;color:#ef6c00}
        .risk-low{background:#e8f5e9;color:#2e7d32}
        .risk-score{font-size:32px;font-weight:700;color:#333}
        .risk-score span{font-size:14px;color:#888;font-weight:400}
        .int-list{list-style:none}
        .int-item{padding:14px;margin-bottom:10px;border-radius:10px;border-left:4px solid}
        .int-item.contraindicated{background:#ffebee;border-color:#c62828}
        .int-item.major{background:#fff3e0;border-color:#ef6c00}
        .int-item.moderate{background:#fffde7;border-color:#f9a825}
        .int-item.minor{background:#e8f5e9;border-color:#2e7d32}
        .int-drugs{font-weight:600;font-size:15px;color:#333;margin-bottom:4px}
        .int-sev{font-size:11px;font-weight:600;text-transform:uppercase;margin-bottom:4px}
        .int-desc{font-size:13px;color:#666;line-height:1.4}
        .no-int{text-align:center;padding:30px;color:#2e7d32;font-size:16px}
        .loading{text-align:center;padding:30px}
        .spinner{border:4px solid #f3f3f3;border-top:4px solid #667eea;border-radius:50%;width:36px;height:36px;animation:spin 1s linear infinite;margin:0 auto 12px}
        @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
        .status{text-align:center;padding:8px;font-size:12px;color:#666}
        .status-ready{color:#2e7d32}
        .status-loading{color:#ef6c00}
        .error{color:#c62828;text-align:center;padding:20px}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>💊 Drug-Drug Interaction Analyzer</h1>
        <p class="subtitle">Knowledge Graph-Based Risk Assessment</p>
        <div style="margin-bottom:15px">
            <label>Enter Drug Names</label>
            <input type="text" id="drugs" placeholder="e.g., warfarin, aspirin, ibuprofen">
            <p class="hint">Separate multiple drugs with commas</p>
        </div>
        <div class="btns">
            <button class="btn btn-primary" onclick="analyze()">🔍 Analyze</button>
            <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
        </div>
        <div class="examples">
            <span>Examples:</span>
            <button class="btn btn-ex" onclick="setDrugs('warfarin, aspirin, ibuprofen')">High Risk</button>
            <button class="btn btn-ex" onclick="setDrugs('metformin, lisinopril')">Moderate</button>
            <button class="btn btn-ex" onclick="setDrugs('acetaminophen, omeprazole')">Low</button>
        </div>
    </div>
    <div id="results" class="card"></div>
    <div class="status" id="status">Checking...</div>
</div>
<script>
window.onload=function(){checkStatus();document.getElementById('drugs').onkeypress=function(e){if(e.key==='Enter')analyze()}};
function checkStatus(){fetch('/status').then(r=>r.json()).then(d=>{var s=document.getElementById('status');if(d.loading){s.className='status status-loading';s.textContent='⏳ '+d.message;setTimeout(checkStatus,1000)}else{s.className='status status-ready';s.textContent='✅ '+d.message}})}
function setDrugs(d){document.getElementById('drugs').value=d;analyze()}
function clearAll(){document.getElementById('drugs').value='';document.getElementById('results').style.display='none'}
function analyze(){var drugs=document.getElementById('drugs').value.trim();if(!drugs){alert('Enter drug names');return}var r=document.getElementById('results');r.style.display='block';r.innerHTML='<div class="loading"><div class="spinner"></div><p>Analyzing...</p></div>';fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({drugs:drugs})}).then(r=>r.json()).then(d=>{if(d.error){r.innerHTML='<p class="error">⚠️ '+d.error+'</p>';return}showResults(d)}).catch(e=>{r.innerHTML='<p class="error">⚠️ '+e+'</p>'})}
function showResults(d){var r=document.getElementById('results');var rc=d.risk_level.toLowerCase();var cls=rc==='high'||rc==='critical'?'risk-high':rc==='moderate'?'risk-moderate':'risk-low';var h='<div class="result-header"><span class="risk-badge '+cls+'">'+d.risk_level+' Risk</span><div class="risk-score">'+d.risk_score.toFixed(2)+' <span>/ 1.00</span></div></div><p style="margin-bottom:15px;color:#666"><b>Drugs:</b> '+d.drugs.join(', ')+'</p>';if(d.interactions.length===0){h+='<div class="no-int">✅ No significant interactions</div>'}else{h+='<h3 style="margin-bottom:12px">Interactions ('+d.interactions.length+')</h3><ul class="int-list">';for(var i of d.interactions){var sc=i.severity.toLowerCase().replace(' ','-');h+='<li class="int-item '+sc+'"><div class="int-drugs">'+i.drug1+' ↔ '+i.drug2+'</div><div class="int-sev">'+i.severity+'</div><div class="int-desc">'+(i.description||'')+'</div></li>'}h+='</ul>'}r.innerHTML=h}
</script>
</body>
</html>"""


class DDIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for DDI API"""
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
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
        self.end_headers()
        self.wfile.write(response)
    
    def do_GET(self):
        if self.path == '/':
            self._send_html(HTML_PAGE)
        elif self.path == '/status':
            self._send_json(loading_status)
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            
            try:
                data = json.loads(body)
                result = analyze_drugs(data.get('drugs', ''))
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_error(404)


def analyze_drugs(drug_input: str) -> dict:
    """Analyze drug interactions"""
    global assessor
    
    if loading_status["loading"]:
        return {"error": "Knowledge graph is still loading. Please wait."}
    
    if not assessor:
        return {"error": "Knowledge graph not available."}
    
    if not drug_input:
        return {"error": "Please enter drug names."}
    
    # Parse drugs
    drugs = [d.strip().lower() for d in drug_input.replace(';', ',').split(',') if d.strip()]
    
    if len(drugs) < 2:
        return {"error": "Please enter at least 2 drugs."}
    
    try:
        result = assessor.assess_polypharmacy_risk(drugs)
        
        interactions = []
        for ddi in result.ddi_pairs:
            interactions.append({
                "drug1": ddi.get('drug1', 'Unknown'),
                "drug2": ddi.get('drug2', 'Unknown'),
                "severity": ddi.get('severity', 'Unknown'),
                "description": (ddi.get('description', '') or '')[:150]
            })
        
        return {
            "drugs": drugs,
            "risk_level": result.risk_level,
            "risk_score": result.overall_risk_score,
            "interactions": interactions
        }
        
    except Exception as e:
        return {"error": str(e)}


def load_knowledge_graph():
    """Load knowledge graph in background"""
    global assessor, loading_status
    
    if not KG_AVAILABLE:
        loading_status = {"loading": False, "message": "KG module not found"}
        return
    
    try:
        loading_status = {"loading": True, "message": "Loading knowledge graph..."}
        loader = KnowledgeGraphLoader()
        kg = loader.load()
        assessor = PolypharmacyRiskAssessor(kg)
        loading_status = {"loading": False, "message": f"{len(kg.drugs)} drugs, {len(kg.ddis)} DDIs loaded"}
    except Exception as e:
        loading_status = {"loading": False, "message": f"Error: {str(e)[:40]}"}


def main():
    PORT = 8080
    
    # Load KG in background
    thread = threading.Thread(target=load_knowledge_graph, daemon=True)
    thread.start()
    
    print("\n" + "=" * 50)
    print("  DDI Analyzer - Web Interface")
    print("=" * 50)
    print(f"\n  🌐 Open: http://localhost:{PORT}")
    print("\n  Press Ctrl+C to stop\n")
    
    with socketserver.TCPServer(("", PORT), DDIHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
