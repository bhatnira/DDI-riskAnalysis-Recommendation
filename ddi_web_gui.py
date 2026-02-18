#!/usr/bin/env python3
"""
Simple Web GUI for Drug-Drug Interaction Analysis
Run with: python ddi_web_gui.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import threading

# Import the DDI analysis components
try:
    from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

app = Flask(__name__)

# Global assessor
assessor = None
loading_status = {"loading": True, "message": "Initializing..."}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDI Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
            margin-bottom: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }
        .input-section {
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #444;
        }
        input[type="text"] {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        }
        .hint {
            color: #888;
            font-size: 13px;
            margin-top: 6px;
        }
        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:active { transform: translateY(0); }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        .btn-example {
            background: #e8f4f8;
            color: #2196F3;
            padding: 8px 16px;
            font-size: 13px;
        }
        .examples {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .examples span {
            color: #666;
            font-size: 13px;
            line-height: 32px;
        }
        #results {
            display: none;
            margin-top: 20px;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        .risk-badge {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
        }
        .risk-high { background: #ffebee; color: #c62828; }
        .risk-moderate { background: #fff3e0; color: #ef6c00; }
        .risk-low { background: #e8f5e9; color: #2e7d32; }
        .risk-score {
            font-size: 36px;
            font-weight: 700;
            color: #333;
        }
        .risk-score span {
            font-size: 14px;
            color: #888;
            font-weight: 400;
        }
        .interaction-list {
            list-style: none;
        }
        .interaction-item {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 10px;
            border-left: 4px solid;
        }
        .interaction-item.contraindicated { 
            background: #ffebee; 
            border-color: #c62828;
        }
        .interaction-item.major { 
            background: #fff3e0; 
            border-color: #ef6c00;
        }
        .interaction-item.moderate { 
            background: #fffde7; 
            border-color: #f9a825;
        }
        .interaction-item.minor { 
            background: #e8f5e9; 
            border-color: #2e7d32;
        }
        .interaction-drugs {
            font-weight: 600;
            font-size: 16px;
            color: #333;
            margin-bottom: 5px;
        }
        .interaction-severity {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .interaction-desc {
            font-size: 14px;
            color: #666;
            line-height: 1.5;
        }
        .no-interactions {
            text-align: center;
            padding: 40px;
            color: #2e7d32;
            font-size: 18px;
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status-bar {
            text-align: center;
            padding: 10px;
            font-size: 13px;
            color: #666;
        }
        .status-ready { color: #2e7d32; }
        .status-loading { color: #ef6c00; }
        .error { color: #c62828; }
        @media (max-width: 600px) {
            .card { padding: 20px; }
            h1 { font-size: 22px; }
            .risk-score { font-size: 28px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>💊 Drug-Drug Interaction Analyzer</h1>
            <p class="subtitle">Knowledge Graph-Based Risk Assessment</p>
            
            <div class="input-section">
                <label for="drugs">Enter Drug Names</label>
                <input type="text" id="drugs" placeholder="e.g., warfarin, aspirin, ibuprofen" autocomplete="off">
                <p class="hint">Separate multiple drugs with commas</p>
            </div>
            
            <div class="buttons">
                <button class="btn btn-primary" onclick="analyze()">🔍 Analyze Interactions</button>
                <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
            </div>
            
            <div class="examples">
                <span>Quick examples:</span>
                <button class="btn btn-example" onclick="setDrugs('warfarin, aspirin, ibuprofen')">High Risk</button>
                <button class="btn btn-example" onclick="setDrugs('metformin, lisinopril, amlodipine')">Moderate</button>
                <button class="btn btn-example" onclick="setDrugs('acetaminophen, omeprazole')">Low Risk</button>
            </div>
        </div>
        
        <div id="results" class="card">
            <div id="results-content"></div>
        </div>
        
        <div class="status-bar" id="status">
            Checking system status...
        </div>
    </div>
    
    <script>
        // Check status on load
        window.onload = function() {
            checkStatus();
            document.getElementById('drugs').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') analyze();
            });
        };
        
        function checkStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const status = document.getElementById('status');
                    if (data.loading) {
                        status.className = 'status-bar status-loading';
                        status.textContent = '⏳ ' + data.message;
                        setTimeout(checkStatus, 1000);
                    } else {
                        status.className = 'status-bar status-ready';
                        status.textContent = '✅ ' + data.message;
                    }
                });
        }
        
        function setDrugs(drugs) {
            document.getElementById('drugs').value = drugs;
            analyze();
        }
        
        function clearAll() {
            document.getElementById('drugs').value = '';
            document.getElementById('results').style.display = 'none';
        }
        
        function analyze() {
            const drugs = document.getElementById('drugs').value.trim();
            if (!drugs) {
                alert('Please enter drug names');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            const content = document.getElementById('results-content');
            
            resultsDiv.style.display = 'block';
            content.innerHTML = '<div class="loading"><div class="spinner"></div><p>Analyzing interactions...</p></div>';
            
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({drugs: drugs})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    content.innerHTML = '<p class="error">⚠️ ' + data.error + '</p>';
                    return;
                }
                displayResults(data);
            })
            .catch(err => {
                content.innerHTML = '<p class="error">⚠️ Error: ' + err.message + '</p>';
            });
        }
        
        function displayResults(data) {
            const content = document.getElementById('results-content');
            
            let riskClass = 'risk-low';
            if (data.risk_level.toLowerCase() === 'high' || data.risk_level.toLowerCase() === 'critical') {
                riskClass = 'risk-high';
            } else if (data.risk_level.toLowerCase() === 'moderate') {
                riskClass = 'risk-moderate';
            }
            
            let html = `
                <div class="result-header">
                    <div>
                        <span class="risk-badge ${riskClass}">${data.risk_level} Risk</span>
                    </div>
                    <div class="risk-score">
                        ${data.risk_score.toFixed(2)} <span>/ 1.00</span>
                    </div>
                </div>
                <p style="margin-bottom: 15px; color: #666;"><strong>Drugs analyzed:</strong> ${data.drugs.join(', ')}</p>
            `;
            
            if (data.interactions.length === 0) {
                html += '<div class="no-interactions">✅ No significant interactions found</div>';
            } else {
                html += '<h3 style="margin-bottom: 15px;">Interactions Found (' + data.interactions.length + ')</h3>';
                html += '<ul class="interaction-list">';
                
                for (const int of data.interactions) {
                    const sevClass = int.severity.toLowerCase().replace(' ', '-');
                    html += `
                        <li class="interaction-item ${sevClass}">
                            <div class="interaction-drugs">${int.drug1} ↔ ${int.drug2}</div>
                            <div class="interaction-severity">${int.severity}</div>
                            <div class="interaction-desc">${int.description || 'No description available'}</div>
                        </li>
                    `;
                }
                
                html += '</ul>';
            }
            
            content.innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify(loading_status)

@app.route('/analyze', methods=['POST'])
def analyze():
    global assessor
    
    if loading_status["loading"]:
        return jsonify({"error": "Knowledge graph is still loading. Please wait."})
    
    if not assessor:
        return jsonify({"error": "Knowledge graph not available."})
    
    data = request.get_json()
    drug_input = data.get('drugs', '')
    
    if not drug_input:
        return jsonify({"error": "Please enter drug names."})
    
    # Parse drugs
    drugs = [d.strip().lower() for d in drug_input.replace(';', ',').split(',') if d.strip()]
    
    if len(drugs) < 2:
        return jsonify({"error": "Please enter at least 2 drugs to check interactions."})
    
    try:
        result = assessor.assess_regimen(drugs)
        
        interactions = []
        for ddi in result.ddi_pairs:
            interactions.append({
                "drug1": ddi.get('drug1', 'Unknown'),
                "drug2": ddi.get('drug2', 'Unknown'),
                "severity": ddi.get('severity', 'Unknown'),
                "description": ddi.get('description', '')[:200] if ddi.get('description') else ''
            })
        
        return jsonify({
            "drugs": drugs,
            "risk_level": result.risk_level,
            "risk_score": result.overall_risk_score,
            "interactions": interactions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})


def load_knowledge_graph():
    """Load knowledge graph"""
    global assessor, loading_status
    
    if not KG_AVAILABLE:
        loading_status = {"loading": False, "message": "Error: Knowledge graph module not available"}
        return
    
    try:
        loading_status = {"loading": True, "message": "Loading knowledge graph..."}
        loader = KnowledgeGraphLoader()
        kg = loader.load_from_drugbank_csv()
        assessor = PolypharmacyRiskAssessor(kg)
        loading_status = {"loading": False, "message": f"Ready - {len(kg.drugs)} drugs, {len(kg.interactions)} interactions"}
    except Exception as e:
        loading_status = {"loading": False, "message": f"Error: {str(e)[:50]}"}


if __name__ == '__main__':
    # Load KG in background
    thread = threading.Thread(target=load_knowledge_graph, daemon=True)
    thread.start()
    
    print("\n" + "="*50)
    print("  DDI Analyzer Web GUI")
    print("="*50)
    print("\n  Open your browser to: http://localhost:5000\n")
    print("  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
