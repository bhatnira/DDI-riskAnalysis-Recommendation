#!/usr/bin/env python3
"""
DDI Analyzer GUI with Natural Language Chat
Combines drug interaction checking with conversational AI interface

Run with: python ddi_chat_gui.py
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

# Import the DDI analysis components
try:
    from kg_polypharmacy_risk import PolypharmacyRiskAssessor, KnowledgeGraphLoader
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

# Global state
assessor = None
kg_loader = None
loading_status = {"loading": True, "message": "Initializing..."}
chat_history = []

# LLM Configuration (Local Only - Ollama with Quantized Models)
LLM_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "default_model": "meditron:7b-q4_K_M",
    "temperature": 0.7,
    "max_tokens": 1024
}

# Available models (quantized for efficiency)
AVAILABLE_MODELS = {
    "meditron:7b-q4_K_M": {
        "name": "Meditron 7B (Medical)",
        "description": "Medical/clinical focus, trained on medical literature",
        "params": "7B",
        "size": "4.1 GB"
    },
    "medllama2:7b-q4_K_M": {
        "name": "MedLlama2 7B (Medical)",
        "description": "Medical-tuned LLaMA for healthcare applications",
        "params": "7B",
        "size": "4.1 GB"
    },
    "llama3:latest": {
        "name": "Llama 3 8B (General)",
        "description": "Strong general model with good reasoning",
        "params": "8B",
        "size": "4.7 GB"
    },
    "mistral:7b-instruct-q4_K_M": {
        "name": "Mistral 7B Instruct",
        "description": "Fast instruction-following, good for Q&A",
        "params": "7B",
        "size": "4.4 GB"
    }
}

SYSTEM_PROMPT = """You are DDI Assistant, an expert AI specializing in drug-drug interactions and pharmaceutical safety.

You have access to a comprehensive knowledge graph containing:
- Drug information and properties
- Drug-drug interactions with severity levels (Contraindicated, Major, Moderate, Minor)
- Side effects and adverse reactions
- Protein targets and mechanisms

When responding:
1. Be accurate and evidence-based about drug interactions
2. Explain severity levels clearly (Contraindicated > Major > Moderate > Minor)
3. Provide practical recommendations when asked
4. Always recommend consulting healthcare providers for final decisions
5. If you detect drug names in the query, mention relevant interactions from the knowledge graph

Keep responses concise but informative."""

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDI Assistant</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,-apple-system,sans-serif;background:#f5f5f5;min-height:100vh;display:flex;flex-direction:column}
        .header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:15px 20px;text-align:center}
        .header h1{font-size:22px;margin-bottom:4px}
        .header p{font-size:13px;opacity:0.9}
        .container{display:flex;flex:1;max-width:1400px;margin:0 auto;width:100%;gap:20px;padding:20px}
        .panel{background:#fff;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,0.1);overflow:hidden}
        .panel-header{background:#f8f9fa;padding:12px 15px;border-bottom:1px solid #e0e0e0;font-weight:600;font-size:14px}
        
        /* Quick Check Panel */
        .quick-check{width:380px;flex-shrink:0}
        .quick-content{padding:15px}
        .quick-content label{display:block;font-size:13px;font-weight:600;margin-bottom:6px;color:#444}
        .quick-content input{width:100%;padding:10px 12px;border:2px solid #e0e0e0;border-radius:8px;font-size:14px;margin-bottom:10px}
        .quick-content input:focus{outline:none;border-color:#667eea}
        .hint{color:#888;font-size:11px;margin-bottom:12px}
        .btn{padding:10px 18px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;user-select:none;-webkit-tap-highlight-color:transparent}
        .btn:hover{transform:translateY(-1px);opacity:0.9}
        .btn:active{transform:translateY(1px);opacity:0.8}
        .btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 2px 8px rgba(102,126,234,0.3)}
        .btn-primary:hover{box-shadow:0 4px 12px rgba(102,126,234,0.4)}
        .btn-secondary{background:#f0f0f0;color:#333;margin-left:8px}
        .btn-secondary:hover{background:#e0e0e0}
        .btn-ex{background:#e8f4f8;color:#2196F3;padding:6px 12px;font-size:11px;margin:3px;display:inline-block}
        .btn-ex:hover{background:#2196F3;color:#fff}
        .examples{margin-top:15px;padding-top:15px;border-top:1px solid #eee}
        .examples span{font-size:11px;color:#666}
        #quick-results{margin-top:15px;max-height:400px;overflow-y:auto}
        .risk-badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:11px;font-weight:700;text-transform:uppercase}
        .risk-high{background:#ffebee;color:#c62828}
        .risk-moderate{background:#fff3e0;color:#ef6c00}
        .risk-low{background:#e8f5e9;color:#2e7d32}
        .int-item{padding:10px;margin:8px 0;border-radius:8px;border-left:3px solid;font-size:13px}
        .int-item.contraindicated{background:#ffebee;border-color:#c62828}
        .int-item.major{background:#fff3e0;border-color:#ef6c00}
        .int-item.moderate{background:#fffde7;border-color:#f9a825}
        .int-item.minor{background:#e8f5e9;border-color:#2e7d32}
        .int-drugs{font-weight:600;margin-bottom:3px}
        .int-sev{font-size:10px;font-weight:700;text-transform:uppercase;color:#666}
        
        /* Chat Panel */
        .chat-panel{flex:1;display:flex;flex-direction:column;min-width:400px}
        .chat-messages{flex:1;padding:15px;overflow-y:auto;max-height:calc(100vh - 280px);min-height:400px}
        .message{margin-bottom:15px;display:flex}
        .message.user{justify-content:flex-end}
        .message.assistant{justify-content:flex-start}
        .message-content{max-width:80%;padding:12px 16px;border-radius:16px;font-size:14px;line-height:1.5}
        .user .message-content{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-bottom-right-radius:4px}
        .assistant .message-content{background:#f0f0f0;color:#333;border-bottom-left-radius:4px}
        .message-content pre{background:rgba(0,0,0,0.1);padding:8px;border-radius:6px;margin:8px 0;overflow-x:auto;font-size:12px}
        .message-content code{font-family:monospace}
        .chat-input{padding:15px;border-top:1px solid #e0e0e0;display:flex;gap:10px}
        .chat-input input{flex:1;padding:12px 15px;border:2px solid #e0e0e0;border-radius:24px;font-size:14px}
        .chat-input input:focus{outline:none;border-color:#667eea}
        .chat-input button{padding:12px 24px;border-radius:24px}
        .typing{display:flex;align-items:center;gap:4px;padding:10px}
        .typing span{width:8px;height:8px;background:#667eea;border-radius:50%;animation:bounce 1.4s infinite ease-in-out}
        .typing span:nth-child(1){animation-delay:-0.32s}
        .typing span:nth-child(2){animation-delay:-0.16s}
        @keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
        .status{text-align:center;padding:8px;font-size:11px;color:#666;background:#f8f9fa}
        .status-ready{color:#2e7d32}
        .status-loading{color:#ef6c00}
        .welcome{text-align:center;padding:40px 20px;color:#888}
        .welcome h3{color:#333;margin-bottom:10px}
        .suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:15px}
        .suggestion{background:#f0f0f0;padding:8px 14px;border-radius:16px;font-size:12px;cursor:pointer;transition:.2s}
        .suggestion:hover{background:#667eea;color:#fff}
        @media(max-width:900px){.container{flex-direction:column}.quick-check{width:100%}}
    </style>
</head>
<body>
<div class="header">
    <h1>💊 DDI Assistant</h1>
    <p>Drug-Drug Interaction Analysis & Conversational AI</p>
</div>

<div class="container">
    <!-- Quick Check Panel -->
    <div class="panel quick-check">
        <div class="panel-header">⚡ Quick Interaction Check</div>
        <div class="quick-content">
            <label>Drug Names</label>
            <input type="text" id="drugs" placeholder="warfarin, aspirin, ibuprofen">
            <p class="hint">Separate multiple drugs with commas</p>
            <button type="button" class="btn btn-primary" onclick="quickCheck()">Check Interactions</button>
            <button type="button" class="btn btn-secondary" onclick="clearQuick()">Clear</button>
            
            <div class="examples">
                <span>Examples:</span><br>
                <button type="button" class="btn btn-ex" onclick="setDrugs('warfarin, aspirin, ibuprofen')">High Risk</button>
                <button type="button" class="btn btn-ex" onclick="setDrugs('metformin, lisinopril, amlodipine')">Moderate</button>
                <button type="button" class="btn btn-ex" onclick="setDrugs('acetaminophen, omeprazole')">Low Risk</button>
            </div>
            
            <div id="quick-results"></div>
        </div>
    </div>
    
    <!-- Chat Panel -->
    <div class="panel chat-panel">
        <div class="panel-header" style="display:flex;justify-content:space-between;align-items:center">
            <span>💬 Ask DDI Assistant</span>
            <button type="button" class="btn" style="padding:4px 10px;font-size:11px;background:#e8f4f8" onclick="showModelSelector()" id="model-btn" title="Change LLM Model">🤖 Model</button>
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="welcome">
                <h3>Welcome to DDI Assistant!</h3>
                <p>Ask me anything about drug interactions, side effects, and medication safety.</p>
                <div class="suggestions">
                    <div class="suggestion" onclick="askQuestion('What are the risks of taking warfarin with aspirin?')">Warfarin + Aspirin risks?</div>
                    <div class="suggestion" onclick="askQuestion('What are safer alternatives to ibuprofen for someone on blood thinners?')">Alternatives to ibuprofen?</div>
                    <div class="suggestion" onclick="askQuestion('Explain the severity levels of drug interactions')">Severity levels explained</div>
                    <div class="suggestion" onclick="askQuestion('What should I know about mixing antidepressants?')">Antidepressant interactions</div>
                </div>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="chat-input" placeholder="Ask about drug interactions, safety, alternatives..." onkeypress="if(event.key==='Enter')sendChat()">
            <button type="button" class="btn btn-primary" onclick="sendChat()">Send</button>
        </div>
    </div>
</div>

<div class="status" id="status">Connecting...</div>

<!-- Model Selection Modal -->
<div id="model-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:1000;justify-content:center;align-items:center">
    <div style="background:#fff;border-radius:12px;padding:20px;max-width:500px;width:90%;max-height:80vh;overflow-y:auto">
        <h3 style="margin-bottom:15px">Select LLM Model</h3>
        <div id="model-list"></div>
        <div style="margin-top:15px;text-align:right">
            <button class="btn btn-secondary" onclick="closeModelModal()">Cancel</button>
        </div>
    </div>
</div>
<script>
let chatHistory = [];

window.onload = function() {
    console.log('DDI Assistant loaded');
    checkStatus();
    document.getElementById('drugs').addEventListener('keypress', function(e) { 
        if(e.key==='Enter') { e.preventDefault(); quickCheck(); }
    });
};

let currentModel = '';
let availableModels = {};

function checkStatus() {
    fetch('/status')
        .then(r=>r.json())
        .then(d=>{
            const s = document.getElementById('status');
            if(d.loading) {
                s.className = 'status status-loading';
                s.textContent = '⏳ ' + d.message;
                setTimeout(checkStatus, 1000);
            } else {
                s.className = 'status status-ready';
                currentModel = d.current_model || '';
                availableModels = d.available_models || {};
                const modelInfo = availableModels[currentModel] || {};
                const modelName = modelInfo.name || d.llm || 'template';
                s.textContent = '✅ ' + d.message + ' | 🤖 ' + modelName;
                document.getElementById('model-btn').textContent = '🤖 ' + (modelName.split(' ')[0] || 'Model');
            }
        })
        .catch(e => console.error('Status check failed:', e));
}

function showModelSelector() {
    const modal = document.getElementById('model-modal');
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    
    for(const [key, info] of Object.entries(availableModels)) {
        const isActive = key === currentModel;
        const div = document.createElement('div');
        div.style = 'padding:12px;margin:8px 0;border-radius:8px;cursor:pointer;border:2px solid ' + (isActive ? '#667eea' : '#e0e0e0') + ';background:' + (isActive ? '#f0f4ff' : '#fff');
        div.innerHTML = '<div style="font-weight:600;margin-bottom:4px">' + (isActive ? '✓ ' : '') + info.name + '</div>' +
                        '<div style="font-size:12px;color:#666">' + info.description + '</div>' +
                        '<div style="font-size:11px;color:#888;margin-top:4px">Parameters: ' + info.params + ' | Size: ' + info.size + '</div>';
        div.onclick = () => selectModel(key);
        list.appendChild(div);
    }
    
    modal.style.display = 'flex';
}

function closeModelModal() {
    document.getElementById('model-modal').style.display = 'none';
}

function selectModel(modelKey) {
    fetch('/set_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: modelKey})
    })
    .then(r => r.json())
    .then(d => {
        if(d.success) {
            currentModel = modelKey;
            closeModelModal();
            checkStatus();
            addMessage('assistant', '🔄 Switched to ' + (availableModels[modelKey]?.name || modelKey));
        } else {
            alert('Error: ' + d.error);
        }
    })
    .catch(e => alert('Error switching model: ' + e));
}

function setDrugs(drugs) {
    console.log('Setting drugs:', drugs);
    document.getElementById('drugs').value = drugs;
    quickCheck();
}

function clearQuick() {
    document.getElementById('drugs').value = '';
    document.getElementById('quick-results').innerHTML = '';
}

function quickCheck() {
    console.log('quickCheck called');
    const drugs = document.getElementById('drugs').value.trim();
    console.log('Drugs input:', drugs);
    if(!drugs) { alert('Enter drug names'); return; }
    
    const results = document.getElementById('quick-results');
    results.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    
    fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({drugs: drugs})
    })
    .then(r => {
        console.log('Response status:', r.status);
        return r.json();
    })
    .then(d => {
        console.log('Response data:', d);
        if(d.error) { results.innerHTML = '<p style="color:#c62828">⚠️ ' + d.error + '</p>'; return; }
        showQuickResults(d);
    })
    .catch(e => { 
        console.error('Fetch error:', e);
        results.innerHTML = '<p style="color:#c62828">⚠️ ' + e + '</p>'; 
    });
}

function showQuickResults(d) {
    const results = document.getElementById('quick-results');
    const rc = d.risk_level.toLowerCase();
    const cls = rc==='high'||rc==='critical' ? 'risk-high' : rc==='moderate' ? 'risk-moderate' : 'risk-low';
    
    let h = '<div style="margin-top:15px;padding:12px;background:#f8f9fa;border-radius:8px">';
    h += '<span class="risk-badge ' + cls + '">' + d.risk_level + '</span>';
    h += '<span style="float:right;font-size:20px;font-weight:700">' + d.risk_score.toFixed(2) + '</span>';
    h += '<p style="margin-top:8px;font-size:12px;color:#666">Drugs: ' + d.drugs.join(', ') + '</p></div>';
    
    if(d.interactions.length === 0) {
        h += '<p style="text-align:center;padding:20px;color:#2e7d32">✅ No significant interactions</p>';
    } else {
        h += '<p style="margin:12px 0 8px;font-size:12px;font-weight:600">Interactions (' + d.interactions.length + '):</p>';
        for(const i of d.interactions) {
            const sc = i.severity.toLowerCase().replace(' ','-');
            h += '<div class="int-item ' + sc + '"><div class="int-drugs">' + i.drug1 + ' ↔ ' + i.drug2 + '</div>';
            h += '<div class="int-sev">' + i.severity + '</div></div>';
        }
    }
    results.innerHTML = h;
}

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
    
    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'message assistant';
    typing.id = 'typing-indicator';
    typing.innerHTML = '<div class="message-content"><div class="typing"><span></span><span></span><span></span></div></div>';
    document.getElementById('chat-messages').appendChild(typing);
    scrollChat();
    
    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, history: chatHistory.slice(-10)})
    })
    .then(r => r.json())
    .then(d => {
        document.getElementById('typing-indicator')?.remove();
        if(d.error) {
            addMessage('assistant', '⚠️ ' + d.error);
        } else {
            addMessage('assistant', d.response);
            chatHistory.push({role: 'assistant', content: d.response});
        }
    })
    .catch(e => {
        document.getElementById('typing-indicator')?.remove();
        addMessage('assistant', '⚠️ Error: ' + e.message);
    });
}

function addMessage(role, content) {
    const messages = document.getElementById('chat-messages');
    // Remove welcome message if present
    const welcome = messages.querySelector('.welcome');
    if(welcome) welcome.remove();
    
    const div = document.createElement('div');
    div.className = 'message ' + role;
    
    // Simple markdown-like formatting
    let formatted = content
        .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\\n/g, '<br>');
    
    div.innerHTML = '<div class="message-content">' + formatted + '</div>';
    messages.appendChild(div);
    scrollChat();
}

function scrollChat() {
    const messages = document.getElementById('chat-messages');
    messages.scrollTop = messages.scrollHeight;
}
</script>
</body>
</html>"""


class LLMClient:
    """LLM client with multi-model support (quantized models)"""
    
    def __init__(self):
        self.backend = None
        self.model_name = "template"
        self.current_model = None
        self.installed_models = {}
        self._detect_backend()
    
    def _detect_backend(self):
        # Try Ollama first
        try:
            response = requests.get(f"{LLM_CONFIG['ollama_url']}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                installed = {m['name']: m for m in models}
                self.installed_models = installed
                
                if models:
                    # Find best available model (prefer medical models)
                    model = None
                    for preferred in ['meditron', 'medllama2', 'llama3', 'mistral']:
                        for m_name in installed.keys():
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
        
        # Fallback to template (no Ollama available)
        self.backend = 'template'
        self.model_name = 'template'
    
    def set_model(self, model_key: str) -> bool:
        """Switch to a different model"""
        if model_key in self.installed_models or model_key in AVAILABLE_MODELS:
            self.current_model = model_key
            self.model_name = AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)
            return True
        return False
    
    def get_available_models(self) -> Dict:
        """Get dict of available models with their info"""
        available = {}
        for model_key, info in AVAILABLE_MODELS.items():
            if model_key in self.installed_models:
                available[model_key] = info
        return available
    
    def generate(self, messages: List[Dict], context: str = "") -> str:
        if self.backend == 'ollama':
            return self._generate_ollama(messages)
        else:
            return self._generate_template(messages, context)
    
    def _generate_ollama(self, messages: List[Dict]) -> str:
        try:
            response = requests.post(
                f"{LLM_CONFIG['ollama_url']}/api/chat",
                json={
                    "model": self.current_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": LLM_CONFIG['temperature']}
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '[No response]')
        except Exception as e:
            return f"[Ollama error: {str(e)[:50]}]"
        return "[Ollama request failed]"
    
    def _generate_template(self, messages: List[Dict], context: str = "") -> str:
        """Template-based responses when no LLM available"""
        if not messages:
            return "I'm here to help with drug interaction questions."
        
        user_msg = messages[-1].get('content', '').lower()
        
        # Check for drug names and provide KG-based response
        if context:
            return f"""Based on the knowledge graph analysis:

{context}

**Recommendations:**
- Review each interaction with your healthcare provider
- Consider timing of doses to minimize interactions
- Monitor for symptoms mentioned in interaction descriptions

Always consult healthcare providers for medical decisions."""
        
        # General responses
        if 'severity' in user_msg or 'level' in user_msg:
            return """**Drug Interaction Severity Levels:**

• **Contraindicated**: Should never be combined - life-threatening risk
• **Major**: High risk of serious adverse effects - avoid combination
• **Moderate**: May worsen conditions or require monitoring
• **Minor**: Limited clinical significance - usually safe

The severity guides clinical decision-making but individual factors matter. Always consult healthcare providers."""
        
        if 'alternative' in user_msg:
            return """When looking for drug alternatives, consider:

1. **Same drug class**: Similar mechanism but different interaction profile
2. **Different class**: Alternative approach to treating the condition
3. **Non-pharmacological**: Lifestyle changes where appropriate

Use the quick check panel to analyze specific drug combinations. For personalized alternatives, consult a pharmacist or physician."""
        
        if any(w in user_msg for w in ['warfarin', 'blood thinner', 'anticoagulant']):
            return """**Warfarin Interaction Information:**

Warfarin has many significant interactions due to its narrow therapeutic window:

• **High Risk**: NSAIDs (ibuprofen, aspirin), antibiotics, antifungals
• **Moderate Risk**: Acetaminophen (high doses), many herbal supplements
• **Requires Monitoring**: Diet changes (vitamin K), alcohol

Use the quick check to analyze specific combinations with warfarin."""
        
        if any(w in user_msg for w in ['metformin', 'diabetes', 'blood sugar']):
            return """**Metformin Interaction Information:**

Metformin is generally well-tolerated but watch for:

• **Contrast dye**: May cause kidney issues - stop metformin before procedures
• **Alcohol**: Increases lactic acidosis risk
• **ACE inhibitors**: May affect kidney function

Use the quick check to analyze specific combinations."""

        if any(w in user_msg for w in ['statin', 'cholesterol', 'atorvastatin', 'simvastatin']):
            return """**Statin Interaction Information:**

Statins have important interactions to consider:

• **Grapefruit juice**: Increases statin levels significantly
• **Fibrates**: Increased muscle damage risk (rhabdomyolysis)
• **Certain antibiotics**: Clarithromycin, erythromycin increase levels
• **Amlodipine**: May increase simvastatin levels

Use the quick check for specific combinations."""
        
        if any(w in user_msg for w in ['nsaid', 'ibuprofen', 'naproxen', 'pain', 'anti-inflammatory']):
            return """**NSAID Interaction Information:**

NSAIDs (ibuprofen, naproxen, etc.) interact with many medications:

• **Blood thinners**: Increased bleeding risk (warfarin, aspirin)
• **ACE inhibitors/ARBs**: Reduced blood pressure effect, kidney risk
• **Diuretics**: Reduced effectiveness
• **SSRIs**: Increased bleeding risk

Acetaminophen is often a safer alternative for pain."""
        
        return """I can help with:

• **Drug interactions**: Enter drug names in your message or use the quick check panel
• **Severity explanations**: Ask about severity levels
• **Specific drugs**: Ask about warfarin, statins, NSAIDs, metformin, etc.

**Try asking:**
- "What are the risks of warfarin and aspirin?"
- "Is metformin safe with lisinopril?"
- "Explain interaction severity levels"

Enter at least 2 drug names to see interactions from our knowledge graph."""


# Initialize LLM client
llm_client = None


def extract_drugs_from_text(text: str) -> List[str]:
    """Extract potential drug names from text"""
    if not kg_loader:
        return []
    
    text_lower = text.lower()
    found_drugs = []
    
    # Check against known drug names
    for drug_name in kg_loader.drug_name_to_id.keys():
        if drug_name in text_lower and len(drug_name) > 3:
            found_drugs.append(drug_name)
    
    # Also check common drug name patterns
    common_drugs = ['aspirin', 'ibuprofen', 'warfarin', 'metformin', 'lisinopril', 
                    'amlodipine', 'metoprolol', 'omeprazole', 'acetaminophen', 'tylenol',
                    'advil', 'motrin', 'coumadin', 'plavix', 'xarelto', 'eliquis']
    for drug in common_drugs:
        if drug in text_lower and drug not in found_drugs:
            found_drugs.append(drug)
    
    return list(set(found_drugs))


def get_kg_context(drugs: List[str]) -> str:
    """Get knowledge graph context for drugs"""
    if not assessor or len(drugs) < 2:
        return ""
    
    try:
        result = assessor.assess_polypharmacy_risk(drugs)
        
        context_parts = [f"**Analysis for: {', '.join(drugs)}**"]
        context_parts.append(f"Risk Level: **{result.risk_level}** (Score: {result.overall_risk_score:.2f})")
        
        if result.ddi_pairs:
            context_parts.append(f"\n**Found {len(result.ddi_pairs)} interactions:**")
            for ddi in result.ddi_pairs[:5]:  # Limit to 5
                context_parts.append(f"• {ddi.get('drug1')} + {ddi.get('drug2')}: {ddi.get('severity')}")
        else:
            context_parts.append("\nNo significant interactions found in the knowledge graph.")
        
        return "\n".join(context_parts)
    except:
        return ""


class DDIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler"""
    
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
        self.end_headers()
        self.wfile.write(response)
    
    def do_GET(self):
        if self.path == '/':
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
        else:
            self.send_error(404)
    
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
        elif self.path == '/chat':
            result = handle_chat(data.get('message', ''), data.get('history', []))
            self._send_json(result)
        elif self.path == '/set_model':
            result = set_model(data.get('model', ''))
            self._send_json(result)
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


def handle_chat(message: str, history: List[Dict]) -> dict:
    """Handle chat message"""
    global llm_client
    
    if not message:
        return {"error": "Empty message"}
    
    if not llm_client:
        return {"error": "Chat not initialized"}
    
    # Extract drugs from message and get KG context
    drugs = extract_drugs_from_text(message)
    kg_context = get_kg_context(drugs) if len(drugs) >= 2 else ""
    
    # Build messages for LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history
    for h in history[-8:]:
        messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
    
    # Add context if we have drug info
    if kg_context:
        messages.append({
            "role": "user", 
            "content": f"{message}\n\n[Knowledge Graph Context:\n{kg_context}]"
        })
    else:
        messages.append({"role": "user", "content": message})
    
    try:
        response = llm_client.generate(messages, context=kg_context)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


def set_model(model_key: str) -> dict:
    """Switch LLM model"""
    global llm_client
    
    if not llm_client:
        return {"success": False, "error": "LLM client not initialized"}
    
    if not model_key:
        return {"success": False, "error": "No model specified"}
    
    if llm_client.set_model(model_key):
        return {"success": True, "model": model_key, "name": llm_client.model_name}
    else:
        return {"success": False, "error": f"Model '{model_key}' not available"}


def load_knowledge_graph():
    """Load knowledge graph in background"""
    global assessor, kg_loader, loading_status, llm_client
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    if not KG_AVAILABLE:
        loading_status = {"loading": False, "message": "KG module not found"}
        return
    
    try:
        loading_status = {"loading": True, "message": "Loading knowledge graph..."}
        kg_loader = KnowledgeGraphLoader()
        kg_loader.load()
        assessor = PolypharmacyRiskAssessor(kg_loader)
        loading_status = {"loading": False, "message": f"{len(kg_loader.drugs):,} drugs, {len(kg_loader.ddis):,} DDIs"}
    except Exception as e:
        loading_status = {"loading": False, "message": f"Error: {str(e)[:40]}"}


def main():
    PORT = 8080
    
    # Load KG in background
    thread = threading.Thread(target=load_knowledge_graph, daemon=True)
    thread.start()
    
    print("\n" + "=" * 55)
    print("  DDI Assistant - Chat & Analysis Interface")
    print("=" * 55)
    print(f"\n  🌐 Open: http://localhost:{PORT}")
    print("\n  Features:")
    print("    • Quick drug interaction check")
    print("    • Natural language chat about DDIs")
    print("    • Knowledge graph-powered responses")
    print("\n  Local LLM (Ollama):")
    print("    Install: curl -fsSL https://ollama.com/install.sh | sh")
    print("    Run: ollama run llama3  (or mistral, phi, etc.)")
    print("\n  Press Ctrl+C to stop\n")
    
    with socketserver.TCPServer(("", PORT), DDIHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
