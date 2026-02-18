#!/usr/bin/env python3
"""
DDI Risk Analysis - Gradio Application
Clean, reliable GUI with progress indicators and LLM synthesis
"""

import gradio as gr
import pandas as pd
import time
import json
import os
from datetime import datetime

# ============================================================
# Knowledge Graph Loader
# ============================================================

class DrugDatabase:
    """Load and query the DDI knowledge graph"""
    
    def __init__(self):
        self.drugs = {}
        self.ddis = []
        self.side_effects = {}
        self.proteins = {}
        self.loaded = False
        
        # Drug name aliases
        self.aliases = {
            'aspirin': 'acetylsalicylic acid',
            'tylenol': 'acetaminophen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'coumadin': 'warfarin',
            'lipitor': 'atorvastatin',
            'zocor': 'simvastatin',
            'plavix': 'clopidogrel',
            'nexium': 'esomeprazole',
            'prilosec': 'omeprazole',
            'zoloft': 'sertraline',
            'prozac': 'fluoxetine',
            'xanax': 'alprazolam',
            'valium': 'diazepam',
            'ambien': 'zolpidem',
            'viagra': 'sildenafil',
            'cialis': 'tadalafil',
            'metformin': 'metformin',
            'lisinopril': 'lisinopril',
        }
    
    def load(self, progress_callback=None):
        """Load all data from CSV files"""
        base_path = "knowledge_graph_fact_based/neo4j_export"
        
        if progress_callback:
            progress_callback(0.1, "Loading drug database...")
        
        # Load drugs
        drugs_file = f"{base_path}/drugs.csv"
        if os.path.exists(drugs_file):
            df = pd.read_csv(drugs_file, low_memory=False)
            for _, row in df.iterrows():
                name = str(row.get('name', '')).lower().strip()
                if name:
                    self.drugs[name] = {
                        'name': row.get('name', ''),
                        'drugbank_id': row.get('drugbank_id', ''),
                        'type': row.get('type', ''),
                        'groups': row.get('groups', ''),
                        'atc_codes': row.get('atc_codes', ''),
                        'indication': row.get('indication', ''),
                        'mechanism_of_action': row.get('mechanism_of_action', ''),
                        'pharmacodynamics': row.get('pharmacodynamics', ''),
                        'molecular_weight': row.get('molecular_weight', ''),
                    }
        
        if progress_callback:
            progress_callback(0.3, f"Loaded {len(self.drugs):,} drugs...")
        
        # Load DDIs
        ddi_file = f"{base_path}/ddi_edges.csv"
        if os.path.exists(ddi_file):
            df = pd.read_csv(ddi_file, low_memory=False)
            self.ddis = df.to_dict('records')
        
        if progress_callback:
            progress_callback(0.5, f"Loaded {len(self.ddis):,} interactions...")
        
        # Load side effects
        se_file = f"{base_path}/side_effect_edges.csv"
        if os.path.exists(se_file):
            df = pd.read_csv(se_file, low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                se_name = row.get('side_effect_name', row.get('umls_name', ''))
                if drug_id not in self.side_effects:
                    self.side_effects[drug_id] = []
                self.side_effects[drug_id].append({
                    'name': se_name,
                    'umls_cui': row.get('umls_cui', ''),
                })
        
        if progress_callback:
            progress_callback(0.7, "Loading protein targets...")
        
        # Load proteins
        protein_file = f"{base_path}/drug_protein_edges.csv"
        if os.path.exists(protein_file):
            df = pd.read_csv(protein_file, low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                if drug_id not in self.proteins:
                    self.proteins[drug_id] = []
                self.proteins[drug_id].append({
                    'protein_id': row.get('protein_id', ''),
                    'protein_name': row.get('protein_name', ''),
                    'gene_name': row.get('gene_name', ''),
                    'action': row.get('action', ''),
                })
        
        if progress_callback:
            progress_callback(0.9, "Finalizing...")
        
        self.loaded = True
        
        if progress_callback:
            progress_callback(1.0, f"✅ Ready: {len(self.drugs):,} drugs, {len(self.ddis):,} DDIs")
        
        return f"Loaded {len(self.drugs):,} drugs, {len(self.ddis):,} DDIs"
    
    def resolve_drug(self, name):
        """Resolve drug name including aliases"""
        name_lower = name.lower().strip()
        
        # Check alias
        if name_lower in self.aliases:
            name_lower = self.aliases[name_lower]
        
        # Direct lookup
        if name_lower in self.drugs:
            return self.drugs[name_lower]
        
        # Partial match
        for drug_name, drug_data in self.drugs.items():
            if name_lower in drug_name or drug_name in name_lower:
                return drug_data
        
        return None
    
    def find_interactions(self, drug_ids):
        """Find all interactions between given drugs"""
        interactions = []
        drug_id_set = set(drug_ids)
        
        for ddi in self.ddis:
            d1 = ddi.get('drug1_id', ddi.get('source', ''))
            d2 = ddi.get('drug2_id', ddi.get('target', ''))
            
            if d1 in drug_id_set and d2 in drug_id_set:
                interactions.append({
                    'drug1': ddi.get('drug1_name', d1),
                    'drug2': ddi.get('drug2_name', d2),
                    'severity': ddi.get('severity', 'Unknown'),
                    'description': ddi.get('description', ''),
                })
        
        return interactions
    
    def get_side_effects(self, drug_id):
        """Get side effects for a drug"""
        return self.side_effects.get(drug_id, [])
    
    def get_proteins(self, drug_id):
        """Get protein targets for a drug"""
        return self.proteins.get(drug_id, [])


# ============================================================
# LLM Client (Ollama)
# ============================================================

class OllamaClient:
    """Simple Ollama client for LLM synthesis"""
    
    def __init__(self, model="meditron:7b-q4_K_M"):
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def generate(self, prompt, max_tokens=2000):
        """Generate text using Ollama"""
        import urllib.request
        import urllib.error
        
        try:
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3}
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('response', '')
        
        except Exception as e:
            return f"[LLM unavailable: {str(e)[:50]}]"


# ============================================================
# Global instances
# ============================================================

db = DrugDatabase()
llm = OllamaClient()


# ============================================================
# Main Analysis Functions
# ============================================================

def analyze_drugs(drug_input, progress=gr.Progress()):
    """
    Main analysis function with progress tracking
    """
    if not drug_input or not drug_input.strip():
        return "⚠️ Please enter drug names separated by commas", "", ""
    
    # Parse drug names
    drug_names = [d.strip() for d in drug_input.replace('\n', ',').split(',') if d.strip()]
    
    if len(drug_names) < 2:
        return "⚠️ Please enter at least 2 drugs to check for interactions", "", ""
    
    # ========== Stage 1: Load Knowledge Graph ==========
    progress(0.05, desc="🔄 Connecting to Knowledge Graph...")
    time.sleep(0.3)
    
    if not db.loaded:
        progress(0.1, desc="📊 Loading drug database...")
        db.load(lambda p, m: progress(0.1 + p * 0.2, desc=m))
    
    # ========== Stage 2: Resolve Drug Names ==========
    progress(0.35, desc="🔍 Resolving drug names...")
    time.sleep(0.2)
    
    resolved_drugs = []
    not_found = []
    
    for name in drug_names:
        drug = db.resolve_drug(name)
        if drug:
            resolved_drugs.append(drug)
        else:
            not_found.append(name)
    
    if not resolved_drugs:
        return f"⚠️ No drugs found in database. Tried: {', '.join(drug_names)}", "", ""
    
    # ========== Stage 3: Query Interactions ==========
    progress(0.45, desc="⚡ Querying drug-drug interactions...")
    time.sleep(0.3)
    
    drug_ids = [d['drugbank_id'] for d in resolved_drugs if d.get('drugbank_id')]
    interactions = db.find_interactions(drug_ids)
    
    # ========== Stage 4: Calculate Risk Score ==========
    progress(0.55, desc="📈 Calculating risk score...")
    time.sleep(0.2)
    
    risk_score = 0.0
    severity_counts = {'contraindicated': 0, 'major': 0, 'moderate': 0, 'minor': 0}
    
    for i in interactions:
        sev = i['severity'].lower()
        if 'contraindicated' in sev:
            risk_score += 0.5
            severity_counts['contraindicated'] += 1
        elif 'major' in sev:
            risk_score += 0.3
            severity_counts['major'] += 1
        elif 'moderate' in sev:
            risk_score += 0.15
            severity_counts['moderate'] += 1
        else:
            risk_score += 0.05
            severity_counts['minor'] += 1
    
    risk_score = min(risk_score, 1.0)
    risk_level = "CRITICAL" if risk_score >= 0.8 else "HIGH" if risk_score >= 0.5 else "MODERATE" if risk_score >= 0.2 else "LOW"
    
    # ========== Stage 5: Gather Side Effects ==========
    progress(0.65, desc="💊 Analyzing shared side effects...")
    time.sleep(0.2)
    
    all_side_effects = {}
    for drug in resolved_drugs:
        drug_id = drug.get('drugbank_id', '')
        for se in db.get_side_effects(drug_id):
            se_name = se['name']
            if se_name not in all_side_effects:
                all_side_effects[se_name] = []
            all_side_effects[se_name].append(drug['name'])
    
    shared_side_effects = {k: v for k, v in all_side_effects.items() if len(v) > 1}
    
    # ========== Stage 6: Gather Protein Targets ==========
    progress(0.75, desc="🧬 Analyzing protein targets...")
    time.sleep(0.2)
    
    all_proteins = {}
    for drug in resolved_drugs:
        drug_id = drug.get('drugbank_id', '')
        for p in db.get_proteins(drug_id):
            p_name = p['protein_name']
            if p_name and p_name not in all_proteins:
                all_proteins[p_name] = {'drugs': [], 'gene': p['gene_name']}
            if p_name:
                all_proteins[p_name]['drugs'].append(drug['name'])
    
    shared_proteins = {k: v for k, v in all_proteins.items() if len(v['drugs']) > 1}
    
    # ========== Stage 7: Generate LLM Synthesis ==========
    progress(0.85, desc="🤖 Synthesizing report with AI...")
    
    llm_report = generate_llm_synthesis(resolved_drugs, interactions, shared_side_effects, shared_proteins, risk_level, risk_score)
    
    # ========== Stage 8: Format Output ==========
    progress(0.95, desc="📋 Formatting report...")
    time.sleep(0.2)
    
    # Build summary
    summary = build_summary(resolved_drugs, not_found, interactions, risk_level, risk_score, severity_counts)
    
    # Build detailed report
    detailed = build_detailed_report(resolved_drugs, interactions, shared_side_effects, shared_proteins)
    
    progress(1.0, desc="✅ Analysis complete!")
    
    return summary, detailed, llm_report


def generate_llm_synthesis(drugs, interactions, side_effects, proteins, risk_level, risk_score):
    """Generate LLM synthesis of the analysis"""
    
    drug_names = [d['name'] for d in drugs]
    
    prompt = f"""You are a clinical pharmacist reviewing a drug combination for potential risks.

DRUGS BEING ANALYZED: {', '.join(drug_names)}

RISK ASSESSMENT: {risk_level} (Score: {risk_score:.2f})

DRUG-DRUG INTERACTIONS FOUND ({len(interactions)}):
"""
    
    for i in interactions[:5]:
        prompt += f"- {i['drug1']} + {i['drug2']}: {i['severity']} - {i['description'][:150]}\n"
    
    prompt += f"""
SHARED SIDE EFFECTS ({len(side_effects)} effects overlap):
{', '.join(list(side_effects.keys())[:10])}

SHARED PROTEIN TARGETS ({len(proteins)}):
{', '.join(list(proteins.keys())[:5])}

Please provide:
1. A brief clinical summary of the main risks
2. Key monitoring recommendations
3. Any safer alternatives to consider

Keep the response concise and clinically focused."""

    return llm.generate(prompt)


def build_summary(drugs, not_found, interactions, risk_level, risk_score, severity_counts):
    """Build the summary section"""
    
    # Risk color
    risk_colors = {
        'CRITICAL': '🔴',
        'HIGH': '🟠', 
        'MODERATE': '🟡',
        'LOW': '🟢'
    }
    
    summary = f"""
## {risk_colors.get(risk_level, '⚪')} Risk Assessment: **{risk_level}** (Score: {risk_score:.2f})

### 💊 Drugs Analyzed ({len(drugs)})
"""
    
    for drug in drugs:
        db_id = drug.get('drugbank_id', 'N/A')
        summary += f"- **{drug['name']}** ([{db_id}](https://go.drugbank.com/drugs/{db_id}))\n"
    
    if not_found:
        summary += f"\n⚠️ Not found in database: {', '.join(not_found)}\n"
    
    summary += f"""
### ⚠️ Interactions Found ({len(interactions)})
"""
    
    if severity_counts['contraindicated'] > 0:
        summary += f"- 🔴 Contraindicated: {severity_counts['contraindicated']}\n"
    if severity_counts['major'] > 0:
        summary += f"- 🟠 Major: {severity_counts['major']}\n"
    if severity_counts['moderate'] > 0:
        summary += f"- 🟡 Moderate: {severity_counts['moderate']}\n"
    if severity_counts['minor'] > 0:
        summary += f"- 🟢 Minor: {severity_counts['minor']}\n"
    
    if not interactions:
        summary += "✅ No significant interactions detected\n"
    
    return summary


def build_detailed_report(drugs, interactions, side_effects, proteins):
    """Build detailed report section"""
    
    report = "## 📋 Detailed Analysis\n\n"
    
    # Drug profiles
    report += "### 💊 Drug Profiles\n\n"
    for drug in drugs:
        db_id = drug.get('drugbank_id', '')
        report += f"#### {drug['name']}\n"
        report += f"- **DrugBank ID**: [{db_id}](https://go.drugbank.com/drugs/{db_id})\n"
        report += f"- **Type**: {drug.get('type', 'N/A')}\n"
        report += f"- **Groups**: {drug.get('groups', 'N/A')}\n"
        
        indication = drug.get('indication', '')
        if indication:
            report += f"- **Indication**: {indication[:300]}...\n"
        
        mechanism = drug.get('mechanism_of_action', '')
        if mechanism:
            report += f"- **Mechanism**: {mechanism[:300]}...\n"
        
        report += "\n"
    
    # Interactions
    if interactions:
        report += "### ⚠️ Drug-Drug Interactions\n\n"
        for i in interactions:
            sev_emoji = '🔴' if 'contraindicated' in i['severity'].lower() or 'major' in i['severity'].lower() else '🟡'
            report += f"**{sev_emoji} {i['drug1']} ↔ {i['drug2']}**\n"
            report += f"- Severity: {i['severity']}\n"
            report += f"- {i['description']}\n"
            report += f"- Source: DrugBank DDI Database\n\n"
    
    # Shared side effects
    if side_effects:
        report += "### 🔬 Shared Side Effects (may be amplified)\n\n"
        for se_name, drugs_list in list(side_effects.items())[:15]:
            report += f"- **{se_name}**: {', '.join(drugs_list)}\n"
        report += "\n*Source: SIDER Database*\n\n"
    
    # Shared proteins
    if proteins:
        report += "### 🧬 Shared Protein Targets\n\n"
        for p_name, p_data in list(proteins.items())[:10]:
            report += f"- **{p_name}** ({p_data['gene']}): {', '.join(p_data['drugs'])}\n"
        report += "\n*Source: DrugBank/UniProt*\n\n"
    
    # Metadata
    report += f"\n---\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    report += "*Data sources: DrugBank, SIDER, UniProt*\n"
    
    return report


# ============================================================
# Chat Function
# ============================================================

# Store last analysis context for chat
analysis_context = {"drugs": [], "interactions": [], "risk_level": "", "report": ""}

def chat_with_llm(message, history, model_choice):
    """Chat with LLM about the drug analysis"""
    
    if not message or not message.strip():
        return history, ""
    
    # Build context from last analysis
    context = ""
    if analysis_context["drugs"]:
        context = f"""
Context from previous drug analysis:
- Drugs analyzed: {', '.join(analysis_context['drugs'])}
- Risk level: {analysis_context['risk_level']}
- Number of interactions: {len(analysis_context['interactions'])}

Key interactions found:
"""
        for i in analysis_context['interactions'][:5]:
            context += f"- {i.get('drug1', '')} + {i.get('drug2', '')}: {i.get('severity', '')} - {i.get('description', '')[:100]}\n"
    
    # Select model
    model_map = {
        "Meditron 7B (Medical)": "meditron:7b-q4_K_M",
        "MedLlama2 7B (Medical)": "medllama2:7b-q4_K_M",
        "Llama 3 8B (General)": "llama3:latest",
        "Mistral 7B (Fast)": "mistral:7b-instruct-q4_K_M"
    }
    selected_model = model_map.get(model_choice, "meditron:7b-q4_K_M")
    
    # Build conversation history for context
    conv_history = ""
    for h in history[-3:]:  # Last 3 exchanges
        conv_history += f"User: {h[0]}\nAssistant: {h[1]}\n"
    
    prompt = f"""You are an expert clinical pharmacist assistant. Help the user understand drug interactions and safety.

{context}

Previous conversation:
{conv_history}

User question: {message}

Provide a helpful, accurate response. If discussing drug risks, be specific about severity and monitoring needs.
If you don't have information about something, say so clearly."""

    # Generate response
    llm.model = selected_model
    response = llm.generate(prompt, max_tokens=1500)
    
    # Update history
    history.append((message, response))
    
    return history, ""


# ============================================================
# Gradio Interface
# ============================================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="DDI Risk Analyzer",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .risk-critical { background-color: #7f1d1d !important; color: white !important; }
        .risk-high { background-color: #dc2626 !important; color: white !important; }
        .risk-moderate { background-color: #f59e0b !important; color: white !important; }
        .risk-low { background-color: #059669 !important; color: white !important; }
        .chatbot { min-height: 400px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 💊 DDI Risk Analysis System
        
        **Drug-Drug Interaction Analysis with Knowledge Graph & AI Synthesis**
        
        Enter drug names below to analyze potential interactions, shared side effects, and protein targets.
        The system uses a comprehensive knowledge graph with data from DrugBank, SIDER, and UniProt.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                drug_input = gr.Textbox(
                    label="Enter Drug Names",
                    placeholder="warfarin, aspirin, ibuprofen",
                    lines=2,
                    info="Separate multiple drugs with commas"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Drugs", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### Quick Examples")
                with gr.Row():
                    ex1 = gr.Button("🔴 High Risk", size="sm")
                    ex2 = gr.Button("🟠 CV Combo", size="sm")
                    ex3 = gr.Button("🟡 Serotonin", size="sm")
                    ex4 = gr.Button("🟢 Low Risk", size="sm")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(
                    label="Risk Summary",
                    value="*Enter drugs above and click 'Analyze Drugs'*"
                )
        
        with gr.Tabs():
            with gr.TabItem("📋 Detailed Report"):
                detailed_output = gr.Markdown(value="")
            
            with gr.TabItem("🤖 AI Synthesis"):
                llm_output = gr.Markdown(value="")
            
            with gr.TabItem("💬 Chat with AI"):
                gr.Markdown("""
                ### Ask Follow-up Questions
                Chat with the AI to learn more about the drug interactions, ask for alternatives, 
                or get more detailed explanations.
                """)
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "Meditron 7B (Medical)",
                            "MedLlama2 7B (Medical)", 
                            "Llama 3 8B (General)",
                            "Mistral 7B (Fast)"
                        ],
                        value="Meditron 7B (Medical)",
                        label="Select LLM Model",
                        scale=1
                    )
                
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    show_label=False
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask about the drug interactions, alternatives, monitoring...",
                        label="Your Question",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Markdown("**Suggested questions:**")
                with gr.Row():
                    q1 = gr.Button("What are safer alternatives?", size="sm")
                    q2 = gr.Button("What should be monitored?", size="sm")
                    q3 = gr.Button("Explain the mechanism", size="sm")
                    q4 = gr.Button("When is this combination used?", size="sm")
        
        # Store state
        chat_history = gr.State([])
        
        # Analysis function wrapper to store context
        def analyze_and_store(drug_input, progress=gr.Progress()):
            summary, detailed, llm_report = analyze_drugs(drug_input, progress)
            
            # Store context for chat
            drug_names = [d.strip() for d in drug_input.replace('\n', ',').split(',') if d.strip()]
            analysis_context["drugs"] = drug_names
            analysis_context["risk_level"] = "See summary"
            analysis_context["report"] = detailed
            
            # Extract interactions from the analysis
            drug_ids = []
            for name in drug_names:
                drug = db.resolve_drug(name)
                if drug:
                    drug_ids.append(drug.get('drugbank_id', ''))
            analysis_context["interactions"] = db.find_interactions(drug_ids)
            
            return summary, detailed, llm_report
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_and_store,
            inputs=[drug_input],
            outputs=[summary_output, detailed_output, llm_output],
            show_progress="full"
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Enter drugs above and click 'Analyze Drugs'*", "", "", []),
            outputs=[drug_input, summary_output, detailed_output, llm_output, chat_history]
        )
        
        # Chat handlers
        send_btn.click(
            fn=chat_with_llm,
            inputs=[chat_input, chatbot, model_dropdown],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            fn=chat_with_llm,
            inputs=[chat_input, chatbot, model_dropdown],
            outputs=[chatbot, chat_input]
        )
        
        # Suggested question buttons
        q1.click(fn=lambda: "What are safer alternatives to this drug combination?", outputs=chat_input)
        q2.click(fn=lambda: "What symptoms or lab values should be monitored with these drugs?", outputs=chat_input)
        q3.click(fn=lambda: "Can you explain the pharmacological mechanism of these interactions?", outputs=chat_input)
        q4.click(fn=lambda: "Are there clinical situations where this combination might be appropriate?", outputs=chat_input)
        
        # Example buttons
        ex1.click(fn=lambda: "warfarin, aspirin, ibuprofen", outputs=drug_input)
        ex2.click(fn=lambda: "metformin, lisinopril, amlodipine, metoprolol", outputs=drug_input)
        ex3.click(fn=lambda: "sertraline, tramadol, ondansetron", outputs=drug_input)
        ex4.click(fn=lambda: "acetaminophen, omeprazole", outputs=drug_input)
        
        gr.Markdown("""
        ---
        **Data Sources:**
        - [DrugBank](https://go.drugbank.com/) - Drug information and interactions
        - [SIDER](http://sideeffects.embl.de/) - Side effect data
        - [UniProt](https://www.uniprot.org/) - Protein targets
        
        **LLM Models (via Ollama):**
        - Meditron 7B - Medical/clinical focused
        - MedLlama2 7B - Medical-tuned LLaMA
        - Llama 3 8B - Strong general reasoning
        - Mistral 7B - Fast inference
        
        *This tool is for educational purposes only. Always consult healthcare professionals.*
        """)
    
    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DDI Risk Analysis - Gradio Application")
    print("="*60)
    
    # Pre-load database
    print("\n📊 Pre-loading knowledge graph...")
    db.load(lambda p, m: print(f"   {m}"))
    
    print("\n🚀 Starting Gradio interface...")
    print("   The browser should open automatically.")
    print("   If not, go to: http://127.0.0.1:7860")
    print("\n   Press Ctrl+C to stop\n")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
