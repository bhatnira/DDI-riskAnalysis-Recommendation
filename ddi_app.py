#!/usr/bin/env python3
"""
DDI Risk Analysis Application with Alternative Recommendations
Three-column design:
  COL1: Drug input (list, narrative, or image)
  COL2: Analysis report with severity, risk & safer alternatives
  COL3: Chat with selected LLM to learn more
"""

import gradio as gr
import pandas as pd
import time
import json
import os
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# Optional: Image processing for prescription photos
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ============================================================
# Knowledge Graph Database with Alternative Recommendations
# ============================================================

class KnowledgeGraph:
    """Drug-Drug Interaction Knowledge Graph with Alternative Drug Finder"""
    
    def __init__(self):
        self.drugs = {}
        self.drugs_by_id = {}
        self.ddis = []
        self.ddi_index = {}  # {(drug1_id, drug2_id): ddi_info}
        self.side_effects = {}
        self.proteins = {}
        self.atc_index = defaultdict(list)  # {atc_prefix: [drug_ids]}
        self.loaded = False
        
        # Common drug name aliases (brand -> generic)
        self.aliases = {
            'aspirin': 'acetylsalicylic acid', 'tylenol': 'acetaminophen',
            'advil': 'ibuprofen', 'motrin': 'ibuprofen', 'coumadin': 'warfarin',
            'lipitor': 'atorvastatin', 'zocor': 'simvastatin', 'plavix': 'clopidogrel',
            'nexium': 'esomeprazole', 'prilosec': 'omeprazole', 'zoloft': 'sertraline',
            'prozac': 'fluoxetine', 'xanax': 'alprazolam', 'valium': 'diazepam',
            # ACE inhibitors
            'cardace': 'ramipril', 'altace': 'ramipril', 'tritace': 'ramipril',
            'vasotec': 'enalapril', 'zestril': 'lisinopril', 'prinivil': 'lisinopril',
            'lotensin': 'benazepril', 'capoten': 'captopril',
            # Beta blockers
            'lopressor': 'metoprolol', 'toprol': 'metoprolol', 'tenormin': 'atenolol',
            'coreg': 'carvedilol', 'inderal': 'propranolol',
            # Calcium channel blockers
            'norvasc': 'amlodipine', 'cardizem': 'diltiazem', 'calan': 'verapamil',
            # Diuretics
            'lasix': 'furosemide', 'bumex': 'bumetanide', 'demadex': 'torsemide',
            # Statins
            'crestor': 'rosuvastatin', 'lescol': 'fluvastatin', 'pravachol': 'pravastatin',
            # Anticoagulants
            'eliquis': 'apixaban', 'xarelto': 'rivaroxaban', 'pradaxa': 'dabigatran',
            # Diabetes
            'glucophage': 'metformin', 'januvia': 'sitagliptin', 'jardiance': 'empagliflozin',
            # Pain
            'celebrex': 'celecoxib', 'voltaren': 'diclofenac', 'aleve': 'naproxen',
            'ultram': 'tramadol', 'vicodin': 'hydrocodone', 'percocet': 'oxycodone',
            # Antibiotics
            'augmentin': 'amoxicillin', 'zithromax': 'azithromycin', 'cipro': 'ciprofloxacin',
            # GI
            'pepcid': 'famotidine', 'zantac': 'ranitidine', 'prevacid': 'lansoprazole',
        }
        
        # Severity weights for risk calculation
        self.severity_weights = {
            'contraindicated': 1.0,
            'major': 0.7,
            'moderate': 0.4,
            'minor': 0.15,
            'unknown': 0.2
        }
        
        # Drug name list for fuzzy matching
        self.drug_names = []
    
    def load(self):
        """Load knowledge graph from CSV files"""
        base = "knowledge_graph_fact_based/neo4j_export"
        
        # Load drugs
        if os.path.exists(f"{base}/drugs.csv"):
            df = pd.read_csv(f"{base}/drugs.csv", low_memory=False)
            for _, row in df.iterrows():
                name = str(row.get('name', '')).lower().strip()
                drug_id = row.get('drugbank_id', '')
                if name and drug_id:
                    drug_data = dict(row)
                    self.drugs[name] = drug_data
                    self.drugs_by_id[drug_id] = drug_data
                    
                    # Build ATC index for finding alternatives
                    atc = str(row.get('atc_codes', ''))
                    if atc and atc != 'nan':
                        for code in atc.split('|'):
                            if len(code) >= 4:
                                # Index by first 4 chars (therapeutic subgroup)
                                prefix = code[:4]
                                self.atc_index[prefix].append(drug_id)
                                # Also index by first 5 chars (chemical subgroup)
                                if len(code) >= 5:
                                    prefix5 = code[:5]
                                    self.atc_index[prefix5].append(drug_id)
        
        # Load DDIs and build index
        if os.path.exists(f"{base}/ddi_edges.csv"):
            df = pd.read_csv(f"{base}/ddi_edges.csv", low_memory=False)
            self.ddis = df.to_dict('records')
            
            # Build DDI lookup index
            for ddi in self.ddis:
                d1 = ddi.get('drug1_id', ddi.get('source', ''))
                d2 = ddi.get('drug2_id', ddi.get('target', ''))
                if d1 and d2:
                    # Store both directions for easy lookup
                    self.ddi_index[(d1, d2)] = ddi
                    self.ddi_index[(d2, d1)] = ddi
        
        # Load side effects
        if os.path.exists(f"{base}/side_effect_edges.csv"):
            df = pd.read_csv(f"{base}/side_effect_edges.csv", low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                if drug_id not in self.side_effects:
                    self.side_effects[drug_id] = []
                self.side_effects[drug_id].append({
                    'name': row.get('side_effect_name', row.get('umls_name', '')),
                    'umls_cui': row.get('umls_cui', ''),
                })
        
        # Load proteins
        if os.path.exists(f"{base}/drug_protein_edges.csv"):
            df = pd.read_csv(f"{base}/drug_protein_edges.csv", low_memory=False)
            for _, row in df.iterrows():
                drug_id = row.get('drug_id', '')
                if drug_id not in self.proteins:
                    self.proteins[drug_id] = []
                self.proteins[drug_id].append({
                    'name': row.get('protein_name', ''),
                    'gene': row.get('gene_name', ''),
                })
        
        # Build list of drug names for fuzzy matching
        self.drug_names = list(self.drugs.keys())
        
        self.loaded = True
        return f"Loaded {len(self.drugs):,} drugs, {len(self.ddis):,} DDIs, {len(self.atc_index)} ATC groups"
    
    def parse_drug_input(self, raw_input):
        """Parse drug input with various separators: +, comma, newline, 'and'"""
        # Normalize separators
        text = raw_input.lower()
        # Replace various separators with comma
        text = re.sub(r'\s*\+\s*', ',', text)  # + separator
        text = re.sub(r'\s+and\s+', ',', text)  # "and" separator
        text = re.sub(r'\s*[;|]\s*', ',', text)  # semicolon, pipe
        text = re.sub(r'\n+', ',', text)  # newlines
        
        # Split and clean
        drugs = [d.strip() for d in text.split(',') if d.strip()]
        # Remove duplicates while preserving order
        seen = set()
        unique_drugs = []
        for d in drugs:
            if d not in seen:
                seen.add(d)
                unique_drugs.append(d)
        return unique_drugs
    
    def fuzzy_match(self, query, threshold=0.6):
        """Find closest drug name matches using string similarity"""
        query = query.lower().strip()
        
        # First check exact match
        if query in self.drugs:
            return [(query, 1.0, self.drugs[query])]
        
        # Check aliases
        if query in self.aliases:
            alias_name = self.aliases[query]
            if alias_name in self.drugs:
                return [(alias_name, 1.0, self.drugs[alias_name])]
        
        # Check partial match
        for name in self.drug_names:
            if query in name or name in query:
                return [(name, 0.95, self.drugs[name])]
        
        # Fuzzy match using SequenceMatcher
        matches = []
        for name in self.drug_names:
            ratio = SequenceMatcher(None, query, name).ratio()
            if ratio >= threshold:
                matches.append((name, ratio, self.drugs[name]))
        
        # Sort by similarity
        matches.sort(key=lambda x: -x[1])
        return matches[:5]  # Return top 5 matches
    
    def identify_drugs(self, raw_input):
        """
        Parse input and identify drugs with fuzzy matching.
        Returns: (found_drugs, suggestions, not_found)
        """
        parsed_names = self.parse_drug_input(raw_input)
        
        found_drugs = []  # List of (input_name, matched_name, drug_data)
        suggestions = {}  # {input_name: [(match_name, score, drug_data), ...]}
        not_found = []    # Names with no good matches
        
        for name in parsed_names:
            matches = self.fuzzy_match(name)
            
            if not matches:
                not_found.append(name)
            elif matches[0][1] >= 0.95:  # High confidence match
                found_drugs.append((name, matches[0][0], matches[0][2]))
            elif matches[0][1] >= 0.6:   # Possible matches - need confirmation
                suggestions[name] = matches
            else:
                not_found.append(name)
        
        return found_drugs, suggestions, not_found
    
    def resolve(self, name):
        """Resolve drug name to database entry"""
        n = name.lower().strip()
        if n in self.aliases:
            n = self.aliases[n]
        if n in self.drugs:
            return self.drugs[n]
        for k, v in self.drugs.items():
            if n in k or k in n:
                return v
        return None
    
    def get_severity_weight(self, severity_str):
        """Get numerical weight for severity string"""
        sev = severity_str.lower() if severity_str else ''
        for key, weight in self.severity_weights.items():
            if key in sev:
                return weight
        return 0.2  # default unknown
    
    def get_interaction(self, drug1_id, drug2_id):
        """Get interaction between two drugs"""
        return self.ddi_index.get((drug1_id, drug2_id))
    
    def get_interactions(self, drug_ids):
        """Find all interactions between a set of drugs"""
        interactions = []
        id_list = list(drug_ids)
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                ddi = self.get_interaction(id_list[i], id_list[j])
                if ddi:
                    interactions.append(ddi)
        return interactions
    
    def calculate_risk_score(self, drug_ids):
        """Calculate polypharmacy risk score for a set of drugs"""
        interactions = self.get_interactions(drug_ids)
        if not interactions:
            return 0.0, [], {}
        
        score = 0.0
        counts = {'contraindicated': 0, 'major': 0, 'moderate': 0, 'minor': 0}
        
        for i in interactions:
            sev = i.get('severity', '').lower()
            weight = self.get_severity_weight(sev)
            score += weight
            
            for key in counts:
                if key in sev:
                    counts[key] += 1
                    break
        
        # Normalize score (max 1.0)
        score = min(score / max(len(interactions), 1), 1.0)
        return score, interactions, counts
    
    def find_alternatives(self, drug_id, other_drug_ids, max_alternatives=5):
        """Find alternative drugs with lower interaction risk"""
        drug_data = self.drugs_by_id.get(drug_id, {})
        if not drug_data:
            return []
        
        atc = str(drug_data.get('atc_codes', ''))
        if not atc or atc == 'nan':
            return []
        
        # Get ATC prefixes for this drug
        prefixes = []
        for code in atc.split('|'):
            if len(code) >= 5:
                prefixes.append(code[:5])
            if len(code) >= 4:
                prefixes.append(code[:4])
        
        # Find candidate alternatives from same therapeutic class
        candidates = set()
        for prefix in prefixes:
            for cand_id in self.atc_index.get(prefix, []):
                if cand_id != drug_id and cand_id not in other_drug_ids:
                    candidates.add(cand_id)
        
        if not candidates:
            return []
        
        # Calculate risk for each alternative
        original_risk, _, _ = self.calculate_risk_score(set(other_drug_ids) | {drug_id})
        
        alternatives = []
        for cand_id in candidates:
            cand_data = self.drugs_by_id.get(cand_id, {})
            if not cand_data:
                continue
            
            # Calculate risk with this alternative
            test_ids = set(other_drug_ids) | {cand_id}
            alt_risk, alt_interactions, _ = self.calculate_risk_score(test_ids)
            
            # Get specific interaction with each other drug
            interactions_with_others = []
            for other_id in other_drug_ids:
                ddi = self.get_interaction(cand_id, other_id)
                if ddi:
                    interactions_with_others.append(ddi)
            
            if alt_risk < original_risk:
                alternatives.append({
                    'drug_id': cand_id,
                    'name': cand_data.get('name', ''),
                    'drugbank_id': cand_id,
                    'atc_codes': cand_data.get('atc_codes', ''),
                    'original_risk': original_risk,
                    'alternative_risk': alt_risk,
                    'risk_reduction': original_risk - alt_risk,
                    'num_interactions': len(interactions_with_others),
                    'interactions': interactions_with_others
                })
        
        # Sort by risk reduction (best first)
        alternatives.sort(key=lambda x: -x['risk_reduction'])
        return alternatives[:max_alternatives]


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """Ollama LLM client"""
    
    MODELS = {
        "Meditron 7B (Medical)": "meditron:7b-q4_K_M",
        "MedLlama2 7B (Medical)": "medllama2:7b-q4_K_M",
        "Llama 3 8B (General)": "llama3:latest",
        "Mistral 7B (Fast)": "mistral:7b-instruct-q4_K_M"
    }
    
    def generate(self, prompt, model_name="Mistral 7B (Fast)"):
        import urllib.request
        model = self.MODELS.get(model_name, "mistral:7b-instruct-q4_K_M")
        try:
            data = json.dumps({
                "model": model, "prompt": prompt, "stream": False,
                "options": {"num_predict": 1500, "temperature": 0.3}
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate", data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode()).get('response', '')
        except Exception as e:
            return f"[LLM Error: {str(e)[:50]}]"


# ============================================================
# CONVERSATIONAL AI ASSISTANT - Natural ChatGPT-like Experience
# ============================================================

class ConversationMemory:
    """
    Stores conversation context and analysis reports for natural dialogue
    """
    def __init__(self):
        self.report = ""  # Full analysis report
        self.drugs = []
        self.risk_level = ""
        self.interactions = []
        self.alternatives = {}
        self.conversation_history = []  # [(role, message), ...]
        self.max_history = 10  # Keep last N exchanges
        
    def update_from_analysis(self, report, drugs, risk, interactions, alternatives):
        """Store analysis results in memory"""
        self.report = report
        self.drugs = drugs
        self.risk_level = risk
        self.interactions = interactions
        self.alternatives = alternatives
        
    def add_message(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append((role, content))
        # Trim to max history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_history_text(self):
        """Get formatted conversation history"""
        if not self.conversation_history:
            return ""
        
        history = []
        for role, content in self.conversation_history[-8:]:  # Last 4 exchanges
            if role == "user":
                history.append(f"User: {content}")
            else:
                history.append(f"Assistant: {content}")
        return "\n".join(history)
    
    def clear(self):
        """Clear memory for new session"""
        self.report = ""
        self.drugs = []
        self.risk_level = ""
        self.interactions = []
        self.alternatives = {}
        self.conversation_history = []


class NaturalChatAssistant:
    """
    Knowledge Graph-First Conversational Assistant
    
    Prioritizes data from the knowledge graph and only uses LLM's 
    own knowledge when KG data is insufficient.
    """
    
    SYSTEM_PROMPT = """You are a clinical pharmacology assistant powered by a DrugBank Knowledge Graph.

CRITICAL: Your answers must be based on the KNOWLEDGE GRAPH DATA provided below.
- ALWAYS cite specific data from the knowledge graph when available
- When referencing interactions, quote the exact description from the KG
- Mention DrugBank IDs when discussing specific drugs
- If data is NOT in your knowledge context, clearly state: "This information is not in my knowledge graph, but based on general pharmacology..."

Your style:
- Warm and professional, like a knowledgeable pharmacist
- Educational without being condescending
- Use markdown formatting (bold, bullet points) for clarity
- Keep responses focused and evidence-based

For greetings/casual chat: respond briefly and naturally.
For drug questions: provide detailed KG-sourced information."""

    def __init__(self, knowledge_graph, llm_client):
        self.kg = knowledge_graph
        self.llm = llm_client
        self.memory = ConversationMemory()
        
    def update_memory(self, report, drugs, risk, interactions, alternatives):
        """Update conversation memory with new analysis"""
        self.memory.update_from_analysis(report, drugs, risk, interactions, alternatives)
    
    def extract_drug_from_query(self, message):
        """Extract drug names mentioned in the user's query"""
        msg_lower = message.lower()
        mentioned_drugs = []
        
        # Check analyzed drugs first
        for drug in self.memory.drugs:
            if drug.lower() in msg_lower:
                mentioned_drugs.append(drug)
        
        # Check knowledge graph for any other drug mentions
        for drug_name in self.kg.drugs.keys():
            if drug_name in msg_lower and drug_name not in [d.lower() for d in mentioned_drugs]:
                mentioned_drugs.append(drug_name)
                if len(mentioned_drugs) >= 4:  # Limit to avoid huge context
                    break
        
        return mentioned_drugs
    
    def get_drug_details_from_kg(self, drug_name):
        """Extract comprehensive drug information from knowledge graph"""
        drug_data = self.kg.drugs.get(drug_name.lower(), {})
        if not drug_data:
            return None
        
        details = {
            'name': drug_name.title(),
            'drugbank_id': drug_data.get('drugbank_id', 'Unknown'),
            'description': str(drug_data.get('description', ''))[:500] if str(drug_data.get('description', '')) != 'nan' else '',
            'mechanism': str(drug_data.get('mechanism_of_action', ''))[:600] if str(drug_data.get('mechanism_of_action', '')) != 'nan' else '',
            'pharmacodynamics': str(drug_data.get('pharmacodynamics', ''))[:400] if str(drug_data.get('pharmacodynamics', '')) != 'nan' else '',
            'indication': str(drug_data.get('indication', ''))[:300] if str(drug_data.get('indication', '')) != 'nan' else '',
            'atc_codes': str(drug_data.get('atc_codes', '')) if str(drug_data.get('atc_codes', '')) != 'nan' else '',
            'half_life': str(drug_data.get('half_life', '')) if str(drug_data.get('half_life', '')) != 'nan' else '',
            'route': str(drug_data.get('route', '')) if str(drug_data.get('route', '')) != 'nan' else '',
        }
        
        # Get protein targets
        db_id = drug_data.get('drugbank_id', '')
        proteins = self.kg.proteins.get(db_id, [])[:5]
        if proteins:
            details['proteins'] = [{'name': p.get('name', ''), 'uniprot': p.get('uniprot_id', '')} for p in proteins if p.get('name')]
        
        # Get side effects
        side_effects = self.kg.side_effects.get(db_id, [])[:10]
        if side_effects:
            details['side_effects'] = [se.get('name', '') for se in side_effects if se.get('name')]
        
        return details
    
    def get_interaction_details(self, drug1, drug2):
        """Get specific interaction details from KG"""
        for inter in self.memory.interactions:
            d1 = inter.get('drug1_name', inter.get('drug1_id', '')).lower()
            d2 = inter.get('drug2_name', inter.get('drug2_id', '')).lower()
            if (drug1.lower() in [d1, d2]) and (drug2.lower() in [d1, d2]):
                return inter
        return None
        
    def build_knowledge_context(self, user_message=""):
        """Build comprehensive knowledge context from KG data"""
        context_parts = []
        
        # Extract drugs mentioned in query
        query_drugs = self.extract_drug_from_query(user_message)
        all_drugs = list(set(self.memory.drugs + query_drugs))
        
        if not all_drugs and not self.memory.drugs:
            context_parts.append("=== NO ANALYSIS YET ===")
            context_parts.append("User hasn't analyzed any drugs yet. Guide them to use the left panel.")
            return "\n".join(context_parts)
        
        # Add analysis summary
        if self.memory.report:
            context_parts.append("=== ANALYSIS SUMMARY ===")
            context_parts.append(f"Analyzed drugs: {', '.join(self.memory.drugs)}")
            context_parts.append(f"Risk Level: {self.memory.risk_level}")
            context_parts.append(f"Total interactions found: {len(self.memory.interactions)}")
        
        # Add detailed drug information from KG
        context_parts.append("\n=== KNOWLEDGE GRAPH: DRUG DATA ===")
        for drug_name in all_drugs[:5]:
            details = self.get_drug_details_from_kg(drug_name)
            if details:
                context_parts.append(f"\n### {details['name']} ({details['drugbank_id']})")
                
                if details.get('description'):
                    context_parts.append(f"**Description:** {details['description']}")
                
                if details.get('mechanism'):
                    context_parts.append(f"**Mechanism of Action:** {details['mechanism']}")
                
                if details.get('pharmacodynamics'):
                    context_parts.append(f"**Pharmacodynamics:** {details['pharmacodynamics']}")
                
                if details.get('indication'):
                    context_parts.append(f"**Indication:** {details['indication']}")
                
                if details.get('proteins'):
                    prot_str = ', '.join([f"{p['name']} ({p['uniprot']})" for p in details['proteins'][:3]])
                    context_parts.append(f"**Protein Targets:** {prot_str}")
                
                if details.get('side_effects'):
                    se_str = ', '.join(details['side_effects'][:8])
                    context_parts.append(f"**Known Side Effects:** {se_str}")
                
                if details.get('half_life'):
                    context_parts.append(f"**Half-life:** {details['half_life']}")
        
        # Add interaction details from KG
        if self.memory.interactions:
            context_parts.append("\n=== KNOWLEDGE GRAPH: INTERACTIONS ===")
            for inter in self.memory.interactions[:8]:
                d1 = inter.get('drug1_name', inter.get('drug1_id', ''))
                d2 = inter.get('drug2_name', inter.get('drug2_id', ''))
                sev = inter.get('severity', 'Unknown')
                desc = str(inter.get('description', ''))[:400]
                context_parts.append(f"\n**{d1} + {d2}** (Severity: {sev})")
                context_parts.append(f"Description: {desc}")
        
        # Add alternatives from KG
        if self.memory.alternatives:
            context_parts.append("\n=== KNOWLEDGE GRAPH: ALTERNATIVES ===")
            for drug, alts in list(self.memory.alternatives.items())[:4]:
                if alts:
                    context_parts.append(f"\n**Alternatives to {drug.title()}:**")
                    for alt in alts[:4]:
                        alt_name = alt['name']
                        reason = alt.get('reason', '')
                        context_parts.append(f"- {alt_name}: {reason}")
        
        return "\n".join(context_parts)
    
    def respond(self, user_message, model_name="Mistral 7B (Fast)"):
        """
        Generate a KG-informed response to the user's message
        """
        # Add user message to history
        self.memory.add_message("user", user_message)
        
        # Build comprehensive knowledge context
        knowledge = self.build_knowledge_context(user_message)
        history = self.memory.get_history_text()
        
        prompt = f"""{self.SYSTEM_PROMPT}

=== KNOWLEDGE GRAPH DATA ===
{knowledge}

=== CONVERSATION HISTORY ===
{history}

=== USER MESSAGE ===
{user_message}

Provide a helpful response. Base your answer on the KNOWLEDGE GRAPH DATA above. 
If the information isn't in the knowledge graph, state that clearly before adding general knowledge:"""

        # Generate response
        response = self.llm.generate(prompt, model_name)
        
        # Store assistant response in history
        self.memory.add_message("assistant", response)
        
        return response


# ============================================================
# Global instances & state
# ============================================================

kg = KnowledgeGraph()
llm = LLMClient()
chat_assistant = None  # Natural conversation assistant
current_analysis = {"drugs": [], "interactions": [], "risk": "", "report": "", "alternatives": {}}
identified_drugs = {"confirmed": [], "suggestions": {}, "not_found": []}

def get_chat_assistant():
    """Get or initialize natural chat assistant"""
    global chat_assistant
    if chat_assistant is None:
        if not kg.loaded:
            kg.load()
        chat_assistant = NaturalChatAssistant(kg, llm)
    return chat_assistant


# ============================================================
# Drug Identification with Fuzzy Matching
# ============================================================

def identify_drugs_preview(drug_input, progress=gr.Progress()):
    """
    Step 1: Parse input and identify drugs with fuzzy matching.
    Shows compact preview with tables for suggestions.
    """
    if not drug_input or not drug_input.strip():
        return "Please enter drug names", "", gr.update(visible=False)
    
    # Load KG if needed
    progress(0.2, desc="Loading Knowledge Graph...")
    if not kg.loaded:
        kg.load()
    
    progress(0.5, desc="Identifying drugs...")
    
    found_drugs, suggestions, not_found = kg.identify_drugs(drug_input)
    
    # Store for later use
    identified_drugs["confirmed"] = found_drugs
    identified_drugs["suggestions"] = suggestions
    identified_drugs["not_found"] = not_found
    
    total_found = len(found_drugs)
    total_issues = len(suggestions) + len(not_found)
    
    # === Build compact preview ===
    preview = "## Drug Identification\n\n"
    
    # Summary paragraph
    if found_drugs:
        matched_names = [f"**{d[1]}**" for d in found_drugs]
        preview += f"Successfully identified **{total_found}** drug(s): {', '.join(matched_names)}.\n\n"
    
    # Confirmed drugs table (if any name mappings occurred)
    name_changes = [(i, m, d) for i, m, d in found_drugs if i.lower() != m.lower()]
    if name_changes:
        preview += "| Your Input | Matched As | DrugBank |\n"
        preview += "|------------|------------|----------|\n"
        for inp, matched, data in name_changes:
            db_id = data.get('drugbank_id', '')
            preview += f"| {inp} | {matched} | [{db_id}](https://go.drugbank.com/drugs/{db_id}) |\n"
        preview += "\n"
    
    # Suggestions for misspelled names
    if suggestions:
        preview += "### Possible Matches (please verify)\n\n"
        preview += "The following terms were not found exactly. Did you mean:\n\n"
        preview += "| You Entered | Suggestion | Match % |\n"
        preview += "|-------------|------------|--------|\n"
        for input_name, matches in suggestions.items():
            top = matches[0]
            preview += f"| `{input_name}` | **{top[0]}** | {int(top[1]*100)}% |\n"
        preview += "\n*Edit your input with the correct spelling and click 'Identify' again.*\n\n"
    
    # Not found
    if not_found:
        preview += f"### Not Found: `{', '.join(not_found)}`\n\n"
        preview += "Try using generic drug names instead of brand names, or check spelling.\n\n"
    
    # Next step guidance
    preview += "---\n\n"
    if total_found >= 2:
        preview += f"**Ready!** {total_found} drugs confirmed. Click **'Generate Comprehensive Report'** to check interactions"
        if total_issues > 0:
            preview += f" (or fix the {total_issues} unmatched term(s) above first)"
        preview += ".\n"
    elif total_found == 1:
        preview += "Only 1 drug confirmed — enter at least 2 drugs to analyze interactions.\n"
    else:
        preview += "No drugs identified. Check spelling or try generic names (e.g., 'acetaminophen' instead of 'Tylenol').\n"
    
    # Build confirmed list for editing
    confirmed_list = ", ".join([d[1] for d in found_drugs])
    
    progress(1.0, desc="Done")
    
    # Show edit group if we found any drugs
    show_edit = len(found_drugs) > 0
    return preview, confirmed_list, gr.update(visible=show_edit)


# ============================================================
# Narrative & Image Drug Extraction
# ============================================================

def extract_drugs_from_narrative(narrative_text, progress=gr.Progress()):
    """
    Extract drug names from a natural language narrative.
    Example: "I take cardace 5mg in the morning and aspirin at night for my heart"
    Returns: comma-separated drug list
    """
    if not narrative_text or not narrative_text.strip():
        return "", "*Enter a description of your medications*"
    
    progress(0.2, desc="Loading drug database...")
    if not kg.loaded:
        kg.load()
    
    progress(0.4, desc="Analyzing text...")
    
    # Normalize text
    text = narrative_text.lower()
    
    # Common patterns to remove (dosages, frequencies, etc.)
    text = re.sub(r'\d+\s*(mg|ml|mcg|g|iu|units?)\b', ' ', text)  # Remove dosages
    text = re.sub(r'\b(once|twice|thrice|\d+\s*times?)\s*(daily|a\s*day|per\s*day|weekly)?\b', ' ', text)
    text = re.sub(r'\b(morning|evening|night|afternoon|bedtime|before|after|with)\s*(meals?|food|breakfast|lunch|dinner)?\b', ' ', text)
    text = re.sub(r'\b(tablet|capsule|pill|dose|dosage)s?\b', ' ', text)
    
    # Extract potential drug words (2+ char words)
    words = re.findall(r'\b[a-z]{2,}\b', text)
    
    # Also try 2-word combinations (for drugs like "folic acid")
    words_list = text.split()
    bigrams = [' '.join(words_list[i:i+2]) for i in range(len(words_list)-1)]
    
    # Try to match each word/bigram against database
    found_drugs = []
    matched_terms = set()
    
    # Check bigrams first (longer matches preferred)
    for phrase in bigrams:
        phrase = phrase.strip()
        if len(phrase) < 3 or phrase in matched_terms:
            continue
        
        # Check exact match first
        if phrase in kg.aliases:
            found_drugs.append(kg.aliases[phrase])
            matched_terms.add(phrase)
            continue
            
        # Check if it's a known drug name
        for drug_name in kg.drug_names:
            if phrase == drug_name.lower():
                found_drugs.append(drug_name)
                matched_terms.add(phrase)
                break
        else:
            # Fuzzy match if close enough
            for drug_name in kg.drug_names[:500]:  # Check top drugs
                if SequenceMatcher(None, phrase, drug_name.lower()).ratio() > 0.85:
                    found_drugs.append(drug_name)
                    matched_terms.add(phrase)
                    break
    
    # Then check single words
    for word in words:
        if word in matched_terms or len(word) < 3:
            continue
        
        # Check aliases
        if word in kg.aliases:
            found_drugs.append(kg.aliases[word])
            matched_terms.add(word)
            continue
        
        # Exact match check
        for drug_name in kg.drug_names:
            if word == drug_name.lower():
                found_drugs.append(drug_name)
                matched_terms.add(word)
                break
        else:
            # Higher threshold for single words to avoid false positives
            for drug_name in kg.drug_names[:500]:
                if len(word) >= 4 and SequenceMatcher(None, word, drug_name.lower()).ratio() > 0.88:
                    found_drugs.append(drug_name)
                    matched_terms.add(word)
                    break
    
    progress(1.0, desc="Done")
    
    # Remove duplicates preserving order
    seen = set()
    unique_drugs = []
    for d in found_drugs:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique_drugs.append(d)
    
    if unique_drugs:
        drug_list = ", ".join(unique_drugs)
        status = f"**Found {len(unique_drugs)} drug(s):** {drug_list}\n\n*Review and click 'Check Drugs' to validate*"
        return drug_list, status
    else:
        return "", "*No drugs detected in the text. Try including drug names like 'aspirin', 'metoprolol', 'cardace', etc.*"


def extract_drugs_from_image(image, progress=gr.Progress()):
    """
    Extract drug names from an uploaded image using OCR.
    Works with photos of prescription labels, medication bottles, etc.
    """
    if image is None:
        return "", "*Upload an image of your prescription or medication*"
    
    if not HAS_OCR:
        return "", "**OCR not available.** Install pytesseract: `pip install pytesseract pillow` and ensure tesseract is installed on your system."
    
    progress(0.2, desc="Processing image...")
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        progress(0.5, desc="Extracting text with OCR...")
        
        # Run OCR
        extracted_text = pytesseract.image_to_string(image)
        
        progress(0.7, desc="Finding drug names...")
        
        if not extracted_text.strip():
            return "", "*Could not extract text from image. Try a clearer photo.*"
        
        # Use the narrative extractor on OCR text
        drug_list, status = extract_drugs_from_narrative(extracted_text, progress=progress)
        
        if drug_list:
            status = f"**OCR Text:** _{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}_\n\n" + status
        else:
            status = f"**OCR Text:** _{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}_\n\n*No drugs detected. The image may be unclear or not contain drug names.*"
        
        return drug_list, status
        
    except Exception as e:
        return "", f"**Error processing image:** {str(e)}"


# ============================================================
# Knowledge Graph Analysis with Alternatives
# ============================================================

def analyze_ddi(drug_input, progress=gr.Progress()):
    """Analyze drug interactions and find safer alternatives"""
    
    if not drug_input or not drug_input.strip():
        return (
            "Please enter drug names separated by commas, + signs, or newlines\n\n**Examples:**\n- `warfarin, aspirin, ibuprofen`\n- `Warfarin + Aspirin`\n- `sildenafil + nitroglycerin`",
            gr.update(visible=False), gr.update(), gr.update(), ""
        )
    
    # Load KG if needed
    progress(0.1, desc="Loading Knowledge Graph...")
    if not kg.loaded:
        kg.load()
    
    # Parse drugs with smart separator handling
    progress(0.2, desc="Identifying drugs...")
    found_drugs, suggestions, not_found = kg.identify_drugs(drug_input)
    
    if len(found_drugs) < 2:
        # Not enough drugs - show preview instead
        preview = "**Need at least 2 identified drugs for interaction analysis.**\n\n"
        preview += "---\n\n"
        if found_drugs:
            preview += f"Found: {', '.join([d[1] for d in found_drugs])}\n\n"
        if suggestions:
            preview += "**Did you mean:**\n"
            for input_name, matches in suggestions.items():
                top_match = matches[0]
                preview += f"- `{input_name}` → **{top_match[0]}** ({int(top_match[1]*100)}% match)\n"
            preview += "\n*Edit your input with the correct spelling and try again.*\n\n"
        if not_found:
            preview += f"Not found: {', '.join(not_found)}\n"
        return (preview, gr.update(visible=False), gr.update(), gr.update(), "")
    
    # Resolve confirmed drugs
    progress(0.3, desc="Resolving drug data...")
    resolved = [d[2] for d in found_drugs]  # drug_data is third element
    not_found_final = [s for s in suggestions.keys()] + not_found
    
    if not resolved:
        input_drugs = kg.parse_drug_input(drug_input)
        return (f"No drugs found in database: {', '.join(input_drugs)}", gr.update(visible=False), gr.update(), gr.update(), "")
    
    # Get drug IDs
    drug_ids = [d.get('drugbank_id', '') for d in resolved if d.get('drugbank_id')]
    
    # Calculate risk score
    progress(0.4, desc="Calculating polypharmacy risk...")
    risk_score, interactions, counts = kg.calculate_risk_score(drug_ids)
    
    risk_level = "CRITICAL" if risk_score >= 0.8 else "HIGH" if risk_score >= 0.5 else "MODERATE" if risk_score >= 0.2 else "LOW"
    
    # Find alternatives for high-risk drugs
    progress(0.6, desc="Finding safer alternatives...")
    alternatives_map = {}
    
    if interactions:
        # Find which drugs are involved in severe interactions
        severe_drugs = set()
        for i in interactions:
            sev = i.get('severity', '').lower()
            if 'contraindicated' in sev or 'major' in sev:
                severe_drugs.add(i.get('drug1_id', ''))
                severe_drugs.add(i.get('drug2_id', ''))
        
        # Find alternatives for each severe drug
        for drug_id in severe_drugs:
            if drug_id in drug_ids:
                other_ids = [d for d in drug_ids if d != drug_id]
                alternatives = kg.find_alternatives(drug_id, other_ids)
                if alternatives:
                    drug_name = kg.drugs_by_id.get(drug_id, {}).get('name', drug_id)
                    alternatives_map[drug_name] = alternatives
    
    # Get shared side effects
    progress(0.75, desc="Analyzing shared effects...")
    all_se = {}
    all_proteins = {}
    for drug in resolved:
        did = drug.get('drugbank_id', '')
        for se in kg.side_effects.get(did, []):
            n = se['name']
            if n not in all_se: all_se[n] = []
            all_se[n].append(drug.get('name', ''))
        for p in kg.proteins.get(did, []):
            n = p['name']
            if n and n not in all_proteins: all_proteins[n] = {'drugs': [], 'gene': p['gene']}
            if n: all_proteins[n]['drugs'].append(drug.get('name', ''))
    
    shared_se = {k: v for k, v in all_se.items() if len(v) > 1}
    shared_proteins = {k: v for k, v in all_proteins.items() if len(v['drugs']) > 1}
    
    # Store for chat context
    current_analysis["drugs"] = [d.get('name', '') for d in resolved]
    current_analysis["interactions"] = interactions
    current_analysis["risk"] = risk_level
    current_analysis["alternatives"] = alternatives_map
    
    # Build comprehensive report
    progress(0.9, desc="Generating report...")
    report = build_report(resolved, not_found_final, interactions, risk_level, risk_score, counts, 
                         shared_se, shared_proteins, alternatives_map)
    current_analysis["report"] = report
    
    # Update chat assistant's memory with the new analysis
    assistant = get_chat_assistant()
    assistant.update_memory(
        report=report,
        drugs=current_analysis["drugs"],
        risk=risk_level,
        interactions=interactions,
        alternatives=alternatives_map
    )
    
    # Prepare checkbox choices for drug selection panel
    current_drug_names = [d.get('name', '').title() for d in resolved]
    
    # Collect all alternative names (flatten)
    alt_choices = []
    for original_drug, alts in alternatives_map.items():
        for alt in alts[:3]:  # Top 3 alternatives per drug
            alt_name = alt['name'].title()
            if alt_name not in alt_choices and alt_name.lower() not in [d.lower() for d in current_drug_names]:
                alt_choices.append(f"{alt_name} (replaces {original_drug.title()})")
    
    progress(1.0, desc="Complete")
    
    # Return: report, selection panel visibility, current drugs checkbox, alternatives checkbox
    return (
        report,
        gr.update(visible=True),  # selection_panel
        gr.update(choices=current_drug_names, value=current_drug_names),  # current_drugs_check (all selected)
        gr.update(choices=alt_choices, value=[]),  # alternatives_check (none selected)
        ""  # selection_status
    )


# ============================================================
# Re-analyze with Selected Drugs
# ============================================================

def reanalyze_with_selection(current_drugs_selected, alternatives_selected, progress=gr.Progress()):
    """Re-run analysis with user's selected drugs (original + chosen alternatives)"""
    
    # Extract drug names from selections
    selected_drugs = list(current_drugs_selected) if current_drugs_selected else []
    
    # Extract alternative drug names (format: "DrugName (replaces Original)")
    for alt in (alternatives_selected or []):
        # Parse "DrugName (replaces Original)"
        if " (replaces " in alt:
            drug_name = alt.split(" (replaces ")[0].strip()
        else:
            drug_name = alt.strip()
        if drug_name and drug_name not in selected_drugs:
            selected_drugs.append(drug_name)
    
    if len(selected_drugs) < 2:
        return (
            f"Select at least 2 drugs to analyze. Currently selected: {', '.join(selected_drugs) if selected_drugs else 'none'}",
            gr.update(), gr.update(), gr.update(),
            f"Need at least 2 drugs (have {len(selected_drugs)})"
        )
    
    # Run analysis with the selected drug combination
    drug_input = ", ".join(selected_drugs)
    
    status_msg = f"Re-analyzing with: **{', '.join(selected_drugs)}**"
    
    # Call analyze_ddi with new combination
    result = analyze_ddi(drug_input, progress)
    
    # Update status with what was analyzed
    new_status = f"Analyzed **{len(selected_drugs)}** drugs: {', '.join(selected_drugs)}"
    
    # Result is a tuple (report, panel_visible, current_drugs, alternatives, status)
    # Return with updated status
    return (result[0], result[1], result[2], result[3], new_status)


def build_report(drugs, not_found, interactions, risk_level, risk_score, counts, 
                 shared_se, shared_proteins, alternatives_map):
    """Build compact, elegant DDI report with collapsible sections"""
    
    drug_names = [d.get('name', 'Unknown').title() for d in drugs]
    
    # Risk styling
    risk_styles = {
        'CRITICAL': ('', '#dc3545', 'Avoid this combination'),
        'HIGH': ('', '#fd7e14', 'Use with extreme caution'),
        'MODERATE': ('', '#ffc107', 'Monitor closely'),
        'LOW': ('', '#28a745', 'Generally safe')
    }
    icon, color, advice = risk_styles.get(risk_level, ('', '#6c757d', 'Unknown'))
    
    # === COMPACT HEADER ===
    report = f"""<div style="text-align:center; padding:16px 0; border-bottom:2px solid {color}; margin-bottom:16px;">
<h2 style="margin:8px 0 4px 0; color:{color};">{risk_level} RISK</h2>
<p style="margin:0; color:#666; font-size:0.95em;">{advice} • Score: {risk_score:.2f}</p>
</div>

**Analyzed:** {', '.join(drug_names)} ({len(drugs)} drugs) • **Found:** {len(interactions)} interaction{'s' if len(interactions) != 1 else ''}
"""
    
    if not_found:
        report += f"\n\n*Not found: {', '.join(not_found)}*"
    
    # === INTERACTION SUMMARY (always visible) ===
    if interactions:
        # Build severity badges inline
        badges = []
        if counts.get('contraindicated', 0):
            badges.append(f"{counts['contraindicated']} contraindicated")
        if counts.get('major', 0):
            badges.append(f"{counts['major']} major")
        if counts.get('moderate', 0):
            badges.append(f"{counts['moderate']} moderate")
        if counts.get('minor', 0):
            badges.append(f"{counts['minor']} minor")
        
        report += f"\n\n**Severity:** {' • '.join(badges)}"
        
        # Show top 3 most severe interactions inline
        severity_order = {'contraindicated': 0, 'major': 1, 'moderate': 2, 'minor': 3}
        sorted_int = sorted(interactions, 
            key=lambda x: severity_order.get(x.get('severity', '').lower().split()[0], 4))
        
        report += "\n\n**Key Interactions:**"
        for i in sorted_int[:3]:
            sev = i.get('severity', '').lower()
            d1 = i.get('drug1_name', i.get('drug1_id', '?')).title()
            d2 = i.get('drug2_name', i.get('drug2_id', '?')).title()
            desc = str(i.get('description', ''))[:100]
            if len(str(i.get('description', ''))) > 100:
                desc += "..."
            report += f"\n- **{d1} + {d2}:** {desc}"
    else:
        report += "\n\n**No significant interactions** detected in the knowledge graph."
    
    # === COLLAPSIBLE: Full Interactions ===
    if len(interactions) > 3:
        report += f"""

<details>
<summary>View all {len(interactions)} interactions</summary>

| Drugs | Severity | Effect |
|-------|----------|--------|"""
        for i in sorted_int:
            sev = i.get('severity', 'Unknown')
            sev_lower = sev.lower()
            badge = 'Contra.' if 'contraindicated' in sev_lower else 'Major' if 'major' in sev_lower else 'Moderate' if 'moderate' in sev_lower else 'Minor'
            d1 = i.get('drug1_name', i.get('drug1_id', '?')).title()
            d2 = i.get('drug2_name', i.get('drug2_id', '?')).title()
            desc = str(i.get('description', ''))[:80] + ('...' if len(str(i.get('description', ''))) > 80 else '')
            report += f"\n| {d1} + {d2} | {badge} | {desc} |"
        report += "\n\n</details>"
    
    # === COLLAPSIBLE: Alternatives ===
    if alternatives_map:
        alt_summary = []
        for orig, alts in alternatives_map.items():
            if alts:
                a = alts[0]
                alt_summary.append(f"**{a['name'].title()}** for {orig.title()} (-{a['risk_reduction']*100:.0f}% risk)")
        
        if alt_summary:
            report += f"\n\n**Safer Alternatives:** {', '.join(alt_summary[:2])}"
            
            if len(list(alternatives_map.items())) > 0:
                report += """

<details>
<summary>View all alternatives</summary>

"""
                for orig, alts in alternatives_map.items():
                    if not alts:
                        continue
                    report += f"**Replace {orig.title()}:**\n"
                    for alt in alts[:3]:
                        report += f"- [{alt['name'].title()}](https://go.drugbank.com/drugs/{alt['drugbank_id']}) — Risk: {alt['alternative_risk']:.2f} ({alt['num_interactions']} DDIs)\n"
                    report += "\n"
                report += "</details>"
    
    # === COLLAPSIBLE: Monitoring ===
    monitoring = []
    for i in interactions:
        desc = str(i.get('description', '')).lower()
        if 'bleeding' in desc or 'anticoagul' in desc:
            monitoring.append("🩸 **Bleeding:** PT/INR, CBC")
        if 'serotonin' in desc:
            monitoring.append("🧠 **Serotonin syndrome:** Mental status, autonomic signs")
        if 'qt' in desc:
            monitoring.append("❤️ **QT prolongation:** ECG monitoring")
        if 'hypotension' in desc:
            monitoring.append("📉 **Hypotension:** BP monitoring")
        if 'renal' in desc or 'kidney' in desc:
            monitoring.append("🫘 **Nephrotoxicity:** Creatinine, BUN")
        if 'hepat' in desc or 'liver' in desc:
            monitoring.append("🫀 **Hepatotoxicity:** LFTs")
    
    monitoring = list(dict.fromkeys(monitoring))  # Remove duplicates
    
    if monitoring:
        report += f"""

<details>
<summary>Recommended Monitoring</summary>

{chr(10).join(monitoring)}

</details>"""
    
    # === COLLAPSIBLE: Drug Details ===
    report += """

<details>
<summary>Drug Details</summary>

| Drug | DrugBank ID |
|------|-------------|"""
    for d in drugs:
        db_id = d.get('drugbank_id', 'N/A')
        report += f"\n| {d.get('name', 'Unknown').title()} | [{db_id}](https://go.drugbank.com/drugs/{db_id}) |"
    report += "\n\n</details>"
    
    # === COLLAPSIBLE: Molecular (only if data exists) ===
    if shared_se or shared_proteins:
        report += """

<details>
<summary>Molecular Overlap</summary>

"""
        if shared_proteins:
            prots = [f"{name} ({data['gene']})" for name, data in list(shared_proteins.items())[:3]]
            report += f"**Shared targets:** {', '.join(prots)}\n\n"
        if shared_se:
            ses = [f"{name}" for name, _ in list(shared_se.items())[:5]]
            report += f"**Overlapping side effects:** {', '.join(ses)}\n\n"
        report += "</details>"
    
    # === COMPACT FOOTER ===
    report += f"""

---
<p style="text-align:center; font-size:0.8em; color:#888;">
Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • DrugBank • SIDER • UniProt<br/>
<em>For educational purposes only — consult healthcare professionals</em>
</p>
"""
    
    return report


# ============================================================
# LLM Chat
# ============================================================

def extract_text_from_message(msg):
    """Extract plain text from various Gradio 6 message formats"""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        # Check for 'content' key (standard format)
        if 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Content might be list of parts
                texts = []
                for part in content:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict) and 'text' in part:
                        texts.append(part['text'])
                return ' '.join(texts)
        # Check for 'text' key directly
        if 'text' in msg:
            return msg['text']
    if isinstance(msg, list):
        # List of message parts
        texts = []
        for part in msg:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and 'text' in part:
                texts.append(part['text'])
        return ' '.join(texts)
    return str(msg)


def chat(message, history, model_name):
    """
    Natural Conversation Chat
    
    Uses conversation memory and stored analysis report for
    fluid, ChatGPT-like dialogue about drug interactions.
    """
    try:
        # Extract clean text from message (handles Gradio 6 format)
        clean_message = extract_text_from_message(message)
        if not clean_message.strip():
            return history, ""
        
        # Get the natural chat assistant
        assistant = get_chat_assistant()
        
        # Generate natural response using conversation memory
        response = assistant.respond(clean_message, model_name)
        
        # Handle empty or error response
        if not response or response.startswith("[LLM Error"):
            response = f"I apologize, but I'm having trouble connecting to the language model. Please make sure Ollama is running (`ollama serve`) and the model is available.\n\nError: {response}"
        
        # Return in Gradio 6 dict format [{'role': 'user/assistant', 'content': '...'}]
        if history is None:
            history = []
        
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": response})
        return history, ""
    except Exception as e:
        # Handle any unexpected errors
        if history is None:
            history = []
        error_msg = f"An error occurred: {str(e)}"
        history.append({"role": "user", "content": str(message)})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


# ============================================================
# Gradio Interface
# ============================================================

def create_app():
    
    # Custom CSS for full-width 3-column layout
    custom_css = """
    .gradio-container { 
        max-width: 100% !important; 
        width: 100% !important;
        padding: 20px 40px !important;
        margin: 0 !important;
    }
    .main { max-width: 100% !important; }
    .contain { max-width: 100% !important; }
    .wrap { max-width: 100% !important; }
    .section-title { font-size: 1.1em; font-weight: 600; color: #4a5568; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #667eea; }
    
    /* Enhanced Button Styling */
    .gr-button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    .gr-button:active {
        transform: translateY(0) !important;
    }
    
    /* Primary buttons - vibrant gradient */
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Secondary buttons - clean outline style */
    .gr-button-secondary {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        color: #4a5568 !important;
    }
    .gr-button-secondary:hover {
        background: #f7fafc !important;
        border-color: #667eea !important;
        color: #667eea !important;
    }
    
    /* Stop/danger buttons - for clear/reset actions */
    .gr-button-stop {
        background: white !important;
        border: 2px solid #fc8181 !important;
        color: #c53030 !important;
    }
    .gr-button-stop:hover {
        background: #fff5f5 !important;
        border-color: #e53e3e !important;
        box-shadow: 0 4px 12px rgba(229, 62, 62, 0.2) !important;
    }
    
    /* Large buttons */
    .gr-button-lg {
        padding: 14px 28px !important;
        font-size: 1.05em !important;
    }
    
    /* Small buttons */
    .gr-button-sm {
        padding: 8px 16px !important;
        font-size: 0.9em !important;
    }
    
    /* Chat send button - compact and accent */
    .chat-send-btn {
        background: linear-gradient(135deg, #38b2ac 0%, #319795 100%) !important;
    }
    .chat-send-btn:hover {
        background: linear-gradient(135deg, #2c9f9a 0%, #2a8584 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 178, 172, 0.4) !important;
    }
    
    .gr-textbox { border-radius: 8px !important; }
    details { background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0; }
    details summary { cursor: pointer; font-weight: 600; color: #4a5568; }
    details summary:hover { color: #667eea; }
    .column-panel { 
        min-height: 600px; 
        border: 1px solid #e2e8f0; 
        border-radius: 12px; 
        padding: 20px !important;
        background: #fafbfc;
    }
    """
    
    with gr.Blocks(title="DDI Risk Analyzer", css=custom_css, fill_width=True) as app:
        
        gr.Markdown("""
        <div style="text-align:center; padding:16px 0; margin-bottom:24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;">
        <h1 style="margin:0; font-size:1.8em; color:#fff;">Drug Regimen Analysis</h1>
        <p style="margin:6px 0 0 0; color:#e0e0e0; font-size:1em;">Knowledge Graph-powered interaction detection & recommendations</p>
        </div>
        """)
        
        # Three columns layout with equal spacing - full width
        with gr.Row(equal_height=True):
            # ============================================================
            # COLUMN 1: Drug Input (List, Narrative, or Image)
            # ============================================================
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["column-panel"]):
                    gr.Markdown("### 1. Enter Medications", elem_classes=["section-title"])
                    
                    with gr.Tabs():
                        # Tab 1: Drug List
                        with gr.Tab("Drug List"):
                            drug_input = gr.Textbox(
                                label="Drug Names",
                                placeholder="warfarin, aspirin, metoprolol, lisinopril",
                                lines=3,
                                info="Separate with commas"
                            )
                        
                        # Tab 2: Narrative Description
                        with gr.Tab("Narrative"):
                            narrative_input = gr.Textbox(
                                label="Describe Your Medications",
                                placeholder="I take cardace 5mg in the morning and aspirin 81mg at night for my heart condition...",
                                lines=4,
                                info="Describe in your own words"
                            )
                            extract_narrative_btn = gr.Button("Extract Drugs from Text", variant="secondary")
                            narrative_status = gr.Markdown("*Enter a description of your medications*")
                        
                        # Tab 3: Image Upload
                        with gr.Tab("Image"):
                            image_input = gr.Image(
                                label="Upload Prescription/Medication Photo",
                                type="pil",
                                height=150
                            )
                            extract_image_btn = gr.Button("Extract Drugs from Image", variant="secondary")
                            image_status = gr.Markdown("*Upload a photo of your prescription or medication bottles*")
                    
                    with gr.Row():
                        quick_check_btn = gr.Button("Check Drugs", variant="primary")
                        clear_btn = gr.Button("Clear All", variant="stop")
                    
                    # Drug identification preview
                    preview_output = gr.Markdown(
                        value="*Enter drug names and click 'Check Drugs' to identify and validate*"
                    )
                    
                    # Editable confirmed drugs list (shows after checking)
                    with gr.Group(visible=False) as edit_group:
                        gr.Markdown("**Confirmed Drugs** (edit, add, or remove)")
                        confirmed_drugs = gr.Textbox(
                            label="",
                            placeholder="Edit drug names here...",
                            lines=2,
                            info="Separate with commas. Edit and click 'Re-check' to validate changes."
                        )
                        with gr.Row():
                            recheck_btn = gr.Button("Re-check List", variant="secondary", size="sm")
                            use_list_btn = gr.Button("Use This List", variant="primary", size="sm")
                    
                    # Edit panel (shows after analysis for alternatives)
                    with gr.Accordion("Safer Alternatives", open=True, visible=False) as selection_panel:
                        gr.Markdown("**Current Drugs** (uncheck to remove)")
                        current_drugs_check = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True)
                        gr.Markdown("**Suggested Alternatives** (check to add)")
                        alternatives_check = gr.CheckboxGroup(choices=[], value=[], label="", interactive=True)
                        
                        reanalyze_btn = gr.Button("Re-analyze with Changes", variant="primary")
                        selection_status = gr.Markdown("")
                    
                    analyze_btn = gr.Button("Generate Report", variant="primary", size="lg")
            
            # ============================================================
            # COLUMN 2: Report
            # ============================================================
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["column-panel"]):
                    gr.Markdown("### 2. Analysis Report", elem_classes=["section-title"])
                    
                    report_output = gr.Markdown(
                        value="*Click 'Generate Report' to analyze drug interactions*"
                    )
            
            # ============================================================
            # COLUMN 3: Chat Assistant
            # ============================================================
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["column-panel"]):
                    gr.Markdown("### 3. Chat Assistant", elem_classes=["section-title"])
                    
                    model_select = gr.Dropdown(
                        choices=list(LLMClient.MODELS.keys()),
                        value="Mistral 7B (Fast)",
                        label="Model"
                    )
                    
                    chatbot = gr.Chatbot(
                        height=350, 
                        label="",
                        show_label=False,
                        avatar_images=(None, None)
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask about mechanisms, interactions, alternatives...",
                            label="",
                            show_label=False,
                            lines=1,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes=["chat-send-btn"])
        
        # Event handlers
        quick_check_btn.click(
            identify_drugs_preview, 
            inputs=[drug_input], 
            outputs=[preview_output, confirmed_drugs, edit_group], 
            show_progress="full"
        )
        
        # Re-check edited drug list
        recheck_btn.click(
            identify_drugs_preview, 
            inputs=[confirmed_drugs], 
            outputs=[preview_output, confirmed_drugs, edit_group], 
            show_progress="full"
        )
        
        # Use confirmed list and copy to main input
        def use_confirmed_list(drugs):
            """Copy confirmed drugs to main input"""
            return drugs
        
        use_list_btn.click(
            use_confirmed_list,
            inputs=[confirmed_drugs],
            outputs=[drug_input]
        )
        
        # Wrapper to use confirmed_drugs if available, else drug_input
        def analyze_with_fallback(confirmed, raw_input, progress=gr.Progress()):
            """Use confirmed drugs if available, otherwise use raw input"""
            drugs_to_analyze = confirmed if confirmed and confirmed.strip() else raw_input
            return analyze_ddi(drugs_to_analyze, progress)
        
        analyze_btn.click(
            analyze_with_fallback, 
            inputs=[confirmed_drugs, drug_input], 
            outputs=[report_output, selection_panel, current_drugs_check, alternatives_check, selection_status], 
            show_progress="full"
        )
        
        drug_input.submit(
            identify_drugs_preview,
            inputs=[drug_input],
            outputs=[preview_output, confirmed_drugs, edit_group],
            show_progress="full"
        )
        
        reanalyze_btn.click(
            reanalyze_with_selection,
            inputs=[current_drugs_check, alternatives_check],
            outputs=[report_output, selection_panel, current_drugs_check, alternatives_check, selection_status],
            show_progress="full"
        )
        
        # Narrative extraction handler
        extract_narrative_btn.click(
            extract_drugs_from_narrative,
            inputs=[narrative_input],
            outputs=[drug_input, narrative_status],
            show_progress="full"
        )
        
        # Image extraction handler
        extract_image_btn.click(
            extract_drugs_from_image,
            inputs=[image_input],
            outputs=[drug_input, image_status],
            show_progress="full"
        )
        
        def clear_all():
            return (
                "",  # drug_input
                "",  # narrative_input 
                None,  # image_input
                "*Enter drug names and click 'Check Drugs' to identify and validate*",  # preview_output
                "",  # confirmed_drugs
                gr.update(visible=False),  # edit_group
                "*Click 'Generate Report' to analyze drug interactions*",  # report_output
                [],  # chatbot
                gr.update(visible=False),  # selection_panel
                gr.update(choices=[], value=[]),  # current_drugs_check
                gr.update(choices=[], value=[]),  # alternatives_check
                "",  # selection_status
                "*Enter a description of your medications*",  # narrative_status
                "*Upload a photo of your prescription or medication bottles*"  # image_status
            )
        
        clear_btn.click(
            clear_all, 
            outputs=[drug_input, narrative_input, image_input, preview_output, confirmed_drugs, edit_group, report_output, chatbot,
                     selection_panel, current_drugs_check, alternatives_check, selection_status, narrative_status, image_status]
        )
        
        send_btn.click(chat, inputs=[chat_input, chatbot, model_select], outputs=[chatbot, chat_input])
        chat_input.submit(chat, inputs=[chat_input, chatbot, model_select], outputs=[chatbot, chat_input])
    
    return app


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DDI Risk Analyzer with Alternatives")
    print("="*60)
    
    print("\nLoading Knowledge Graph...")
    result = kg.load()
    print(f"   {result}")
    
    print("\nStarting application...")
    print("   URL: http://127.0.0.1:7860")
    print("   Press Ctrl+C to stop\n")
    
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
