#!/usr/bin/env python3
"""
DDI Knowledge Graph Chatbot
Interactive conversational interface for drug-drug interaction queries

Supports multiple LLM backends:
- Ollama (local)
- Groq (free API)
- OpenAI-compatible APIs
- Template fallback (no LLM)

Author: DDI Risk Analysis Research Team
"""

import os
import sys
import json
import re
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from kg_polypharmacy_risk import KnowledgeGraphLoader, PolypharmacyRiskAssessor
from kg_recommendation_system import KGRecommendationEngine


@dataclass
class ChatConfig:
    backend: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    max_history: int = 10
    system_prompt: str = """You are DDI Assistant, an expert AI for drug-drug interactions. 
You have access to a knowledge graph with 4,313 drugs and 759,774 interactions.
Provide accurate, helpful responses about drug safety. Always recommend consulting healthcare providers."""


class LLMBackend(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OllamaBackend(LLMBackend):
    def __init__(self, config: ChatConfig):
        self.config = config
        self.base_url = config.ollama_url
        self.model = self._detect_model()
    
    def _detect_model(self) -> Optional[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'].split(':')[0] for m in response.json().get('models', [])]
                if self.config.ollama_model in models:
                    return self.config.ollama_model
                for fallback in ['mistral', 'llama2', 'phi']:
                    if fallback in models:
                        return fallback
                return models[0] if models else None
        except:
            return None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def get_model_name(self) -> str:
        return f"ollama/{self.model}" if self.model else "ollama/unavailable"
    
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        if not self.model:
            return "[Ollama not available]"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=stream, timeout=120)
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get('message', {}).get('content', '')
                    full_response += chunk
                    if stream:
                        print(chunk, end='', flush=True)
                    if data.get('done'):
                        break
            if stream:
                print()
            return full_response
        except Exception as e:
            return f"[Ollama Error: {e}]"


class GroqBackend(LLMBackend):
    def __init__(self, config: ChatConfig):
        self.config = config
        self.api_key = config.groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.model = config.groq_model
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_model_name(self) -> str:
        return f"groq/{self.model}"
    
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        if not self.api_key:
            return "[Groq API key not set]"
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, stream=stream, timeout=120
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and '[DONE]' not in line_str:
                        data = json.loads(line_str[6:])
                        chunk = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if chunk:
                            full_response += chunk
                            if stream:
                                print(chunk, end='', flush=True)
            if stream:
                print()
            return full_response
        except Exception as e:
            return f"[Groq Error: {e}]"


class OpenAIBackend(LLMBackend):
    def __init__(self, config: ChatConfig):
        self.config = config
        self.api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY', '')
        self.base_url = config.openai_base_url
        self.model = config.openai_model
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_model_name(self) -> str:
        return f"openai/{self.model}"
    
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        if not self.api_key:
            return "[OpenAI API key not set]"
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, stream=stream, timeout=120
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and '[DONE]' not in line_str:
                        data = json.loads(line_str[6:])
                        chunk = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if chunk:
                            full_response += chunk
                            if stream:
                                print(chunk, end='', flush=True)
            if stream:
                print()
            return full_response
        except Exception as e:
            return f"[OpenAI Error: {e}]"


class TemplateBackend(LLMBackend):
    def __init__(self, config: ChatConfig):
        self.config = config
    
    def is_available(self) -> bool:
        return True
    
    def get_model_name(self) -> str:
        return "template"
    
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        if 'Knowledge Graph Context:' in user_msg:
            context = user_msg.split('Knowledge Graph Context:')[-1].split('Based on')[0].strip()
        else:
            context = 'No specific drug data found.'
        
        response = f"Based on the Knowledge Graph:\n\n{context}\n\nConsult a healthcare provider for medical decisions."
        print(response)
        return response


class LLMClient:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.backends = {
            'ollama': OllamaBackend(config),
            'groq': GroqBackend(config),
            'openai': OpenAIBackend(config),
            'template': TemplateBackend(config)
        }
        self.active_backend = self._select_backend()
    
    def _select_backend(self) -> LLMBackend:
        if self.config.backend in self.backends and self.backends[self.config.backend].is_available():
            return self.backends[self.config.backend]
        for name in ['groq', 'ollama', 'openai', 'template']:
            if self.backends[name].is_available():
                return self.backends[name]
        return self.backends['template']
    
    def get_model_name(self) -> str:
        return self.active_backend.get_model_name()
    
    def generate(self, messages: List[Dict], stream: bool = True) -> str:
        return self.active_backend.generate(messages, stream)
    
    def list_backends(self) -> Dict[str, bool]:
        return {name: b.is_available() for name, b in self.backends.items()}


class DrugExtractor:
    def __init__(self, kg_loader: KnowledgeGraphLoader):
        self.drug_names = set(d.lower() for d in kg_loader.drugs.keys())
        self.aliases = {
            'aspirin': ['acetylsalicylic acid', 'asa'],
            'tylenol': ['acetaminophen', 'paracetamol'],
            'advil': ['ibuprofen'], 'motrin': ['ibuprofen'],
            'coumadin': ['warfarin'], 'lipitor': ['atorvastatin'],
            'zocor': ['simvastatin'], 'lasix': ['furosemide'],
            'glucophage': ['metformin'], 'prilosec': ['omeprazole'],
            'plavix': ['clopidogrel'], 'xanax': ['alprazolam'],
        }
        self.reverse_aliases = {alias: generic for generic, aliases in self.aliases.items() for alias in aliases}
    
    def extract_drugs(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        
        for drug in self.drug_names:
            if re.search(r'\b' + re.escape(drug) + r'\b', text_lower):
                found.append(drug)
        
        for alias, generic in self.reverse_aliases.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', text_lower) and generic not in found:
                found.append(generic)
        
        return list(dict.fromkeys(found))


class KnowledgeGraphRAG:
    def __init__(self, kg_loader: KnowledgeGraphLoader):
        self.kg = kg_loader
        self.risk_assessor = PolypharmacyRiskAssessor(kg_loader)
        self.extractor = DrugExtractor(kg_loader)
    
    def build_context(self, query: str, drugs: List[str] = None) -> str:
        if not drugs:
            drugs = self.extractor.extract_drugs(query)
        
        if not drugs:
            return "No specific drugs detected in query."
        
        parts = [f"Drugs detected: {', '.join(drugs)}"]
        
        for drug in drugs[:5]:
            drug_node = next((d for d in self.kg.drugs if d.lower() == drug.lower()), None)
            if drug_node:
                info = self.kg.drugs[drug_node]
                atc = ', '.join(info.get('atc_codes', [])[:3]) or 'Unknown'
                n_inter = len(self.kg.ddis.get(drug_node, {}))
                parts.append(f"\n{drug_node}: ATC={atc}, Interactions={n_inter}")
        
        if len(drugs) >= 2:
            result = self.risk_assessor.assess_polypharmacy_risk(drugs)
            parts.append(f"\nRisk Analysis: {result.risk_level} (score: {result.overall_risk_score:.2f})")
            
            for inter in result.ddi_pairs[:5]:
                d1, d2 = inter.get('drug1', '?'), inter.get('drug2', '?')
                sev = inter.get('severity', 'Unknown')
                desc = inter.get('description', '')[:80]
                parts.append(f"  - {d1} + {d2}: {sev}")
                if desc:
                    parts.append(f"    {desc}")
        
        return '\n'.join(parts)


class DDIChatbot:
    def __init__(self, config: ChatConfig = None):
        self.config = config or ChatConfig()
        
        print("Initializing DDI Chatbot...")
        print("Loading Knowledge Graph...")
        self.kg = KnowledgeGraphLoader().load()
        
        self.rag = KnowledgeGraphRAG(self.kg)
        
        print("Connecting to LLM...")
        self.llm = LLMClient(self.config)
        
        model = self.llm.get_model_name()
        if model != "template":
            print(f"Connected: {model}")
        else:
            print("No LLM available - using template mode")
            print("Set GROQ_API_KEY (free) or run 'ollama serve'")
        
        self.history = []
    
    def process_query(self, query: str) -> str:
        drugs = self.rag.extractor.extract_drugs(query)
        context = self.rag.build_context(query, drugs)
        
        prompt = f"User Query: {query}\n\nKnowledge Graph Context:\n{context}\n\nProvide a helpful response about drug interactions."
        
        self.history.append({"role": "user", "content": prompt})
        if len(self.history) > self.config.max_history * 2:
            self.history = self.history[-self.config.max_history * 2:]
        
        messages = [{"role": "system", "content": self.config.system_prompt}] + self.history
        
        print("\nAssistant: ", end='')
        response = self.llm.generate(messages, stream=True)
        
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def run_interactive(self):
        print("\n" + "=" * 60)
        print("DDI KNOWLEDGE GRAPH CHATBOT")
        print("=" * 60)
        print(f"Knowledge Base: {len(self.kg.drugs):,} drugs, {sum(len(v) for v in self.kg.ddis.values()):,} interactions")
        print(f"LLM: {self.llm.get_model_name()}")
        print("\nExamples: 'Check warfarin and aspirin', 'Is ibuprofen safe with lisinopril?'")
        print("Commands: help, clear, backends, quit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if user_input.lower() == 'clear':
                    self.history = []
                    print("History cleared.")
                    continue
                if user_input.lower() == 'backends':
                    for name, avail in self.llm.list_backends().items():
                        print(f"  {'Y' if avail else 'N'} {name}")
                    print(f"Active: {self.llm.get_model_name()}")
                    continue
                if user_input.lower() == 'help':
                    print("Commands: help, clear, backends, quit")
                    print("Ask about drug interactions, e.g., 'Check warfarin and aspirin'")
                    continue
                
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def quick_check(drugs: List[str]):
    print(f"\nQuick Check: {', '.join(drugs)}")
    print("-" * 40)
    
    kg = KnowledgeGraphLoader().load()
    assessor = PolypharmacyRiskAssessor(kg)
    result = assessor.assess_polypharmacy_risk(drugs)
    
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.overall_risk_score:.2f}")
    
    if result.ddi_pairs:
        print(f"\nInteractions ({len(result.ddi_pairs)}):")
        for inter in result.ddi_pairs[:5]:
            print(f"  {inter.get('drug1', '?')} + {inter.get('drug2', '?')}: {inter.get('severity', '?')}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DDI Knowledge Graph Chatbot')
    parser.add_argument('--backend', default='ollama', choices=['ollama', 'groq', 'openai', 'template'])
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--check', nargs='+', metavar='DRUG', help='Quick interaction check')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    
    args = parser.parse_args()
    
    if args.check:
        quick_check(args.check)
        return
    
    config = ChatConfig(backend=args.backend, ollama_url=args.url)
    if args.model:
        if args.backend == 'ollama':
            config.ollama_model = args.model
        elif args.backend == 'groq':
            config.groq_model = args.model
        elif args.backend == 'openai':
            config.openai_model = args.model
    
    chatbot = DDIChatbot(config)
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
