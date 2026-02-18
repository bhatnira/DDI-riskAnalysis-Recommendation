#!/usr/bin/env python3
"""
LLM-Enhanced KG Recommendation System

Combines Knowledge Graph-based drug recommendations with LLM explanations
for human-readable, conversational drug interaction analysis.

Features:
1. Natural language explanations of KG recommendations
2. Interactive conversational interface
3. Patient-friendly risk summaries
4. Clinical decision support with citations

Requires: ollama serve && ollama pull biomistral (or llama3)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Import KG components
from kg_recommendation_system import (
    KGRecommendationEngine, KnowledgeGraphLoader,
    RecommendationResult, DrugAlternative, RiskContributor
)
from kg_polypharmacy_risk import PolypharmacyRiskAssessor, PolypharmacyRiskResult

# Import LLM client
try:
    from agents.llm_client import OllamaClient, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM client not available. Install or check agents/llm_client.py")


# ============================================================================
# LLM PROMPTS
# ============================================================================

SYSTEM_PROMPT_CLINICAL = """You are a clinical pharmacist AI assistant helping healthcare providers understand drug-drug interactions and polypharmacy risks.

Your role:
- Explain drug interactions in clear, professional language
- Provide evidence-based recommendations
- Highlight clinical significance and patient safety concerns
- Suggest alternatives when appropriate
- Be concise but thorough

Always prioritize patient safety. When uncertain, recommend consultation with a specialist."""

SYSTEM_PROMPT_PATIENT = """You are a friendly healthcare assistant helping patients understand their medications.

Your role:
- Explain drug interactions in simple, non-technical language
- Help patients understand why certain combinations may be concerning
- Provide practical advice they can discuss with their doctor
- Be reassuring while being honest about risks
- Avoid medical jargon

Always encourage patients to talk to their healthcare provider before making any changes."""

PROMPT_EXPLAIN_RISK = """Based on the following drug interaction analysis, provide a clear explanation:

**Current Medications:** {drugs}

**Overall Risk Level:** {risk_level} (Score: {risk_score:.2f}/1.00)

**Key Interactions Found:**
{interactions}

**Risk Breakdown:**
{risk_components}

Please explain:
1. Why this combination is concerning (if it is)
2. Which specific drug pairs are most problematic
3. What patients/providers should watch for
4. General guidance (without specific medical advice)

Keep your response concise and actionable."""

PROMPT_EXPLAIN_RECOMMENDATION = """Based on the drug analysis, here are the recommended changes:

**Current Regimen:** {original_drugs}
**Risk Level:** {risk_level}

**Highest Risk Pairs:**
{risk_pairs}

**Recommended Substitutions:**
{substitutions}

**Optimized Regimen:** {optimized_drugs}
**New Risk:** {new_risk:.2f} (Reduction: {risk_reduction:.2f})

Please explain:
1. Why each substitution was recommended
2. How the alternatives are therapeutically similar
3. What benefits the changes provide
4. Important considerations for implementation

Be specific about the clinical rationale."""

PROMPT_ANSWER_QUESTION = """Context from the drug interaction Knowledge Graph:

**Patient's Medications:** {drugs}
**Risk Assessment:** {risk_level} ({risk_score:.2f})

**Key Drug Pairs and Their Interactions:**
{interactions}

**Recommended Alternatives (if any):**
{alternatives}

**User's Question:** {question}

Please answer the question based on the knowledge graph data provided. If the information isn't available, say so. Always recommend consulting a healthcare provider for medical decisions."""


# ============================================================================
# LLM-ENHANCED RECOMMENDER
# ============================================================================

@dataclass
class LLMRecommendationResult:
    """Result with LLM-generated explanations"""
    # Original KG results
    kg_result: Dict = field(default_factory=dict)
    
    # LLM explanations
    risk_explanation: str = ""
    recommendation_explanation: str = ""
    patient_summary: str = ""
    
    # Chat history for conversational interface
    chat_history: List[Dict] = field(default_factory=list)
    
    # Metadata
    llm_model: str = ""
    generation_time: str = ""


class LLMEnhancedRecommender:
    """
    Combines KG-based recommendations with LLM explanations
    
    Usage:
        recommender = LLMEnhancedRecommender()
        
        # Get explained recommendation
        result = recommender.analyze_regimen(['warfarin', 'aspirin', 'metoprolol'])
        print(result.risk_explanation)
        
        # Interactive chat
        answer = recommender.ask("Why is warfarin + aspirin risky?")
    """
    
    def __init__(self, 
                 model: str = "biomistral",
                 fallback_model: str = "llama3:8b",
                 use_llm: bool = True):
        """
        Initialize the LLM-enhanced recommender
        
        Args:
            model: Primary LLM model (biomistral recommended for medical)
            fallback_model: Fallback if primary unavailable
            use_llm: If False, skip LLM and use template explanations
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_client = None
        self.llm_model = None
        
        # Initialize KG engine
        print("📚 Loading Knowledge Graph...")
        self.kg = KnowledgeGraphLoader().load()
        self.recommendation_engine = KGRecommendationEngine(self.kg)
        self.risk_assessor = PolypharmacyRiskAssessor(self.kg)
        
        # Initialize LLM
        if self.use_llm:
            self._init_llm(model, fallback_model)
        
        # Chat context
        self.current_context: Dict = {}
        self.chat_history: List[Dict] = []
    
    def _init_llm(self, model: str, fallback: str):
        """Initialize LLM client"""
        try:
            self.llm_client = OllamaClient(model=model)
            
            if self.llm_client.is_available():
                available_models = self.llm_client.get_available_models()
                
                # Check for primary model
                if any(model.split(':')[0] in m for m in available_models):
                    self.llm_model = model
                    print(f"🧠 LLM: {model} connected via Ollama")
                # Try fallback
                elif any(fallback.split(':')[0] in m for m in available_models):
                    self.llm_client = OllamaClient(model=fallback)
                    self.llm_model = fallback
                    print(f"🧠 LLM: {fallback} (fallback) connected via Ollama")
                # Use first available
                elif available_models:
                    first_model = available_models[0]
                    self.llm_client = OllamaClient(model=first_model)
                    self.llm_model = first_model
                    print(f"🧠 LLM: {first_model} connected via Ollama")
                else:
                    self.use_llm = False
                    print("⚠️ No LLM models available. Using template mode.")
            else:
                self.use_llm = False
                print("⚠️ Ollama not running. Start with: ollama serve")
                print("   Using template-based explanations.")
        except Exception as e:
            self.use_llm = False
            print(f"⚠️ LLM initialization failed: {e}")
            print("   Using template-based explanations.")
    
    def _generate_llm_response(self, prompt: str, system: str = None, 
                               temperature: float = 0.7) -> str:
        """Generate LLM response with fallback"""
        if not self.use_llm or not self.llm_client:
            return "[LLM unavailable - template response]"
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system or SYSTEM_PROMPT_CLINICAL,
                temperature=temperature,
                max_tokens=1024
            )
            
            if response.success:
                return response.content.strip()
            else:
                return f"[LLM error: {response.error}]"
        except Exception as e:
            return f"[LLM error: {str(e)}]"
    
    def _format_interactions(self, risk_result: PolypharmacyRiskResult) -> str:
        """Format interactions for LLM prompt"""
        lines = []
        for pair in risk_result.ddi_pairs[:5]:
            d1 = pair.get('drug1', '?')
            d2 = pair.get('drug2', '?')
            sev = pair.get('severity', 'Unknown')
            desc = pair.get('description', '')[:150]
            lines.append(f"- {d1} + {d2}: {sev}")
            if desc:
                lines.append(f"  Description: {desc}...")
        return '\n'.join(lines) if lines else "No significant interactions found."
    
    def _format_risk_components(self, risk_result: PolypharmacyRiskResult) -> str:
        """Format risk components for LLM prompt"""
        return f"""- DDI Severity Score: {risk_result.ddi_risk:.3f}
- Side Effect Overlap: {risk_result.side_effect_risk:.3f}
- Protein Target Overlap: {risk_result.protein_overlap_risk:.3f}
- Pathway Conflicts: {risk_result.pathway_risk:.3f}
- Network Centrality: {risk_result.network_risk:.3f}"""
    
    def _format_substitutions(self, rec_result: RecommendationResult) -> str:
        """Format substitutions for LLM prompt"""
        lines = []
        for sub in rec_result.recommended_substitutions:
            lines.append(f"- Replace {sub['replace']} → {sub['with']}")
            lines.append(f"  Therapeutic similarity: {sub['therapeutic_similarity']:.2f}")
            lines.append(f"  Safety score: {sub['safety_score']:.2f}")
            if sub.get('shared_proteins'):
                lines.append(f"  Shared mechanisms: {', '.join(sub['shared_proteins'][:3])}")
        return '\n'.join(lines) if lines else "No substitutions recommended."
    
    def _template_risk_explanation(self, risk_result: PolypharmacyRiskResult) -> str:
        """Template-based explanation when LLM unavailable"""
        level = risk_result.risk_level
        score = risk_result.overall_risk_score
        
        explanations = {
            'CRITICAL': f"""⚠️ CRITICAL RISK DETECTED (Score: {score:.2f})

This medication combination poses SERIOUS safety concerns:
• Contraindicated drug interactions are present
• Immediate clinical review is strongly recommended
• Do not modify medications without consulting your healthcare provider

Key concerns identified:
{self._format_interactions(risk_result)}

RECOMMENDATION: Urgent consultation with prescribing physician or clinical pharmacist.""",
            
            'HIGH': f"""🟠 HIGH RISK IDENTIFIED (Score: {score:.2f})

This medication combination has SIGNIFICANT interaction risks:
• Major drug interactions detected that require attention
• Monitoring and possible medication adjustments recommended

Key interactions:
{self._format_interactions(risk_result)}

RECOMMENDATION: Schedule appointment with healthcare provider to review medications.""",
            
            'MODERATE': f"""🟡 MODERATE RISK (Score: {score:.2f})

This medication combination has some interaction concerns:
• Interactions detected but generally manageable
• Monitoring recommended

Key points:
{self._format_interactions(risk_result)}

RECOMMENDATION: Discuss at next regular appointment with your provider.""",
            
            'LOW': f"""🟢 LOW RISK (Score: {score:.2f})

This medication combination appears generally safe:
• Only minor interactions detected
• Continue regular monitoring

Note: Always inform your healthcare providers of all medications you take."""
        }
        
        return explanations.get(level, f"Risk Level: {level} (Score: {score:.2f})")
    
    def _template_recommendation_explanation(self, rec_result: RecommendationResult) -> str:
        """Template recommendation explanation"""
        lines = [
            f"📋 RECOMMENDATION SUMMARY",
            f"",
            f"Current Risk: {rec_result.original_risk_level} ({rec_result.original_risk_score:.2f})",
            f"Optimized Risk: {rec_result.optimized_risk_score:.2f}",
            f"Risk Reduction: {rec_result.risk_reduction:.2f}",
            ""
        ]
        
        if rec_result.recommended_substitutions:
            lines.append("Suggested Changes:")
            for sub in rec_result.recommended_substitutions:
                lines.append(f"• {sub['replace']} → {sub['with']}")
                lines.append(f"  (Similar mechanism, better safety profile)")
        else:
            lines.append("No specific substitutions recommended at this time.")
        
        lines.extend([
            "",
            "⚠️ Always consult your healthcare provider before changing medications."
        ])
        
        return '\n'.join(lines)
    
    def analyze_regimen(self, drugs: List[str], 
                       audience: str = "clinical") -> LLMRecommendationResult:
        """
        Analyze a drug regimen and generate explained recommendations
        
        Args:
            drugs: List of drug names
            audience: "clinical" for healthcare providers, "patient" for patients
            
        Returns:
            LLMRecommendationResult with explanations
        """
        result = LLMRecommendationResult(
            generation_time=datetime.now().isoformat(),
            llm_model=self.llm_model or "template"
        )
        
        # Get KG-based analysis
        print(f"\n🔍 Analyzing regimen: {', '.join(drugs)}")
        
        risk_result = self.risk_assessor.assess_polypharmacy_risk(drugs)
        rec_result = self.recommendation_engine.recommend(drugs)
        
        # Store KG results
        result.kg_result = {
            'drugs': drugs,
            'risk_score': risk_result.overall_risk_score,
            'risk_level': risk_result.risk_level,
            'ddi_count': len(risk_result.ddi_pairs),
            'recommendations': rec_result.recommended_substitutions,
            'optimized_regimen': rec_result.optimized_regimen,
            'risk_reduction': rec_result.risk_reduction
        }
        
        # Update chat context
        self.current_context = result.kg_result
        self.current_context['risk_result'] = risk_result
        self.current_context['rec_result'] = rec_result
        
        # Generate explanations
        system_prompt = SYSTEM_PROMPT_PATIENT if audience == "patient" else SYSTEM_PROMPT_CLINICAL
        
        if self.use_llm:
            # LLM-generated risk explanation
            risk_prompt = PROMPT_EXPLAIN_RISK.format(
                drugs=', '.join(drugs),
                risk_level=risk_result.risk_level,
                risk_score=risk_result.overall_risk_score,
                interactions=self._format_interactions(risk_result),
                risk_components=self._format_risk_components(risk_result)
            )
            result.risk_explanation = self._generate_llm_response(risk_prompt, system_prompt)
            
            # LLM-generated recommendation explanation
            if rec_result.recommended_substitutions:
                rec_prompt = PROMPT_EXPLAIN_RECOMMENDATION.format(
                    original_drugs=', '.join(drugs),
                    risk_level=risk_result.risk_level,
                    risk_pairs='\n'.join([f"- {rc.drug1} + {rc.drug2}: {rc.ddi_severity}" 
                                         for rc in rec_result.risk_contributors[:3]]),
                    substitutions=self._format_substitutions(rec_result),
                    optimized_drugs=', '.join(rec_result.optimized_regimen),
                    new_risk=rec_result.optimized_risk_score,
                    risk_reduction=rec_result.risk_reduction
                )
                result.recommendation_explanation = self._generate_llm_response(rec_prompt, system_prompt)
        else:
            # Template fallbacks
            result.risk_explanation = self._template_risk_explanation(risk_result)
            result.recommendation_explanation = self._template_recommendation_explanation(rec_result)
        
        # Patient-friendly summary (always generate if for patients)
        if audience == "patient" and self.use_llm:
            patient_prompt = f"""Summarize this drug interaction analysis for a patient in 3-4 simple sentences:
            
Medications: {', '.join(drugs)}
Risk: {risk_result.risk_level}
Main concern: {risk_result.ddi_pairs[0].get('description', 'potential interactions')[:100] if risk_result.ddi_pairs else 'None identified'}

Be reassuring but honest. Encourage them to talk to their doctor."""
            result.patient_summary = self._generate_llm_response(patient_prompt, SYSTEM_PROMPT_PATIENT)
        
        return result
    
    def ask(self, question: str) -> str:
        """
        Answer a question about the current drug analysis
        
        Args:
            question: Natural language question
            
        Returns:
            LLM-generated answer
        """
        if not self.current_context:
            return "Please analyze a drug regimen first using analyze_regimen()."
        
        # Build context from current analysis
        risk_result = self.current_context.get('risk_result')
        rec_result = self.current_context.get('rec_result')
        
        prompt = PROMPT_ANSWER_QUESTION.format(
            drugs=', '.join(self.current_context.get('drugs', [])),
            risk_level=self.current_context.get('risk_level', 'Unknown'),
            risk_score=self.current_context.get('risk_score', 0),
            interactions=self._format_interactions(risk_result) if risk_result else "None",
            alternatives=self._format_substitutions(rec_result) if rec_result else "None",
            question=question
        )
        
        if self.use_llm:
            answer = self._generate_llm_response(prompt, SYSTEM_PROMPT_CLINICAL, temperature=0.5)
        else:
            answer = f"[Template mode] Based on the analysis of {', '.join(self.current_context.get('drugs', []))}, the risk level is {self.current_context.get('risk_level', 'unknown')}. For specific questions, please consult a healthcare provider."
        
        # Track chat history
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        return answer
    
    def interactive_session(self):
        """
        Start an interactive chat session
        """
        print("\n" + "=" * 60)
        print("💊 LLM-Enhanced Drug Interaction Advisor")
        print("=" * 60)
        print("\nCommands:")
        print("  analyze <drug1> <drug2> ...  - Analyze drug combination")
        print("  ask <question>               - Ask about current analysis")
        print("  explain                      - Get detailed explanation")
        print("  recommend                    - Get recommendations")
        print("  patient                      - Patient-friendly summary")
        print("  quit                         - Exit")
        print("\n" + "-" * 60)
        
        while True:
            try:
                user_input = input("\n🔹 You: ").strip()
                
                if not user_input:
                    continue
                
                words = user_input.split()
                command = words[0].lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif command == 'analyze':
                    drugs = words[1:] if len(words) > 1 else []
                    if not drugs:
                        print("❌ Please specify drugs: analyze warfarin aspirin metoprolol")
                        continue
                    
                    result = self.analyze_regimen(drugs)
                    print(f"\n📊 Analysis Complete")
                    print(f"   Risk: {result.kg_result['risk_level']} ({result.kg_result['risk_score']:.2f})")
                    print(f"\n{result.risk_explanation}")
                
                elif command == 'ask':
                    question = ' '.join(words[1:]) if len(words) > 1 else ''
                    if not question:
                        print("❌ Please ask a question: ask why is warfarin risky?")
                        continue
                    
                    answer = self.ask(question)
                    print(f"\n🤖 Assistant: {answer}")
                
                elif command == 'explain':
                    if not self.current_context:
                        print("❌ Analyze a regimen first: analyze warfarin aspirin")
                        continue
                    
                    result = self.analyze_regimen(self.current_context['drugs'])
                    print(f"\n📝 Detailed Explanation:\n{result.risk_explanation}")
                
                elif command == 'recommend':
                    if not self.current_context:
                        print("❌ Analyze a regimen first")
                        continue
                    
                    result = self.analyze_regimen(self.current_context['drugs'])
                    print(f"\n💊 Recommendations:\n{result.recommendation_explanation}")
                
                elif command == 'patient':
                    if not self.current_context:
                        print("❌ Analyze a regimen first")
                        continue
                    
                    result = self.analyze_regimen(self.current_context['drugs'], audience="patient")
                    print(f"\n👤 Patient Summary:\n{result.patient_summary or result.risk_explanation}")
                
                else:
                    # Treat as a question about current context
                    answer = self.ask(user_input)
                    print(f"\n🤖 Assistant: {answer}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# ============================================================================
# DEMO AND CLI
# ============================================================================

def demo():
    """Demonstrate LLM-enhanced recommendations"""
    print("\n" + "=" * 70)
    print("LLM-ENHANCED DRUG RECOMMENDATION DEMO")
    print("=" * 70)
    
    recommender = LLMEnhancedRecommender()
    
    # Demo regimen
    drugs = ['warfarin', 'aspirin', 'metoprolol', 'lisinopril']
    
    print(f"\n📋 Test Regimen: {', '.join(drugs)}")
    
    # Analyze
    result = recommender.analyze_regimen(drugs)
    
    print("\n" + "-" * 70)
    print("📊 RISK EXPLANATION (Clinical)")
    print("-" * 70)
    print(result.risk_explanation)
    
    print("\n" + "-" * 70)
    print("💊 RECOMMENDATION EXPLANATION")
    print("-" * 70)
    print(result.recommendation_explanation)
    
    # Interactive Q&A
    print("\n" + "-" * 70)
    print("🤔 SAMPLE QUESTIONS")
    print("-" * 70)
    
    questions = [
        "Why is warfarin + aspirin concerning?",
        "What should I monitor for?",
        "Are there safer alternatives?"
    ]
    
    for q in questions:
        print(f"\n❓ Q: {q}")
        answer = recommender.ask(q)
        print(f"💬 A: {answer}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM-Enhanced KG Drug Recommendation System'
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive chat session')
    parser.add_argument('--analyze', nargs='+',
                       help='Analyze drug regimen')
    parser.add_argument('--patient', action='store_true',
                       help='Generate patient-friendly explanations')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM, use templates only')
    parser.add_argument('--model', type=str, default='biomistral',
                       help='LLM model to use (default: biomistral)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.interactive:
        recommender = LLMEnhancedRecommender(
            model=args.model,
            use_llm=not args.no_llm
        )
        recommender.interactive_session()
    elif args.analyze:
        recommender = LLMEnhancedRecommender(
            model=args.model,
            use_llm=not args.no_llm
        )
        audience = "patient" if args.patient else "clinical"
        result = recommender.analyze_regimen(args.analyze, audience=audience)
        
        print(f"\n{'=' * 60}")
        print(f"RISK ANALYSIS: {result.kg_result['risk_level']}")
        print(f"{'=' * 60}")
        print(result.risk_explanation)
        
        if result.recommendation_explanation:
            print(f"\n{'=' * 60}")
            print("RECOMMENDATIONS")
            print(f"{'=' * 60}")
            print(result.recommendation_explanation)
    else:
        # Default: interactive
        recommender = LLMEnhancedRecommender(use_llm=not args.no_llm)
        recommender.interactive_session()


if __name__ == "__main__":
    main()
