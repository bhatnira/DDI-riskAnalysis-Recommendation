"""
Multi-Objective Drug Recommender - Implements the paper's recommender algorithm

For a given high-risk drug pair or polypharmacy set, the recommender:
1. Identifies the highest-risk drug contributor
2. Retrieves same-ATC candidate alternatives
3. Computes replacement risk delta
4. Ranks alternatives using multi-objective score:
   - Risk reduction
   - Network centrality reduction
   - Interaction phenotype avoidance
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .drug_risk_network import DrugRiskNetwork, DrugNode


@dataclass
class AlternativeCandidate:
    """Represents a potential alternative drug"""
    drug_name: str
    atc_code: str
    atc_match_level: int  # 3 = pharmacological, 4 = chemical
    pri_score: float
    pri_delta: float  # Reduction in PRI
    centrality_delta: float  # Reduction in centrality
    new_interactions: int  # Interactions with remaining drugs
    avoided_phenotypes: List[str]  # Phenotypes no longer present
    new_phenotypes: List[str]  # New phenotypes introduced
    multi_objective_score: float


class MultiObjectiveRecommender:
    """
    Multi-objective drug alternative recommender
    
    Ranking criteria:
    1. Risk reduction (PRI delta)
    2. Network centrality reduction
    3. Interaction phenotype avoidance
    4. Minimize new severe interactions
    """
    
    # Multi-objective weights
    OBJECTIVE_WEIGHTS = {
        'pri_reduction': 0.35,
        'centrality_reduction': 0.20,
        'phenotype_avoidance': 0.25,
        'new_interaction_penalty': 0.20
    }
    
    # Critical phenotypes to avoid
    CRITICAL_PHENOTYPES = {
        'bleeding', 'arrhythmia', 'nephrotoxicity', 
        'hepatotoxicity', 'serotonin_syndrome'
    }
    
    def __init__(self, risk_network: DrugRiskNetwork):
        self.network = risk_network
        self.atc_to_drugs: Dict[str, List[str]] = defaultdict(list)
        self._build_atc_index()
    
    def _build_atc_index(self) -> None:
        """Build ATC code to drug mapping for therapeutic similarity"""
        for drug_name, node in self.network.nodes.items():
            if node.atc_level_3:
                self.atc_to_drugs[node.atc_level_3].append(drug_name)
            if node.atc_level_4:
                self.atc_to_drugs[node.atc_level_4].append(drug_name)
    
    def identify_highest_risk_contributor(self, 
                                          drug_list: List[str]) -> Tuple[str, Dict]:
        """
        Step 1: Identify the highest-risk drug contributor
        
        Returns:
            Tuple of (drug_name, risk_metrics)
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Compute contribution to total risk for each drug
        contributions = []
        
        for drug in drugs_lower:
            if drug not in self.network.nodes:
                continue
            
            node = self.network.nodes[drug]
            
            # Count interactions with other drugs in the list
            list_interactions = 0
            severe_interactions = 0
            phenotypes_involved = set()
            
            for other in drugs_lower:
                if other == drug or other not in self.network.nodes:
                    continue
                
                edge = self.network.adjacency.get(drug, {}).get(other)
                if edge:
                    list_interactions += 1
                    if edge.severity_label in ['Contraindicated interaction', 'Major interaction']:
                        severe_interactions += 1
                    phenotypes_involved.update(edge.phenotypes)
            
            # Contribution score
            contribution_score = (
                node.pri_score * 0.4 +
                (severe_interactions / max(1, len(drugs_lower) - 1)) * 0.4 +
                node.weighted_degree * 0.2
            )
            
            contributions.append({
                'drug': drug,
                'pri_score': node.pri_score,
                'list_interactions': list_interactions,
                'severe_interactions': severe_interactions,
                'phenotypes': list(phenotypes_involved),
                'contribution_score': contribution_score
            })
        
        # Sort by contribution score
        contributions.sort(key=lambda x: -x['contribution_score'])
        
        if contributions:
            highest = contributions[0]
            return (highest['drug'], highest)
        
        return (None, {})
    
    def get_atc_alternatives(self, 
                            drug: str, 
                            level: int = 4) -> List[str]:
        """
        Step 2: Retrieve same-ATC candidate alternatives
        
        Args:
            drug: Drug to find alternatives for
            level: ATC level (3 = pharmacological, 4 = chemical)
            
        Returns:
            List of alternative drug names
        """
        drug_lower = drug.lower()
        if drug_lower not in self.network.nodes:
            return []
        
        node = self.network.nodes[drug_lower]
        
        # Get ATC prefix at specified level
        if level == 4:
            atc_prefix = node.atc_level_4
        else:
            atc_prefix = node.atc_level_3
        
        if not atc_prefix:
            return []
        
        # Get all drugs with same ATC prefix
        alternatives = [
            d for d in self.atc_to_drugs.get(atc_prefix, [])
            if d != drug_lower
        ]
        
        return alternatives
    
    def compute_replacement_delta(self,
                                  original_drug: str,
                                  alternative_drug: str,
                                  current_drugs: List[str]) -> Dict[str, Any]:
        """
        Step 3: Compute replacement risk delta
        
        Args:
            original_drug: Drug being replaced
            alternative_drug: Proposed alternative
            current_drugs: Full list of current drugs
            
        Returns:
            Delta metrics for the replacement
        """
        orig_lower = original_drug.lower()
        alt_lower = alternative_drug.lower()
        other_drugs = [d.lower() for d in current_drugs if d.lower() != orig_lower]
        
        orig_node = self.network.nodes.get(orig_lower)
        alt_node = self.network.nodes.get(alt_lower)
        
        if not orig_node or not alt_node:
            return {}
        
        # PRI delta
        pri_delta = orig_node.pri_score - alt_node.pri_score
        
        # Centrality delta
        centrality_delta = (
            orig_node.degree_centrality - alt_node.degree_centrality +
            orig_node.betweenness_centrality - alt_node.betweenness_centrality
        ) / 2
        
        # Analyze interactions
        orig_interactions = []
        orig_phenotypes = set()
        for other in other_drugs:
            edge = self.network.adjacency.get(orig_lower, {}).get(other)
            if edge:
                orig_interactions.append({
                    'drug': other,
                    'severity': edge.severity_label,
                    'weight': edge.severity_weight
                })
                orig_phenotypes.update(edge.phenotypes)
        
        new_interactions = []
        new_phenotypes = set()
        for other in other_drugs:
            edge = self.network.adjacency.get(alt_lower, {}).get(other)
            if edge:
                new_interactions.append({
                    'drug': other,
                    'severity': edge.severity_label,
                    'weight': edge.severity_weight
                })
                new_phenotypes.update(edge.phenotypes)
        
        # Phenotype analysis
        avoided_phenotypes = orig_phenotypes - new_phenotypes
        introduced_phenotypes = new_phenotypes - orig_phenotypes
        
        # Severity comparison
        orig_severe = sum(1 for i in orig_interactions 
                        if i['severity'] in ['Contraindicated interaction', 'Major interaction'])
        new_severe = sum(1 for i in new_interactions 
                        if i['severity'] in ['Contraindicated interaction', 'Major interaction'])
        
        return {
            'pri_delta': pri_delta,
            'centrality_delta': centrality_delta,
            'original_interactions': len(orig_interactions),
            'new_interactions': len(new_interactions),
            'interaction_delta': len(orig_interactions) - len(new_interactions),
            'original_severe': orig_severe,
            'new_severe': new_severe,
            'severe_delta': orig_severe - new_severe,
            'avoided_phenotypes': list(avoided_phenotypes),
            'introduced_phenotypes': list(introduced_phenotypes),
            'critical_phenotypes_avoided': len(avoided_phenotypes & self.CRITICAL_PHENOTYPES),
            'critical_phenotypes_introduced': len(introduced_phenotypes & self.CRITICAL_PHENOTYPES)
        }
    
    def compute_multi_objective_score(self,
                                      delta: Dict[str, Any],
                                      alt_node: DrugNode) -> float:
        """
        Step 4: Compute multi-objective ranking score
        
        Higher score = better alternative
        """
        # Normalize components
        pri_reduction = max(0, delta.get('pri_delta', 0))  # Positive = good
        centrality_reduction = max(0, delta.get('centrality_delta', 0))
        
        # Phenotype avoidance score
        phenotype_score = (
            delta.get('critical_phenotypes_avoided', 0) * 2 +
            len(delta.get('avoided_phenotypes', [])) -
            delta.get('critical_phenotypes_introduced', 0) * 3 -
            len(delta.get('introduced_phenotypes', []))
        )
        phenotype_score = max(0, phenotype_score) / 10  # Normalize
        
        # New interaction penalty (lower is better, so invert)
        interaction_penalty = 1 - (delta.get('new_severe', 0) / 10)
        interaction_penalty = max(0, interaction_penalty)
        
        # Combine scores
        score = (
            self.OBJECTIVE_WEIGHTS['pri_reduction'] * pri_reduction +
            self.OBJECTIVE_WEIGHTS['centrality_reduction'] * centrality_reduction +
            self.OBJECTIVE_WEIGHTS['phenotype_avoidance'] * phenotype_score +
            self.OBJECTIVE_WEIGHTS['new_interaction_penalty'] * interaction_penalty
        )
        
        return score
    
    def recommend_alternatives(self,
                              drug_list: List[str],
                              target_drug: str = None,
                              max_alternatives: int = 5) -> Dict[str, Any]:
        """
        Main recommender function
        
        Args:
            drug_list: Current drug list
            target_drug: Specific drug to replace (if None, uses highest risk)
            max_alternatives: Maximum alternatives to return
            
        Returns:
            Recommendations with multi-objective scores
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Step 1: Identify target drug
        if target_drug:
            target = target_drug.lower()
            target_metrics = self.network.get_drug_metrics(target)
        else:
            target, target_metrics = self.identify_highest_risk_contributor(drug_list)
        
        if not target:
            return {'error': 'No valid target drug identified'}
        
        # Step 2: Get ATC alternatives (try level 4 first, then level 3)
        alternatives_l4 = self.get_atc_alternatives(target, level=4)
        alternatives_l3 = self.get_atc_alternatives(target, level=3)
        
        all_alternatives = list(set(alternatives_l4 + alternatives_l3))
        
        if not all_alternatives:
            return {
                'target_drug': target,
                'target_metrics': target_metrics,
                'alternatives': [],
                'message': 'No ATC-matched alternatives found'
            }
        
        # Step 3 & 4: Evaluate each alternative
        candidates = []
        
        for alt in all_alternatives:
            # Skip if alternative is already in the drug list
            if alt in drugs_lower:
                continue
            
            alt_node = self.network.nodes.get(alt)
            if not alt_node:
                continue
            
            # Compute delta
            delta = self.compute_replacement_delta(target, alt, drug_list)
            
            if not delta:
                continue
            
            # Compute multi-objective score
            mo_score = self.compute_multi_objective_score(delta, alt_node)
            
            # Determine ATC match level
            target_node = self.network.nodes.get(target)
            if target_node and alt_node.atc_level_4 == target_node.atc_level_4:
                atc_level = 4
            else:
                atc_level = 3
            
            candidate = AlternativeCandidate(
                drug_name=alt,
                atc_code=alt_node.atc_code,
                atc_match_level=atc_level,
                pri_score=alt_node.pri_score,
                pri_delta=delta.get('pri_delta', 0),
                centrality_delta=delta.get('centrality_delta', 0),
                new_interactions=delta.get('new_interactions', 0),
                avoided_phenotypes=delta.get('avoided_phenotypes', []),
                new_phenotypes=delta.get('introduced_phenotypes', []),
                multi_objective_score=mo_score
            )
            
            candidates.append((candidate, delta))
        
        # Rank by multi-objective score
        candidates.sort(key=lambda x: -x[0].multi_objective_score)
        
        # Format output
        recommendations = []
        for candidate, delta in candidates[:max_alternatives]:
            recommendations.append({
                'drug_name': candidate.drug_name.title(),
                'atc_code': candidate.atc_code,
                'atc_match_level': candidate.atc_match_level,
                'atc_match_type': 'Chemical subgroup' if candidate.atc_match_level == 4 else 'Pharmacological subgroup',
                'multi_objective_score': round(candidate.multi_objective_score, 4),
                'pri_score': round(candidate.pri_score, 4),
                'risk_metrics': {
                    'pri_reduction': round(candidate.pri_delta, 4),
                    'centrality_reduction': round(candidate.centrality_delta, 4),
                    'severe_interaction_delta': delta.get('severe_delta', 0),
                    'total_interaction_delta': delta.get('interaction_delta', 0)
                },
                'phenotype_analysis': {
                    'avoided': candidate.avoided_phenotypes,
                    'introduced': candidate.new_phenotypes,
                    'net_phenotype_improvement': len(candidate.avoided_phenotypes) - len(candidate.new_phenotypes)
                },
                'new_interactions_with_current': candidate.new_interactions
            })
        
        return {
            'target_drug': {
                'name': target.title(),
                'pri_score': round(self.network.nodes[target].pri_score, 4) if target in self.network.nodes else 0,
                'reason': 'Highest risk contributor' if not target_drug else 'User specified'
            },
            'current_drugs': [d.title() for d in drugs_lower],
            'alternatives': recommendations,
            'total_candidates_evaluated': len(all_alternatives),
            'ranking_criteria': list(self.OBJECTIVE_WEIGHTS.keys())
        }
    
    def recommend_for_polypharmacy(self,
                                   drug_list: List[str],
                                   max_replacements: int = 3) -> Dict[str, Any]:
        """
        Recommend alternatives for multiple high-risk drugs
        
        Args:
            drug_list: Current drug list
            max_replacements: Maximum drugs to suggest replacing
            
        Returns:
            Comprehensive polypharmacy recommendations
        """
        drugs_lower = [d.lower() for d in drug_list]
        
        # Get risk analysis
        risk_analysis = self.network.compute_polypharmacy_risk(drug_list)
        
        # Sort drugs by contribution to risk
        drug_risks = []
        for drug in drugs_lower:
            if drug not in self.network.nodes:
                continue
            
            node = self.network.nodes[drug]
            
            # Count severe interactions
            severe_count = 0
            for other in drugs_lower:
                if other == drug:
                    continue
                edge = self.network.adjacency.get(drug, {}).get(other)
                if edge and edge.severity_label in ['Contraindicated interaction', 'Major interaction']:
                    severe_count += 1
            
            drug_risks.append({
                'drug': drug,
                'pri': node.pri_score,
                'severe_interactions': severe_count,
                'combined_risk': node.pri_score + (severe_count * 0.2)
            })
        
        drug_risks.sort(key=lambda x: -x['combined_risk'])
        
        # Generate recommendations for top risk contributors
        all_recommendations = []
        
        for i, drug_risk in enumerate(drug_risks[:max_replacements]):
            drug = drug_risk['drug']
            
            # Get alternatives for this drug
            rec = self.recommend_alternatives(
                drug_list=drug_list,
                target_drug=drug,
                max_alternatives=3
            )
            
            if rec.get('alternatives'):
                all_recommendations.append({
                    'priority': i + 1,
                    'target_drug': drug.title(),
                    'risk_contribution': round(drug_risk['combined_risk'], 4),
                    'severe_interactions_in_list': drug_risk['severe_interactions'],
                    'best_alternative': rec['alternatives'][0] if rec['alternatives'] else None,
                    'all_alternatives': rec['alternatives']
                })
        
        return {
            'overall_risk': {
                'score': risk_analysis.get('risk_score', 0),
                'level': risk_analysis.get('risk_level', 'UNKNOWN'),
                'total_interactions': risk_analysis.get('total_interactions', 0),
                'severity_breakdown': risk_analysis.get('severity_breakdown', {})
            },
            'recommendations': all_recommendations,
            'summary': {
                'drugs_analyzed': len(drugs_lower),
                'drugs_with_alternatives': len(all_recommendations),
                'estimated_risk_reduction': self._estimate_risk_reduction(all_recommendations)
            }
        }
    
    def _estimate_risk_reduction(self, recommendations: List[Dict]) -> float:
        """Estimate potential risk reduction from recommendations"""
        total_reduction = 0
        for rec in recommendations:
            if rec.get('best_alternative'):
                pri_red = rec['best_alternative'].get('risk_metrics', {}).get('pri_reduction', 0)
                total_reduction += max(0, pri_red)
        return round(total_reduction, 4)
