#!/usr/bin/env python3
"""
Advanced Publication Figures Generator
Creates sophisticated visualizations for the paper including:
- Network visualizations
- Advanced statistical plots
- 3D visualizations
- Interactive-ready plots
- Sankey diagrams
- Chord diagrams
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict, Counter
import math

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))


class AdvancedFigureGenerator:
    """Generate advanced publication figures"""
    
    def __init__(self, output_dir: str = "publication/figures_advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.network = None
        self.recommender = None
        
    def load_data(self):
        """Load data and initialize network"""
        print("=" * 70)
        print("📊 ADVANCED FIGURES GENERATOR")
        print("=" * 70)
        
        data_file = 'ddi_cardio_or_antithrombotic_labeled (1).csv'
        print(f"\n📂 Loading data from: {data_file}")
        self.df = pd.read_csv(data_file)
        print(f"   ✓ Loaded {len(self.df):,} interactions")
        
        print("\n🔗 Building Drug Risk Network...")
        from agents import DrugRiskNetwork, MultiObjectiveRecommender
        self.network = DrugRiskNetwork()
        self.network.build_network(self.df)
        self.recommender = MultiObjectiveRecommender(self.network)
        
    def generate_all(self):
        """Generate all advanced figures"""
        self.load_data()
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        print("\n" + "=" * 70)
        print("📈 GENERATING ADVANCED FIGURES")
        print("=" * 70)
        
        self._fig_network_visualization(plt, sns)
        self._fig_network_communities(plt, sns)
        self._fig_degree_distribution(plt, sns)
        self._fig_pri_components_radar(plt, sns)
        self._fig_severity_sunburst(plt, sns)
        self._fig_drug_similarity_tsne(plt, sns)
        self._fig_interaction_flow(plt, sns)
        self._fig_risk_stratification(plt, sns)
        self._fig_temporal_severity(plt, sns)
        self._fig_atc_hierarchy(plt, sns)
        self._fig_phenotype_cooccurrence(plt, sns)
        self._fig_centrality_comparison(plt, sns)
        self._fig_recommendation_impact(plt, sns)
        self._fig_network_robustness(plt, sns)
        self._fig_pri_heatmap_detailed(plt, sns)
        
        plt.close('all')
        
        print("\n" + "=" * 70)
        print("✅ ADVANCED FIGURES COMPLETE")
        print(f"📁 Output: {self.output_dir.absolute()}")
        print("=" * 70)
    
    def _save_fig(self, name: str, plt):
        """Save figure in multiple formats"""
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.output_dir / f'{name}.{fmt}')
        print(f"     ✓ Saved {name}.[png/pdf/svg]")
    
    def _fig_network_visualization(self, plt, sns):
        """Network visualization using force-directed layout"""
        print("\n  → Advanced Network Visualization")
        
        # Get top drugs by PRI for visualization (subset for clarity)
        top_n = 100
        top_drugs = sorted(self.network.nodes.items(), 
                          key=lambda x: -x[1].pri_score)[:top_n]
        drug_names = [d[0] for d in top_drugs]
        
        # Build adjacency for top drugs
        edges = []
        for i, d1 in enumerate(drug_names):
            for j, d2 in enumerate(drug_names[i+1:], i+1):
                edge = self.network.adjacency.get(d1, {}).get(d2)
                if edge:
                    edges.append((i, j, edge.severity_weight))
        
        # Spring layout simulation
        np.random.seed(42)
        pos = np.random.rand(len(drug_names), 2) * 10
        
        # Simple force-directed placement
        for _ in range(50):
            # Repulsion
            for i in range(len(drug_names)):
                for j in range(i+1, len(drug_names)):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff) + 0.01
                    force = diff / (dist ** 2) * 0.5
                    pos[i] += force
                    pos[j] -= force
            
            # Attraction (edges)
            for i, j, w in edges:
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff) + 0.01
                force = diff * 0.01
                pos[i] += force
                pos[j] -= force
        
        # Normalize positions
        pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0) + 0.01)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Draw edges
        for i, j, w in edges:
            alpha = min(0.8, w / 10)
            color = '#d62728' if w >= 7 else '#ff7f0e' if w >= 4 else '#2ca02c'
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                   color=color, alpha=alpha * 0.5, linewidth=0.5)
        
        # Draw nodes
        pri_scores = [self.network.nodes[d].pri_score for d in drug_names]
        node_sizes = [300 + p * 1000 for p in pri_scores]
        
        # Color by category (cardiovascular includes antithrombotic)
        colors = []
        for d in drug_names:
            node = self.network.nodes[d]
            if node.is_cardiovascular:
                colors.append('#e41a1c')
            else:
                colors.append('#999999')
        
        scatter = ax.scatter(pos[:, 0], pos[:, 1], s=node_sizes, c=colors, 
                            alpha=0.7, edgecolors='white', linewidths=1)
        
        # Label top 15 drugs
        for i, (name, _) in enumerate(top_drugs[:15]):
            ax.annotate(name.title()[:12], (pos[i, 0], pos[i, 1]), 
                       fontsize=8, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        
        ax.set_title(f'Drug-Drug Interaction Network\n(Top {top_n} Drugs by PRI)', fontsize=16)
        ax.axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#e41a1c', label='Cardiovascular'),
            Patch(facecolor='#999999', label='Other'),
            Line2D([0], [0], color='#d62728', label='Contraindicated/Major'),
            Line2D([0], [0], color='#2ca02c', label='Moderate/Minor')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        self._save_fig('fig_adv01_network_visualization', plt)
    
    def _fig_network_communities(self, plt, sns):
        """Network communities/clusters visualization"""
        print("\n  → Network Communities Analysis")
        
        # Simple community detection based on ATC codes
        atc_communities = defaultdict(list)
        for name, node in self.network.nodes.items():
            if node.atc_code:
                atc_l1 = node.atc_code[0]
                atc_communities[atc_l1].append(name)
        
        # Get top 8 communities
        top_communities = sorted(atc_communities.items(), 
                                key=lambda x: -len(x[1]))[:8]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        atc_labels = {
            'A': 'Alimentary', 'B': 'Blood', 'C': 'Cardiovascular',
            'D': 'Dermatologicals', 'G': 'Genitourinary', 'H': 'Hormones',
            'J': 'Anti-infectives', 'L': 'Antineoplastic', 'M': 'Musculoskeletal',
            'N': 'Nervous System', 'P': 'Antiparasitic', 'R': 'Respiratory',
            'S': 'Sensory Organs', 'V': 'Various'
        }
        
        # Community sizes
        ax1 = axes[0, 0]
        codes = [c[0] for c in top_communities]
        sizes = [len(c[1]) for c in top_communities]
        colors = plt.cm.tab10.colors[:len(codes)]
        bars = ax1.bar(codes, sizes, color=colors)
        ax1.set_xlabel('ATC Level 1 Code')
        ax1.set_ylabel('Number of Drugs')
        ax1.set_title('A. Community Sizes (by ATC Classification)')
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(size), ha='center', va='bottom', fontsize=9)
        
        # Inter-community interactions
        ax2 = axes[0, 1]
        inter_matrix = np.zeros((len(top_communities), len(top_communities)))
        
        for i, (code1, drugs1) in enumerate(top_communities):
            for j, (code2, drugs2) in enumerate(top_communities):
                if i <= j:
                    count = 0
                    for d1 in drugs1[:50]:  # Sample for speed
                        for d2 in drugs2[:50]:
                            if self.network.adjacency.get(d1, {}).get(d2):
                                count += 1
                    inter_matrix[i, j] = count
                    inter_matrix[j, i] = count
        
        sns.heatmap(inter_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=codes, yticklabels=codes, ax=ax2)
        ax2.set_title('B. Inter-Community Interactions')
        
        # Average PRI by community
        ax3 = axes[1, 0]
        avg_pri = []
        for code, drugs in top_communities:
            pris = [self.network.nodes[d].pri_score for d in drugs]
            avg_pri.append(np.mean(pris))
        
        bars = ax3.bar(codes, avg_pri, color=colors)
        ax3.set_xlabel('ATC Level 1 Code')
        ax3.set_ylabel('Average PRI Score')
        ax3.set_title('C. Average PRI by Community')
        ax3.axhline(np.mean(avg_pri), color='red', linestyle='--', label='Overall Mean')
        ax3.legend()
        
        # Community density (internal connections)
        ax4 = axes[1, 1]
        densities = []
        for code, drugs in top_communities:
            n = len(drugs)
            if n > 1:
                internal = 0
                for i, d1 in enumerate(drugs[:100]):
                    for d2 in drugs[i+1:100]:
                        if self.network.adjacency.get(d1, {}).get(d2):
                            internal += 1
                max_edges = min(100, n) * (min(100, n) - 1) / 2
                densities.append(internal / max_edges if max_edges > 0 else 0)
            else:
                densities.append(0)
        
        ax4.bar(codes, densities, color=colors)
        ax4.set_xlabel('ATC Level 1 Code')
        ax4.set_ylabel('Internal Density')
        ax4.set_title('D. Community Internal Connectivity')
        
        plt.tight_layout()
        self._save_fig('fig_adv02_network_communities', plt)
    
    def _fig_degree_distribution(self, plt, sns):
        """Degree distribution analysis (log-log plot)"""
        print("\n  → Degree Distribution Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Collect degrees
        degrees = [len(self.network.drug_interactions.get(d, set())) 
                  for d in self.network.nodes]
        weighted_degrees = [n.weighted_degree for n in self.network.nodes.values()]
        
        # Degree distribution
        ax1 = axes[0, 0]
        degree_counts = Counter(degrees)
        x = sorted(degree_counts.keys())
        y = [degree_counts[k] for k in x]
        ax1.scatter(x, y, alpha=0.7, s=30)
        ax1.set_xlabel('Degree (k)')
        ax1.set_ylabel('Count P(k)')
        ax1.set_title('A. Degree Distribution')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Fit power law
        log_x = np.log10([k for k in x if k > 0])
        log_y = np.log10([degree_counts[k] for k in x if k > 0])
        if len(log_x) > 2:
            z = np.polyfit(log_x, log_y, 1)
            ax1.plot(x, [10**(z[1] + z[0]*np.log10(k)) for k in x], 
                    'r--', label=f'γ = {-z[0]:.2f}')
            ax1.legend()
        
        # Cumulative degree distribution
        ax2 = axes[0, 1]
        sorted_degrees = sorted(degrees, reverse=True)
        cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
        ax2.plot(sorted_degrees, cumulative, linewidth=2)
        ax2.set_xlabel('Degree (k)')
        ax2.set_ylabel('P(K ≥ k)')
        ax2.set_title('B. Cumulative Degree Distribution')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Weighted degree distribution
        ax3 = axes[1, 0]
        sns.histplot(weighted_degrees, bins=50, ax=ax3, color='#ff7f0e')
        ax3.set_xlabel('Weighted Degree (Normalized)')
        ax3.set_ylabel('Count')
        ax3.set_title('C. Weighted Degree Distribution')
        
        # Degree vs PRI correlation
        ax4 = axes[1, 1]
        pris = [n.pri_score for n in self.network.nodes.values()]
        ax4.scatter(degrees, pris, alpha=0.3, s=10)
        ax4.set_xlabel('Degree')
        ax4.set_ylabel('PRI Score')
        ax4.set_title('D. Degree vs PRI Correlation')
        
        # Add trend line
        z = np.polyfit(degrees, pris, 1)
        x_line = np.linspace(min(degrees), max(degrees), 100)
        ax4.plot(x_line, z[0]*x_line + z[1], 'r--', linewidth=2)
        
        # Calculate R²
        from scipy import stats
        r, _ = stats.pearsonr(degrees, pris)
        ax4.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax4.transAxes,
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        self._save_fig('fig_adv03_degree_distribution', plt)
    
    def _fig_pri_components_radar(self, plt, sns):
        """Radar chart of PRI components for top drugs"""
        print("\n  → PRI Components Radar Chart")
        
        # Get top 8 drugs
        top_drugs = sorted(self.network.nodes.items(), 
                          key=lambda x: -x[1].pri_score)[:8]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), 
                                subplot_kw=dict(projection='polar'))
        
        categories = ['Degree\nCentrality', 'Weighted\nDegree', 
                     'Betweenness\nCentrality', 'Severity\nProfile']
        n_cats = len(categories)
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Complete the loop
        
        for idx, (name, node) in enumerate(top_drugs):
            ax = axes[idx // 4, idx % 4]
            
            # Compute severity profile
            total_interactions = (node.contraindicated_count + node.major_count + 
                                 node.moderate_count + node.minor_count)
            severity_score = (node.contraindicated_count * 10 + node.major_count * 7 +
                             node.moderate_count * 4 + node.minor_count) / (total_interactions * 10 + 0.01)
            
            values = [node.degree_centrality, node.weighted_degree,
                     node.betweenness_centrality, min(1, severity_score)]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=plt.cm.tab10.colors[idx])
            ax.fill(angles, values, alpha=0.25, color=plt.cm.tab10.colors[idx])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'{name.title()[:15]}\nPRI: {node.pri_score:.3f}', 
                        size=10, pad=10)
        
        plt.suptitle('PRI Component Analysis for Top Risk Drugs', fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_fig('fig_adv04_pri_radar', plt)
    
    def _fig_severity_sunburst(self, plt, sns):
        """Sunburst-style visualization of severity hierarchy"""
        print("\n  → Severity Hierarchy Visualization")
        
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
        
        # Severity distribution
        severity_counts = self.df['severity_label'].value_counts()
        total = sum(severity_counts)
        
        # Inner ring: severity
        severity_colors = {
            'Contraindicated interaction': '#d62728',
            'Major interaction': '#ff7f0e',
            'Moderate interaction': '#2ca02c',
            'Minor interaction': '#1f77b4'
        }
        
        # Draw concentric rings
        inner_radius = 0.3
        outer_radius = 0.6
        
        start_angle = 0
        for severity in ['Contraindicated interaction', 'Major interaction', 
                        'Moderate interaction', 'Minor interaction']:
            count = severity_counts.get(severity, 0)
            angle = count / total * 2 * np.pi
            
            theta = np.linspace(start_angle, start_angle + angle, 50)
            r_inner = np.ones_like(theta) * inner_radius
            r_outer = np.ones_like(theta) * outer_radius
            
            ax.fill_between(theta, r_inner, r_outer, 
                           color=severity_colors[severity], alpha=0.8)
            
            # Add label
            mid_angle = start_angle + angle / 2
            label_r = (inner_radius + outer_radius) / 2
            if count / total > 0.05:
                ax.text(mid_angle, label_r, f'{severity.split()[0]}\n{count/total*100:.1f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold')
            
            start_angle += angle
        
        # Outer ring: phenotypes by severity
        phenotype_counts = defaultdict(lambda: defaultdict(int))
        for edge in self.network.edges:
            for phenotype in edge.phenotypes:
                phenotype_counts[edge.severity_label][phenotype] += 1
        
        outer_radius2 = 0.9
        start_angle = 0
        
        for severity in ['Contraindicated interaction', 'Major interaction']:
            severity_angle = severity_counts.get(severity, 0) / total * 2 * np.pi
            
            pheno_total = sum(phenotype_counts[severity].values())
            if pheno_total > 0:
                pheno_start = start_angle
                for pheno, count in sorted(phenotype_counts[severity].items(), 
                                          key=lambda x: -x[1])[:5]:
                    pheno_angle = count / pheno_total * severity_angle
                    
                    theta = np.linspace(pheno_start, pheno_start + pheno_angle, 30)
                    r_inner = np.ones_like(theta) * outer_radius
                    r_outer = np.ones_like(theta) * outer_radius2
                    
                    ax.fill_between(theta, r_inner, r_outer, 
                                   color=severity_colors[severity], alpha=0.4)
                    
                    if pheno_angle > 0.1:
                        mid_angle = pheno_start + pheno_angle / 2
                        ax.text(mid_angle, (outer_radius + outer_radius2) / 2,
                               pheno.replace('_', '\n')[:10], ha='center', va='center',
                               fontsize=7, rotation=np.degrees(mid_angle) - 90)
                    
                    pheno_start += pheno_angle
            
            start_angle += severity_angle
        
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('DDI Severity Distribution with Phenotype Breakdown', fontsize=14, pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=s.replace(' interaction', ''))
                         for s, c in severity_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.2, 1), fontsize=10)
        
        plt.tight_layout()
        self._save_fig('fig_adv05_severity_sunburst', plt)
    
    def _fig_drug_similarity_tsne(self, plt, sns):
        """t-SNE visualization of drug similarity based on interaction profiles"""
        print("\n  → Drug Similarity t-SNE Visualization")
        
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("     ⚠️ sklearn not available, skipping t-SNE")
            return
        
        # Build feature matrix for top drugs
        top_n = 500
        top_drugs = sorted(self.network.nodes.items(), 
                          key=lambda x: -x[1].pri_score)[:top_n]
        drug_names = [d[0] for d in top_drugs]
        
        # Features: centrality metrics + severity counts
        features = []
        for name in drug_names:
            node = self.network.nodes[name]
            features.append([
                node.degree_centrality,
                node.weighted_degree,
                node.betweenness_centrality,
                node.pri_score,
                node.contraindicated_count / 1000,
                node.major_count / 1000,
                node.moderate_count / 100,
                node.minor_count / 100
            ])
        
        features = np.array(features)
        features = StandardScaler().fit_transform(features)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = tsne.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Color by PRI
        ax1 = axes[0]
        pris = [self.network.nodes[d].pri_score for d in drug_names]
        scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], c=pris, 
                             cmap='RdYlGn_r', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax1, label='PRI Score')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.set_title('A. Drug Similarity Map (colored by PRI)')
        
        # Label some outliers
        for i in np.argsort(pris)[-5:]:
            ax1.annotate(drug_names[i].title()[:10], (embedding[i, 0], embedding[i, 1]),
                        fontsize=8, alpha=0.8)
        
        # Color by category (cardiovascular includes antithrombotic)
        ax2 = axes[1]
        colors = []
        for d in drug_names:
            node = self.network.nodes[d]
            if node.is_cardiovascular:
                colors.append('#e41a1c')
            else:
                colors.append('#999999')
        
        ax2.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=30)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('B. Drug Similarity Map (colored by Category)')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e41a1c', label='Cardiovascular'),
            Patch(facecolor='#999999', label='Other')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        self._save_fig('fig_adv06_drug_tsne', plt)
    
    def _fig_interaction_flow(self, plt, sns):
        """Sankey-style flow diagram of interactions"""
        print("\n  → Interaction Flow Diagram")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Aggregate data for flow
        atc_severity = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            atc1 = row['atc_1']
            if isinstance(atc1, str) and len(atc1) > 2:
                atc1 = atc1[2] if atc1[0] == '[' else atc1[0]
            else:
                atc1 = 'U'
            
            severity = row['severity_label']
            atc_severity[atc1][severity] += 1
        
        # Prepare flow data
        atc_order = ['C', 'B', 'N', 'A', 'M', 'J', 'L', 'R']
        severity_order = ['Contraindicated interaction', 'Major interaction',
                         'Moderate interaction', 'Minor interaction']
        severity_short = ['Contraindicated', 'Major', 'Moderate', 'Minor']
        
        # Left side: ATC codes
        left_y = np.linspace(0.9, 0.1, len(atc_order))
        right_y = np.linspace(0.85, 0.15, len(severity_order))
        
        # Calculate totals for sizing
        atc_totals = {atc: sum(atc_severity[atc].values()) for atc in atc_order}
        max_atc = max(atc_totals.values()) if atc_totals else 1
        
        severity_totals = defaultdict(int)
        for atc in atc_order:
            for sev in severity_order:
                severity_totals[sev] += atc_severity[atc][sev]
        max_sev = max(severity_totals.values()) if severity_totals else 1
        
        atc_labels = {
            'A': 'Alimentary', 'B': 'Blood', 'C': 'Cardiovascular',
            'N': 'Nervous', 'M': 'Musculoskeletal', 'J': 'Anti-infectives',
            'L': 'Antineoplastic', 'R': 'Respiratory', 'U': 'Unknown'
        }
        
        colors = plt.cm.tab10.colors
        severity_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        
        # Draw nodes (left - ATC)
        for i, atc in enumerate(atc_order):
            size = atc_totals.get(atc, 0) / max_atc * 0.08 + 0.02
            ax.add_patch(plt.Rectangle((0.05, left_y[i] - size/2), 0.08, size,
                                       color=colors[i % len(colors)], alpha=0.8))
            ax.text(0.02, left_y[i], f'{atc}\n{atc_labels.get(atc, atc)[:8]}',
                   ha='right', va='center', fontsize=9)
        
        # Draw nodes (right - Severity)
        for i, sev in enumerate(severity_order):
            size = severity_totals[sev] / max_sev * 0.1 + 0.02
            ax.add_patch(plt.Rectangle((0.87, right_y[i] - size/2), 0.08, size,
                                       color=severity_colors[i], alpha=0.8))
            ax.text(0.97, right_y[i], f'{severity_short[i]}\n({severity_totals[sev]:,})',
                   ha='left', va='center', fontsize=9)
        
        # Draw flows
        for i, atc in enumerate(atc_order):
            for j, sev in enumerate(severity_order):
                count = atc_severity[atc][sev]
                if count > 0:
                    width = count / max_sev * 0.05
                    alpha = min(0.5, count / max_sev + 0.1)
                    
                    # Bezier curve
                    x_points = [0.13, 0.4, 0.6, 0.87]
                    y_points = [left_y[i], left_y[i], right_y[j], right_y[j]]
                    
                    from matplotlib.patches import FancyBboxPatch
                    from matplotlib.path import Path
                    import matplotlib.patches as mpatches
                    
                    verts = [
                        (x_points[0], y_points[0]),
                        (x_points[1], y_points[1]),
                        (x_points[2], y_points[2]),
                        (x_points[3], y_points[3])
                    ]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    
                    patch = mpatches.PathPatch(path, facecolor='none',
                                              edgecolor=severity_colors[j],
                                              alpha=alpha, linewidth=width * 100)
                    ax.add_patch(patch)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Drug-Drug Interaction Flow: ATC Classification → Severity', fontsize=14)
        
        plt.tight_layout()
        self._save_fig('fig_adv07_interaction_flow', plt)
    
    def _fig_risk_stratification(self, plt, sns):
        """Risk stratification analysis"""
        print("\n  → Risk Stratification Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Collect drug data
        drug_data = []
        for name, node in self.network.nodes.items():
            drug_data.append({
                'drug': name,
                'pri': node.pri_score,
                'degree': node.degree_centrality,
                'weighted': node.weighted_degree,
                'contraindicated': node.contraindicated_count,
                'major': node.major_count,
                'is_cardio': node.is_cardiovascular,
                'is_cardio': node.is_cardiovascular
            })
        
        df = pd.DataFrame(drug_data)
        
        # Risk strata
        df['risk_stratum'] = pd.cut(df['pri'], bins=[0, 0.2, 0.35, 0.5, 1.0],
                                    labels=['Low', 'Moderate', 'High', 'Critical'])
        
        # Strata distribution
        ax1 = axes[0, 0]
        strata_counts = df['risk_stratum'].value_counts()
        colors = ['#2ca02c', '#ffff00', '#ff7f0e', '#d62728']
        ax1.pie(strata_counts, labels=strata_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('A. Drug Distribution by Risk Stratum')
        
        # Severity counts by stratum
        ax2 = axes[0, 1]
        strata_severity = df.groupby('risk_stratum')[['contraindicated', 'major']].mean()
        strata_severity.plot(kind='bar', ax=ax2, color=['#d62728', '#ff7f0e'])
        ax2.set_xlabel('Risk Stratum')
        ax2.set_ylabel('Average Interaction Count')
        ax2.set_title('B. Average Severe Interactions by Stratum')
        ax2.legend(['Contraindicated', 'Major'])
        ax2.tick_params(axis='x', rotation=0)
        
        # PRI distribution by category (cardiovascular includes antithrombotic)
        ax3 = axes[1, 0]
        category_data = []
        for _, row in df.iterrows():
            if row['is_cardio']:
                cat = 'Cardiovascular'
            else:
                cat = 'Other'
            category_data.append({'pri': row['pri'], 'category': cat})
        
        cat_df = pd.DataFrame(category_data)
        sns.boxplot(data=cat_df, x='category', y='pri', ax=ax3,
                   order=['Cardiovascular', 'Other'],
                   palette=['#e41a1c', '#999999'])
        ax3.set_xlabel('Drug Category')
        ax3.set_ylabel('PRI Score')
        ax3.set_title('C. PRI Distribution by Drug Category')
        
        # Cumulative risk curve
        ax4 = axes[1, 1]
        sorted_pri = sorted(df['pri'], reverse=True)
        cumulative_risk = np.cumsum(sorted_pri) / np.sum(sorted_pri)
        x = np.arange(1, len(sorted_pri) + 1) / len(sorted_pri) * 100
        
        ax4.plot(x, cumulative_risk * 100, linewidth=2, color='#1f77b4')
        ax4.fill_between(x, cumulative_risk * 100, alpha=0.3)
        ax4.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% of risk')
        ax4.axvline(x[np.searchsorted(cumulative_risk, 0.8)], color='red', 
                   linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Percentage of Drugs')
        ax4.set_ylabel('Cumulative Risk (%)')
        ax4.set_title('D. Cumulative Risk Curve')
        ax4.text(x[np.searchsorted(cumulative_risk, 0.8)] + 2, 75,
                f'{x[np.searchsorted(cumulative_risk, 0.8)]:.1f}% of drugs\ncontribute 80% of risk',
                fontsize=10)
        
        plt.tight_layout()
        self._save_fig('fig_adv08_risk_stratification', plt)
    
    def _fig_temporal_severity(self, plt, sns):
        """Severity confidence distribution analysis"""
        print("\n  → Severity Confidence Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Confidence distribution by severity
        ax1 = axes[0, 0]
        severity_order = ['Contraindicated interaction', 'Major interaction',
                         'Moderate interaction', 'Minor interaction']
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        
        for sev, color in zip(severity_order, colors):
            data = self.df[self.df['severity_label'] == sev]['severity_confidence']
            if len(data) > 0:
                sns.kdeplot(data, ax=ax1, label=sev.replace(' interaction', ''),
                           color=color, linewidth=2)
        
        ax1.set_xlabel('Severity Confidence')
        ax1.set_ylabel('Density')
        ax1.set_title('A. Confidence Distribution by Severity')
        ax1.legend()
        
        # Confidence vs numeric severity
        ax2 = axes[0, 1]
        ax2.scatter(self.df['severity_confidence'], self.df['severity_numeric'],
                   alpha=0.1, s=5)
        ax2.set_xlabel('Severity Confidence')
        ax2.set_ylabel('Severity Numeric')
        ax2.set_title('B. Confidence vs Numeric Severity')
        
        # Box plot of confidence by severity
        ax3 = axes[1, 0]
        self.df['severity_short'] = self.df['severity_label'].str.replace(' interaction', '')
        sns.boxplot(data=self.df, x='severity_short', y='severity_confidence',
                   ax=ax3, order=['Contraindicated', 'Major', 'Moderate', 'Minor'],
                   palette=colors)
        ax3.set_xlabel('Severity Level')
        ax3.set_ylabel('Confidence')
        ax3.set_title('C. Confidence Distribution by Severity')
        
        # High confidence interactions
        ax4 = axes[1, 1]
        high_conf = self.df[self.df['severity_confidence'] > 0.8]
        high_conf_counts = high_conf['severity_label'].value_counts()
        
        if len(high_conf_counts) > 0:
            ax4.bar(range(len(high_conf_counts)), high_conf_counts.values,
                   color=[colors[severity_order.index(s)] if s in severity_order else '#999999' 
                         for s in high_conf_counts.index])
            ax4.set_xticks(range(len(high_conf_counts)))
            ax4.set_xticklabels([s.replace(' interaction', '') for s in high_conf_counts.index],
                               rotation=45, ha='right')
            ax4.set_ylabel('Count')
            ax4.set_title('D. High Confidence (>0.8) Interactions')
        
        plt.tight_layout()
        self._save_fig('fig_adv09_confidence_analysis', plt)
    
    def _fig_atc_hierarchy(self, plt, sns):
        """ATC code hierarchy treemap-style visualization"""
        print("\n  → ATC Hierarchy Visualization")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Collect ATC data
        atc_l1_counts = defaultdict(int)
        atc_l2_counts = defaultdict(lambda: defaultdict(int))
        atc_pri = defaultdict(list)
        
        for name, node in self.network.nodes.items():
            if node.atc_code and len(node.atc_code) >= 1:
                l1 = node.atc_code[0]
                atc_l1_counts[l1] += 1
                atc_pri[l1].append(node.pri_score)
                
                if len(node.atc_code) >= 3:
                    l2 = node.atc_code[:3]
                    atc_l2_counts[l1][l2] += 1
        
        # Treemap-style nested bar chart
        ax1 = axes[0]
        
        atc_labels = {
            'A': 'Alimentary', 'B': 'Blood', 'C': 'Cardiovascular',
            'D': 'Dermatologicals', 'G': 'Genitourinary', 'H': 'Hormones',
            'J': 'Anti-infectives', 'L': 'Antineoplastic', 'M': 'Musculoskeletal',
            'N': 'Nervous System', 'P': 'Antiparasitic', 'R': 'Respiratory',
            'S': 'Sensory Organs', 'V': 'Various'
        }
        
        sorted_atc = sorted(atc_l1_counts.items(), key=lambda x: -x[1])[:10]
        codes = [c[0] for c in sorted_atc]
        counts = [c[1] for c in sorted_atc]
        avg_pris = [np.mean(atc_pri[c]) for c in codes]
        
        # Create grouped bar
        x = np.arange(len(codes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, counts, width, label='Drug Count', color='#1f77b4')
        ax1.set_ylabel('Drug Count', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, avg_pris, width, label='Avg PRI', color='#d62728')
        ax1_twin.set_ylabel('Average PRI Score', color='#d62728')
        ax1_twin.tick_params(axis='y', labelcolor='#d62728')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{c}\n({atc_labels.get(c, "")[:8]})' for c in codes], fontsize=9)
        ax1.set_title('A. Drug Count and Average PRI by ATC Level 1')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Stacked bar for top ATC codes with subgroups
        ax2 = axes[1]
        
        top_atc = sorted_atc[:5]
        colors = plt.cm.tab20.colors
        
        bottom = np.zeros(len(top_atc))
        color_idx = 0
        
        for atc_code, _ in top_atc:
            sub_counts = sorted(atc_l2_counts[atc_code].items(), 
                               key=lambda x: -x[1])[:5]
            for sub_code, sub_count in sub_counts:
                ax2.barh([atc_code], [sub_count], left=bottom[list(dict(top_atc).keys()).index(atc_code)],
                        color=colors[color_idx % len(colors)], label=sub_code, height=0.6)
                bottom[list(dict(top_atc).keys()).index(atc_code)] += sub_count
                color_idx += 1
        
        ax2.set_xlabel('Drug Count')
        ax2.set_ylabel('ATC Level 1')
        ax2.set_title('B. ATC Level 2 Breakdown (Top 5 per Level 1)')
        
        plt.tight_layout()
        self._save_fig('fig_adv10_atc_hierarchy', plt)
    
    def _fig_phenotype_cooccurrence(self, plt, sns):
        """Phenotype co-occurrence matrix"""
        print("\n  → Phenotype Co-occurrence Analysis")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Build co-occurrence matrix
        phenotypes = set()
        for edge in self.network.edges:
            phenotypes.update(edge.phenotypes)
        
        phenotypes = sorted(phenotypes)
        n = len(phenotypes)
        cooccur = np.zeros((n, n))
        
        for edge in self.network.edges:
            for i, p1 in enumerate(phenotypes):
                for j, p2 in enumerate(phenotypes):
                    if p1 in edge.phenotypes and p2 in edge.phenotypes:
                        cooccur[i, j] += 1
        
        # Normalize
        for i in range(n):
            if cooccur[i, i] > 0:
                cooccur[i, :] /= cooccur[i, i]
                cooccur[:, i] /= cooccur[i, i]
        
        np.fill_diagonal(cooccur, 1)
        
        # Heatmap
        ax1 = axes[0]
        sns.heatmap(cooccur, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[p.replace('_', '\n')[:10] for p in phenotypes],
                   yticklabels=[p.replace('_', ' ').title()[:12] for p in phenotypes],
                   ax=ax1, vmin=0, vmax=1)
        ax1.set_title('A. Phenotype Co-occurrence Matrix')
        
        # Phenotype network
        ax2 = axes[1]
        
        # Simple layout
        n_pheno = len(phenotypes)
        angles = np.linspace(0, 2*np.pi, n_pheno, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.8
        
        # Draw edges (co-occurrences)
        for i in range(n_pheno):
            for j in range(i+1, n_pheno):
                if cooccur[i, j] > 0.1:
                    alpha = min(0.8, cooccur[i, j])
                    ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            alpha=alpha, color='#666666', linewidth=cooccur[i, j] * 3)
        
        # Draw nodes
        phenotype_counts = defaultdict(int)
        for edge in self.network.edges:
            for p in edge.phenotypes:
                phenotype_counts[p] += 1
        
        sizes = [phenotype_counts[p] / 100 + 100 for p in phenotypes]
        ax2.scatter(pos[:, 0], pos[:, 1], s=sizes, c=plt.cm.tab10.colors[:n_pheno],
                   alpha=0.8, edgecolors='white', linewidths=2)
        
        # Labels
        for i, p in enumerate(phenotypes):
            offset = 1.15
            ax2.annotate(p.replace('_', ' ').title()[:15], 
                        (pos[i, 0] * offset, pos[i, 1] * offset),
                        ha='center', va='center', fontsize=9)
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.axis('off')
        ax2.set_title('B. Phenotype Co-occurrence Network')
        
        plt.tight_layout()
        self._save_fig('fig_adv11_phenotype_cooccurrence', plt)
    
    def _fig_centrality_comparison(self, plt, sns):
        """Comparison of different centrality measures"""
        print("\n  → Centrality Measures Comparison")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Collect data
        nodes = list(self.network.nodes.values())
        degree = [n.degree_centrality for n in nodes]
        weighted = [n.weighted_degree for n in nodes]
        betweenness = [n.betweenness_centrality for n in nodes]
        pri = [n.pri_score for n in nodes]
        
        # Degree vs Weighted
        ax1 = axes[0, 0]
        ax1.scatter(degree, weighted, alpha=0.3, s=10, c=pri, cmap='RdYlGn_r')
        ax1.set_xlabel('Degree Centrality')
        ax1.set_ylabel('Weighted Degree')
        ax1.set_title('A. Degree vs Weighted Degree')
        
        # Degree vs Betweenness
        ax2 = axes[0, 1]
        ax2.scatter(degree, betweenness, alpha=0.3, s=10, c=pri, cmap='RdYlGn_r')
        ax2.set_xlabel('Degree Centrality')
        ax2.set_ylabel('Betweenness Centrality')
        ax2.set_title('B. Degree vs Betweenness')
        
        # Weighted vs Betweenness
        ax3 = axes[0, 2]
        scatter = ax3.scatter(weighted, betweenness, alpha=0.3, s=10, c=pri, cmap='RdYlGn_r')
        ax3.set_xlabel('Weighted Degree')
        ax3.set_ylabel('Betweenness Centrality')
        ax3.set_title('C. Weighted vs Betweenness')
        plt.colorbar(scatter, ax=ax3, label='PRI Score')
        
        # Rank correlation between measures
        ax4 = axes[1, 0]
        
        from scipy.stats import spearmanr
        measures = ['Degree', 'Weighted', 'Betweenness', 'PRI']
        data = [degree, weighted, betweenness, pri]
        
        rank_corr = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                rank_corr[i, j], _ = spearmanr(data[i], data[j])
        
        sns.heatmap(rank_corr, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=measures, yticklabels=measures, ax=ax4,
                   vmin=-1, vmax=1, center=0)
        ax4.set_title('D. Spearman Rank Correlation')
        
        # Top 10 drugs by each measure
        ax5 = axes[1, 1]
        
        drug_names = list(self.network.nodes.keys())
        top_by_degree = sorted(range(len(degree)), key=lambda i: -degree[i])[:10]
        top_by_weighted = sorted(range(len(weighted)), key=lambda i: -weighted[i])[:10]
        top_by_betweenness = sorted(range(len(betweenness)), key=lambda i: -betweenness[i])[:10]
        top_by_pri = sorted(range(len(pri)), key=lambda i: -pri[i])[:10]
        
        # Jaccard similarity
        jaccard = np.zeros((4, 4))
        tops = [set(top_by_degree), set(top_by_weighted), 
                set(top_by_betweenness), set(top_by_pri)]
        
        for i in range(4):
            for j in range(4):
                intersection = len(tops[i] & tops[j])
                union = len(tops[i] | tops[j])
                jaccard[i, j] = intersection / union if union > 0 else 0
        
        sns.heatmap(jaccard, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=measures, yticklabels=measures, ax=ax5,
                   vmin=0, vmax=1)
        ax5.set_title('E. Top-10 Overlap (Jaccard Similarity)')
        
        # Distribution comparison
        ax6 = axes[1, 2]
        for i, (name, values) in enumerate(zip(measures, data)):
            sns.kdeplot(values, ax=ax6, label=name, linewidth=2)
        ax6.set_xlabel('Centrality Value (Normalized)')
        ax6.set_ylabel('Density')
        ax6.set_title('F. Centrality Distribution Comparison')
        ax6.legend()
        
        plt.tight_layout()
        self._save_fig('fig_adv12_centrality_comparison', plt)
    
    def _fig_recommendation_impact(self, plt, sns):
        """Impact analysis of drug recommendations"""
        print("\n  → Recommendation Impact Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Test multiple drug combinations
        test_combos = [
            ['warfarin', 'aspirin', 'metoprolol', 'clopidogrel'],
            ['digoxin', 'furosemide', 'spironolactone', 'carvedilol'],
            ['amlodipine', 'lisinopril', 'hydrochlorothiazide'],
            ['heparin', 'warfarin', 'aspirin'],
            ['atorvastatin', 'lisinopril', 'metformin']
        ]
        
        results = []
        for drugs in test_combos:
            valid_drugs = [d for d in drugs if d in self.network.nodes]
            if len(valid_drugs) >= 2:
                rec = self.recommender.recommend_for_polypharmacy(valid_drugs)
                
                original_risk = rec.get('overall_risk', {}).get('score', 0)
                risk_reduction = rec.get('summary', {}).get('estimated_risk_reduction', 0)
                
                results.append({
                    'combo': ', '.join([d.title()[:8] for d in valid_drugs[:3]]),
                    'n_drugs': len(valid_drugs),
                    'original_risk': original_risk,
                    'risk_reduction': risk_reduction,
                    'n_recommendations': len(rec.get('recommendations', []))
                })
        
        results_df = pd.DataFrame(results)
        
        # Original risk vs potential reduction
        ax1 = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, results_df['original_risk'], width, 
                       label='Original Risk', color='#d62728')
        bars2 = ax1.bar(x + width/2, results_df['original_risk'] - results_df['risk_reduction'], 
                       width, label='After Recommendations', color='#2ca02c')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['combo'], rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Risk Score')
        ax1.set_title('A. Risk Reduction from Recommendations')
        ax1.legend()
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        
        # Risk reduction percentage
        ax2 = axes[0, 1]
        reduction_pct = (results_df['risk_reduction'] / results_df['original_risk'] * 100).fillna(0)
        colors = ['#2ca02c' if r > 20 else '#ff7f0e' if r > 10 else '#d62728' for r in reduction_pct]
        ax2.bar(x, reduction_pct, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['combo'], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Risk Reduction (%)')
        ax2.set_title('B. Percentage Risk Reduction')
        ax2.axhline(20, color='green', linestyle='--', alpha=0.5)
        
        # Detailed analysis for one combination
        ax3 = axes[1, 0]
        
        test_drugs = ['warfarin', 'aspirin', 'metoprolol', 'clopidogrel']
        valid_drugs = [d for d in test_drugs if d in self.network.nodes]
        rec = self.recommender.recommend_for_polypharmacy(valid_drugs)
        
        if rec.get('recommendations'):
            drug_risks = []
            for r in rec['recommendations']:
                target = r.get('target_drug', '')
                contribution = r.get('risk_contribution', 0)
                best_alt = r.get('best_alternative', {})
                reduction = best_alt.get('risk_metrics', {}).get('pri_reduction', 0) if best_alt else 0
                drug_risks.append({
                    'drug': target,
                    'contribution': contribution,
                    'reduction': reduction
                })
            
            dr_df = pd.DataFrame(drug_risks)
            
            x2 = np.arange(len(dr_df))
            ax3.bar(x2 - 0.2, dr_df['contribution'], 0.4, label='Risk Contribution', color='#d62728')
            ax3.bar(x2 + 0.2, dr_df['reduction'], 0.4, label='Potential Reduction', color='#2ca02c')
            ax3.set_xticks(x2)
            ax3.set_xticklabels(dr_df['drug'], rotation=45, ha='right')
            ax3.set_ylabel('Score')
            ax3.set_title('C. Drug-Level Recommendation Impact\n(Warfarin, Aspirin, Metoprolol, Clopidogrel)')
            ax3.legend()
        
        # Multi-objective score components
        ax4 = axes[1, 1]
        
        if rec.get('recommendations'):
            obj_data = []
            for r in rec['recommendations']:
                best = r.get('best_alternative', {})
                if best:
                    obj_data.append({
                        'target': r.get('target_drug', ''),
                        'PRI Reduction': best.get('risk_metrics', {}).get('pri_reduction', 0) * 0.35,
                        'Centrality Red.': best.get('risk_metrics', {}).get('centrality_reduction', 0) * 0.20,
                        'Phenotype Score': 0.1,  # Placeholder
                        'Interaction Pen.': 0.15  # Placeholder
                    })
            
            if obj_data:
                obj_df = pd.DataFrame(obj_data)
                obj_df.set_index('target', inplace=True)
                obj_df.plot(kind='bar', stacked=True, ax=ax4, 
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax4.set_xlabel('Target Drug')
                ax4.set_ylabel('Multi-Objective Score Components')
                ax4.set_title('D. Multi-Objective Score Breakdown')
                ax4.legend(loc='upper right', fontsize=9)
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_fig('fig_adv13_recommendation_impact', plt)
    
    def _fig_network_robustness(self, plt, sns):
        """Network robustness analysis"""
        print("\n  → Network Robustness Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Simulate node removal and measure connectivity
        nodes_by_pri = sorted(self.network.nodes.keys(), 
                             key=lambda x: -self.network.nodes[x].pri_score)
        nodes_by_degree = sorted(self.network.nodes.keys(),
                                key=lambda x: -self.network.nodes[x].degree_centrality)
        
        n_nodes = len(nodes_by_pri)
        removal_fractions = np.linspace(0, 0.3, 20)
        
        # Calculate giant component size after removal
        def calc_connectivity(remaining_nodes):
            if len(remaining_nodes) < 2:
                return 0
            # Simple connectivity: average degree of remaining
            total_edges = 0
            remaining_list = list(remaining_nodes)[:500]  # Sample for speed
            for node in remaining_list:
                for neighbor in self.network.drug_interactions.get(node, set()):
                    if neighbor in remaining_nodes:
                        total_edges += 1
            return total_edges / len(remaining_nodes) if remaining_nodes else 0
        
        original_connectivity = calc_connectivity(set(nodes_by_pri))
        if original_connectivity == 0:
            original_connectivity = 1  # Avoid division by zero
        
        pri_removal = []
        degree_removal = []
        random_removal = []
        
        for frac in removal_fractions:
            n_remove = int(frac * n_nodes)
            
            # PRI-based removal
            remaining = set(nodes_by_pri[n_remove:])
            pri_removal.append(calc_connectivity(remaining) / original_connectivity)
            
            # Degree-based removal
            remaining = set(nodes_by_degree[n_remove:])
            degree_removal.append(calc_connectivity(remaining) / original_connectivity)
            
            # Random removal
            np.random.seed(42)
            random_remove = np.random.choice(list(self.network.nodes.keys()), 
                                            n_remove, replace=False)
            remaining = set(self.network.nodes.keys()) - set(random_remove)
            random_removal.append(calc_connectivity(remaining) / original_connectivity)
        
        # Robustness curves
        ax1 = axes[0, 0]
        ax1.plot(removal_fractions * 100, pri_removal, 'r-', linewidth=2, label='PRI-based')
        ax1.plot(removal_fractions * 100, degree_removal, 'b-', linewidth=2, label='Degree-based')
        ax1.plot(removal_fractions * 100, random_removal, 'g--', linewidth=2, label='Random')
        ax1.set_xlabel('Nodes Removed (%)')
        ax1.set_ylabel('Relative Connectivity')
        ax1.set_title('A. Network Robustness Under Node Removal')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Attack vs failure
        ax2 = axes[0, 1]
        attack_efficiency = [1 - p for p in pri_removal]
        failure_efficiency = [1 - r for r in random_removal]
        
        ax2.fill_between(removal_fractions * 100, attack_efficiency, alpha=0.3, color='red', label='Targeted Attack')
        ax2.fill_between(removal_fractions * 100, failure_efficiency, alpha=0.3, color='green', label='Random Failure')
        ax2.set_xlabel('Nodes Removed (%)')
        ax2.set_ylabel('Network Disruption')
        ax2.set_title('B. Attack vs Random Failure')
        ax2.legend()
        
        # Degree distribution resilience
        ax3 = axes[1, 0]
        degrees = [len(self.network.drug_interactions.get(d, set())) 
                  for d in self.network.nodes]
        
        # Remove high-degree nodes
        high_degree_threshold = np.percentile(degrees, 90)
        remaining_after_attack = [d for d, deg in zip(self.network.nodes.keys(), degrees) 
                                 if deg < high_degree_threshold]
        
        original_degrees = degrees
        attacked_degrees = [len(self.network.drug_interactions.get(d, set()) & 
                               set(remaining_after_attack))
                          for d in remaining_after_attack]
        
        sns.histplot(original_degrees, bins=50, ax=ax3, alpha=0.5, label='Original', color='blue')
        sns.histplot(attacked_degrees, bins=50, ax=ax3, alpha=0.5, label='After Attack', color='red')
        ax3.set_xlabel('Degree')
        ax3.set_ylabel('Count')
        ax3.set_title('C. Degree Distribution: Original vs After Attack')
        ax3.legend()
        
        # Critical nodes analysis
        ax4 = axes[1, 1]
        
        # Calculate impact of removing each top node
        top_nodes = nodes_by_pri[:20]
        impacts = []
        
        for node in top_nodes:
            # Impact = sum of degrees of neighbors
            impact = sum(len(self.network.drug_interactions.get(n, set())) 
                        for n in self.network.drug_interactions.get(node, set()))
            impacts.append({
                'node': node,
                'pri': self.network.nodes[node].pri_score,
                'degree': self.network.nodes[node].degree_centrality,
                'impact': impact
            })
        
        impact_df = pd.DataFrame(impacts)
        
        ax4.scatter(impact_df['degree'], impact_df['impact'], 
                   s=impact_df['pri'] * 500, alpha=0.6, c=impact_df['pri'], cmap='Reds')
        ax4.set_xlabel('Degree Centrality')
        ax4.set_ylabel('Removal Impact (Neighbor Degrees)')
        ax4.set_title('D. Critical Node Impact Analysis')
        
        for _, row in impact_df.head(5).iterrows():
            ax4.annotate(row['node'].title()[:10], (row['degree'], row['impact']),
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        self._save_fig('fig_adv14_network_robustness', plt)
    
    def _fig_pri_heatmap_detailed(self, plt, sns):
        """Detailed PRI heatmap for drug pairs"""
        print("\n  → Detailed PRI Heatmap")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get top drugs by different criteria
        top_cardio = [name for name, node in sorted(self.network.nodes.items(), 
                     key=lambda x: -x[1].pri_score) if node.is_cardiovascular][:15]
        # Get top non-cardiovascular drugs for comparison
        top_other = [name for name, node in sorted(self.network.nodes.items(),
                   key=lambda x: -x[1].pri_score) if not node.is_cardiovascular][:15]
        
        # Cardiovascular drugs heatmap
        ax1 = axes[0]
        
        if len(top_cardio) >= 2:
            n = len(top_cardio)
            matrix = np.zeros((n, n))
            
            for i, d1 in enumerate(top_cardio):
                for j, d2 in enumerate(top_cardio):
                    if i != j:
                        edge = self.network.adjacency.get(d1, {}).get(d2)
                        if edge:
                            matrix[i, j] = edge.severity_weight
            
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            sns.heatmap(matrix, mask=mask, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=[d.title()[:12] for d in top_cardio],
                       yticklabels=[d.title()[:12] for d in top_cardio],
                       ax=ax1, cbar_kws={'label': 'Severity Weight'})
            ax1.set_title('A. DDI Severity: Top Cardiovascular Drugs')
        
        # Non-cardiovascular drugs heatmap
        ax2 = axes[1]
        
        if len(top_other) >= 2:
            n = len(top_other)
            matrix = np.zeros((n, n))
            
            for i, d1 in enumerate(top_other):
                for j, d2 in enumerate(top_other):
                    if i != j:
                        edge = self.network.adjacency.get(d1, {}).get(d2)
                        if edge:
                            matrix[i, j] = edge.severity_weight
            
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            sns.heatmap(matrix, mask=mask, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=[d.title()[:12] for d in top_other],
                       yticklabels=[d.title()[:12] for d in top_other],
                       ax=ax2, cbar_kws={'label': 'Severity Weight'})
            ax2.set_title('B. DDI Severity: Top Non-Cardiovascular Drugs')
        
        plt.tight_layout()
        self._save_fig('fig_adv15_pri_heatmap_detailed', plt)


def main():
    generator = AdvancedFigureGenerator()
    generator.generate_all()


if __name__ == '__main__':
    main()
