#!/usr/bin/env python3
"""
Publication Materials Generator
Generates all figures, tables, and data for the paper:
"AI-based Polypharmacy Risk-aware Drug Recommender System"

Output Structure:
- publication/figures/    - All publication-ready figures (PNG, PDF, SVG)
- publication/tables/     - All tables (CSV, LaTeX)
- publication/data/       - Processed data files
- publication/statistics/ - Statistical analysis results
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


class PublicationGenerator:
    """Generate all publication materials"""
    
    def __init__(self, data_path: str = None, output_dir: str = "publication"):
        self.data_path = data_path or self._find_data()
        self.output_dir = Path(output_dir)
        self.df = None
        self.network = None
        self.recommender = None
        
        # Create output directories
        self.dirs = {
            'figures': self.output_dir / 'figures',
            'tables': self.output_dir / 'tables',
            'data': self.output_dir / 'data',
            'statistics': self.output_dir / 'statistics',
            'supplementary': self.output_dir / 'supplementary'
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    
    def _find_data(self) -> str:
        """Find the DDI data file"""
        possible = [
            'data/ddi_cardio_or_antithrombotic_labeled (1).csv',
            'data/ddi_cardio_or_antithrombotic_labeled.csv',
            'ddi_cardio_or_antithrombotic_labeled (1).csv',
            'ddi_cardio_or_antithrombotic_labeled.csv'
        ]
        for name in possible:
            if Path(name).exists():
                return name
        raise FileNotFoundError("DDI data file not found")
    
    def load_data(self):
        """Load and preprocess data"""
        print("=" * 70)
        print("📊 PUBLICATION MATERIALS GENERATOR")
        print("   AI-based Polypharmacy Risk-aware Drug Recommender System")
        print("=" * 70)
        print(f"\n📂 Loading data from: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"   ✓ Loaded {len(self.df):,} drug-drug interactions")
        
        # Initialize network
        print("\n🔗 Building Drug Risk Network...")
        from agents import DrugRiskNetwork, MultiObjectiveRecommender
        self.network = DrugRiskNetwork()
        self.network.build_network(self.df)
        self.recommender = MultiObjectiveRecommender(self.network)
        
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
    
    def generate_all(self):
        """Generate all publication materials"""
        self.load_data()
        
        print("\n" + "=" * 70)
        print("📈 GENERATING FIGURES")
        print("=" * 70)
        self.generate_figures()
        
        print("\n" + "=" * 70)
        print("📋 GENERATING TABLES")
        print("=" * 70)
        self.generate_tables()
        
        print("\n" + "=" * 70)
        print("📊 GENERATING DATA FILES")
        print("=" * 70)
        self.generate_data()
        
        print("\n" + "=" * 70)
        print("📉 GENERATING STATISTICS")
        print("=" * 70)
        self.generate_statistics()
        
        print("\n" + "=" * 70)
        print("✅ GENERATION COMPLETE")
        print("=" * 70)
        self.print_summary()
    
    # =========================================================================
    # FIGURES
    # =========================================================================
    
    def generate_figures(self):
        """Generate all publication figures"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set publication style
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'font.family': 'sans-serif'
            })
            
            self._fig1_severity_distribution(plt, sns)
            self._fig2_network_metrics(plt, sns)
            self._fig3_pri_distribution(plt, sns)
            self._fig4_atc_analysis(plt, sns)
            self._fig5_interaction_heatmap(plt, sns)
            self._fig6_risk_correlation(plt, sns)
            self._fig7_recommender_performance(plt, sns)
            self._fig8_phenotype_analysis(plt, sns)
            
            plt.close('all')
            
        except ImportError as e:
            print(f"⚠️ Matplotlib/Seaborn not available: {e}")
            print("   Generating data-only outputs...")
    
    def _fig1_severity_distribution(self, plt, sns):
        """Figure 1: DDI Severity Distribution"""
        print("\n  → Figure 1: DDI Severity Distribution")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        severity_counts = self.df['severity_label'].value_counts()
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        severity_order = ['Contraindicated interaction', 'Major interaction', 
                         'Moderate interaction', 'Minor interaction']
        
        ax1 = axes[0]
        bars = ax1.bar(range(len(severity_counts)), 
                      [severity_counts.get(s, 0) for s in severity_order],
                      color=colors)
        ax1.set_xticks(range(len(severity_order)))
        ax1.set_xticklabels(['Contraindicated', 'Major', 'Moderate', 'Minor'], rotation=45, ha='right')
        ax1.set_ylabel('Number of Interactions')
        ax1.set_xlabel('Severity Level')
        ax1.set_title('A. Distribution of DDI Severity Levels')
        
        # Add value labels
        for bar, count in zip(bars, [severity_counts.get(s, 0) for s in severity_order]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        ax2 = axes[1]
        percentages = [severity_counts.get(s, 0) for s in severity_order]
        ax2.pie(percentages, labels=['Contraindicated', 'Major', 'Moderate', 'Minor'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('B. Proportion of DDI Severity Levels')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig1_severity_distribution.{fmt}')
        print(f"     ✓ Saved fig1_severity_distribution.[png/pdf/svg]")
    
    def _fig2_network_metrics(self, plt, sns):
        """Figure 2: Network Centrality Metrics"""
        print("\n  → Figure 2: Network Centrality Metrics")
        
        # Collect metrics for all drugs
        metrics = []
        for drug_name, node in self.network.nodes.items():
            metrics.append({
                'drug': drug_name,
                'degree_centrality': node.degree_centrality,
                'weighted_degree': node.weighted_degree,
                'betweenness_centrality': node.betweenness_centrality,
                'pri_score': node.pri_score
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Degree centrality distribution
        ax1 = axes[0, 0]
        sns.histplot(metrics_df['degree_centrality'], bins=50, ax=ax1, color='#1f77b4')
        ax1.set_xlabel('Degree Centrality')
        ax1.set_ylabel('Number of Drugs')
        ax1.set_title('A. Degree Centrality Distribution')
        ax1.axvline(metrics_df['degree_centrality'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {metrics_df["degree_centrality"].mean():.3f}')
        ax1.legend()
        
        # Weighted degree distribution
        ax2 = axes[0, 1]
        sns.histplot(metrics_df['weighted_degree'], bins=50, ax=ax2, color='#ff7f0e')
        ax2.set_xlabel('Weighted Degree (Normalized)')
        ax2.set_ylabel('Number of Drugs')
        ax2.set_title('B. Weighted Degree Distribution')
        ax2.axvline(metrics_df['weighted_degree'].mean(), color='red', linestyle='--',
                   label=f'Mean: {metrics_df["weighted_degree"].mean():.3f}')
        ax2.legend()
        
        # Betweenness centrality distribution
        ax3 = axes[1, 0]
        sns.histplot(metrics_df['betweenness_centrality'], bins=50, ax=ax3, color='#2ca02c')
        ax3.set_xlabel('Betweenness Centrality')
        ax3.set_ylabel('Number of Drugs')
        ax3.set_title('C. Betweenness Centrality Distribution')
        ax3.axvline(metrics_df['betweenness_centrality'].mean(), color='red', linestyle='--',
                   label=f'Mean: {metrics_df["betweenness_centrality"].mean():.3f}')
        ax3.legend()
        
        # PRI score distribution
        ax4 = axes[1, 1]
        sns.histplot(metrics_df['pri_score'], bins=50, ax=ax4, color='#d62728')
        ax4.set_xlabel('Polypharmacy Risk Index (PRI)')
        ax4.set_ylabel('Number of Drugs')
        ax4.set_title('D. PRI Score Distribution')
        ax4.axvline(metrics_df['pri_score'].mean(), color='blue', linestyle='--',
                   label=f'Mean: {metrics_df["pri_score"].mean():.3f}')
        ax4.legend()
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig2_network_metrics.{fmt}')
        print(f"     ✓ Saved fig2_network_metrics.[png/pdf/svg]")
    
    def _fig3_pri_distribution(self, plt, sns):
        """Figure 3: PRI Analysis by Drug Category"""
        print("\n  → Figure 3: PRI by Drug Category")
        
        # Categorize drugs (cardiovascular includes former antithrombotic)
        categories = []
        for drug_name, node in self.network.nodes.items():
            if node.is_cardiovascular:
                cat = 'Cardiovascular'
            else:
                cat = 'Other'
            categories.append({
                'drug': drug_name,
                'category': cat,
                'pri_score': node.pri_score,
                'degree_centrality': node.degree_centrality
            })
        
        cat_df = pd.DataFrame(categories)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        ax1 = axes[0]
        order = ['Cardiovascular', 'Other']
        colors = {'Cardiovascular': '#e41a1c', 'Other': '#999999'}
        sns.boxplot(data=cat_df, x='category', y='pri_score', ax=ax1, 
                   order=order, palette=colors)
        ax1.set_xlabel('Drug Category')
        ax1.set_ylabel('Polypharmacy Risk Index (PRI)')
        ax1.set_title('A. PRI Distribution by Drug Category')
        
        # Violin plot
        ax2 = axes[1]
        sns.violinplot(data=cat_df, x='category', y='degree_centrality', ax=ax2,
                      order=order, palette=colors)
        ax2.set_xlabel('Drug Category')
        ax2.set_ylabel('Degree Centrality')
        ax2.set_title('B. Degree Centrality by Drug Category')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig3_pri_by_category.{fmt}')
        print(f"     ✓ Saved fig3_pri_by_category.[png/pdf/svg]")
    
    def _fig4_atc_analysis(self, plt, sns):
        """Figure 4: ATC Classification Analysis"""
        print("\n  → Figure 4: ATC Classification Analysis")
        
        # Count ATC level 1 codes
        atc_counts = defaultdict(int)
        atc_labels = {
            'A': 'Alimentary',
            'B': 'Blood/Hematopoietic',
            'C': 'Cardiovascular',
            'D': 'Dermatologicals',
            'G': 'Genitourinary',
            'H': 'Hormones',
            'J': 'Anti-infectives',
            'L': 'Antineoplastic',
            'M': 'Musculoskeletal',
            'N': 'Nervous System',
            'P': 'Antiparasitic',
            'R': 'Respiratory',
            'S': 'Sensory Organs',
            'V': 'Various'
        }
        
        for drug_name, node in self.network.nodes.items():
            if node.atc_code:
                atc_l1 = node.atc_code[0] if node.atc_code else 'Unknown'
                atc_counts[atc_l1] += 1
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of ATC level 1
        ax1 = axes[0]
        sorted_atc = sorted(atc_counts.items(), key=lambda x: -x[1])[:10]
        labels = [f"{code}\n({atc_labels.get(code, 'Unknown')[:10]})" for code, _ in sorted_atc]
        values = [v for _, v in sorted_atc]
        
        bars = ax1.bar(labels, values, color=plt.cm.tab10.colors[:len(labels)])
        ax1.set_xlabel('ATC Level 1 Code')
        ax1.set_ylabel('Number of Drugs')
        ax1.set_title('A. Drug Distribution by ATC Classification')
        ax1.tick_params(axis='x', rotation=45)
        
        # PRI by top ATC codes
        ax2 = axes[1]
        atc_pri = defaultdict(list)
        for drug_name, node in self.network.nodes.items():
            if node.atc_code:
                atc_l1 = node.atc_code[0]
                atc_pri[atc_l1].append(node.pri_score)
        
        top_atc = [code for code, _ in sorted_atc[:8]]
        pri_data = [atc_pri[code] for code in top_atc]
        
        bp = ax2.boxplot(pri_data, labels=top_atc, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.tab10.colors):
            patch.set_facecolor(color)
        ax2.set_xlabel('ATC Level 1 Code')
        ax2.set_ylabel('PRI Score')
        ax2.set_title('B. PRI Distribution by ATC Classification')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig4_atc_analysis.{fmt}')
        print(f"     ✓ Saved fig4_atc_analysis.[png/pdf/svg]")
    
    def _fig5_interaction_heatmap(self, plt, sns):
        """Figure 5: Drug-Drug Interaction Heatmap (Top Drugs)"""
        print("\n  → Figure 5: DDI Heatmap (Top Risk Drugs)")
        
        # Get top 20 drugs by PRI
        top_drugs = sorted(self.network.nodes.items(), 
                          key=lambda x: -x[1].pri_score)[:20]
        drug_names = [d[0] for d in top_drugs]
        
        # Build interaction matrix
        n = len(drug_names)
        matrix = np.zeros((n, n))
        
        for i, d1 in enumerate(drug_names):
            for j, d2 in enumerate(drug_names):
                if i != j:
                    edge = self.network.adjacency.get(d1, {}).get(d2)
                    if edge:
                        matrix[i, j] = edge.severity_weight
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        sns.heatmap(matrix, mask=mask, annot=False, cmap=cmap,
                   xticklabels=[d.title()[:15] for d in drug_names],
                   yticklabels=[d.title()[:15] for d in drug_names],
                   ax=ax, cbar_kws={'label': 'Severity Weight'})
        
        ax.set_title('Drug-Drug Interaction Severity Matrix\n(Top 20 Highest-Risk Drugs)')
        ax.set_xlabel('Drug')
        ax.set_ylabel('Drug')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig5_interaction_heatmap.{fmt}')
        print(f"     ✓ Saved fig5_interaction_heatmap.[png/pdf/svg]")
    
    def _fig6_risk_correlation(self, plt, sns):
        """Figure 6: Correlation Between Risk Metrics"""
        print("\n  → Figure 6: Risk Metrics Correlation")
        
        # Collect metrics
        metrics_data = []
        for node in self.network.nodes.values():
            metrics_data.append({
                'Degree Centrality': node.degree_centrality,
                'Weighted Degree': node.weighted_degree,
                'Betweenness': node.betweenness_centrality,
                'PRI Score': node.pri_score,
                'Contraindicated': node.contraindicated_count,
                'Major': node.major_count
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Correlation matrix
        ax1 = axes[0]
        corr = metrics_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   ax=ax1, square=True)
        ax1.set_title('A. Correlation Between Risk Metrics')
        
        # Scatter: Degree vs PRI
        ax2 = axes[1]
        ax2.scatter(metrics_df['Degree Centrality'], metrics_df['PRI Score'], 
                   alpha=0.5, c=metrics_df['Weighted Degree'], cmap='viridis')
        ax2.set_xlabel('Degree Centrality')
        ax2.set_ylabel('PRI Score')
        ax2.set_title('B. Degree Centrality vs PRI Score')
        
        # Add regression line
        z = np.polyfit(metrics_df['Degree Centrality'], metrics_df['PRI Score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(metrics_df['Degree Centrality'].min(), 
                            metrics_df['Degree Centrality'].max(), 100)
        ax2.plot(x_line, p(x_line), 'r--', label=f'R²={corr.loc["Degree Centrality","PRI Score"]**2:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig6_risk_correlation.{fmt}')
        print(f"     ✓ Saved fig6_risk_correlation.[png/pdf/svg]")
    
    def _fig7_recommender_performance(self, plt, sns):
        """Figure 7: Multi-Objective Recommender Performance"""
        print("\n  → Figure 7: Recommender Performance Analysis")
        
        # Test with sample drug combinations
        test_cases = [
            ['warfarin', 'aspirin', 'metoprolol', 'clopidogrel'],
            ['digoxin', 'furosemide', 'spironolactone', 'carvedilol'],
            ['amlodipine', 'lisinopril', 'hydrochlorothiazide', 'metoprolol'],
            ['heparin', 'warfarin', 'aspirin'],
            ['atorvastatin', 'lisinopril', 'metformin', 'aspirin']
        ]
        
        results = []
        for drugs in test_cases:
            valid_drugs = [d for d in drugs if d in self.network.nodes]
            if len(valid_drugs) >= 2:
                rec = self.recommender.recommend_for_polypharmacy(valid_drugs)
                results.append({
                    'drug_count': len(valid_drugs),
                    'risk_score': rec.get('overall_risk', {}).get('score', 0),
                    'risk_level': rec.get('overall_risk', {}).get('level', 'N/A'),
                    'recommendations': len(rec.get('recommendations', [])),
                    'risk_reduction': rec.get('summary', {}).get('estimated_risk_reduction', 0)
                })
        
        if not results:
            print("     ⚠️ No valid test cases for recommender evaluation")
            return
        
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Risk scores by drug count
        ax1 = axes[0]
        ax1.bar(range(len(results_df)), results_df['risk_score'], 
               color=['#d62728' if r == 'CRITICAL' else '#ff7f0e' if r == 'HIGH' else '#2ca02c' 
                     for r in results_df['risk_level']])
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels([f'Case {i+1}\n({r["drug_count"]} drugs)' 
                           for i, r in results_df.iterrows()])
        ax1.set_ylabel('Risk Score')
        ax1.set_title('A. Polypharmacy Risk Scores by Test Case')
        ax1.axhline(70, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        ax1.axhline(50, color='orange', linestyle='--', alpha=0.5, label='High threshold')
        ax1.legend()
        
        # Risk reduction potential
        ax2 = axes[1]
        ax2.bar(range(len(results_df)), results_df['risk_reduction'], color='#2ca02c')
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels([f'Case {i+1}' for i in range(len(results_df))])
        ax2.set_ylabel('Estimated Risk Reduction')
        ax2.set_title('B. Potential Risk Reduction from Recommendations')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig7_recommender_performance.{fmt}')
        print(f"     ✓ Saved fig7_recommender_performance.[png/pdf/svg]")
    
    def _fig8_phenotype_analysis(self, plt, sns):
        """Figure 8: Interaction Phenotype Analysis"""
        print("\n  → Figure 8: Interaction Phenotype Analysis")
        
        # Count phenotypes
        phenotype_counts = defaultdict(int)
        for edge in self.network.edges:
            for phenotype in edge.phenotypes:
                phenotype_counts[phenotype] += 1
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of phenotypes
        ax1 = axes[0]
        sorted_pheno = sorted(phenotype_counts.items(), key=lambda x: -x[1])
        if sorted_pheno:
            labels, values = zip(*sorted_pheno)
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(labels)))
            ax1.barh(range(len(labels)), values, color=colors)
            ax1.set_yticks(range(len(labels)))
            ax1.set_yticklabels([l.replace('_', ' ').title() for l in labels])
            ax1.set_xlabel('Number of Interactions')
            ax1.set_title('A. DDI Phenotype Frequency')
            ax1.invert_yaxis()
        
        # Phenotype by severity
        ax2 = axes[1]
        pheno_severity = defaultdict(lambda: defaultdict(int))
        for edge in self.network.edges:
            for phenotype in edge.phenotypes:
                pheno_severity[phenotype][edge.severity_label] += 1
        
        if pheno_severity:
            top_phenos = [p for p, _ in sorted_pheno[:6]]
            severity_order = ['Contraindicated interaction', 'Major interaction', 
                            'Moderate interaction', 'Minor interaction']
            
            x = np.arange(len(top_phenos))
            width = 0.2
            
            for i, severity in enumerate(severity_order):
                values = [pheno_severity[p][severity] for p in top_phenos]
                ax2.bar(x + i*width, values, width, 
                       label=severity.replace(' interaction', ''),
                       color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'][i])
            
            ax2.set_xticks(x + width * 1.5)
            ax2.set_xticklabels([p.replace('_', ' ').title()[:12] for p in top_phenos], rotation=45, ha='right')
            ax2.set_ylabel('Number of Interactions')
            ax2.set_title('B. Phenotype Distribution by Severity')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(self.dirs['figures'] / f'fig8_phenotype_analysis.{fmt}')
        print(f"     ✓ Saved fig8_phenotype_analysis.[png/pdf/svg]")
    
    # =========================================================================
    # TABLES
    # =========================================================================
    
    def generate_tables(self):
        """Generate all publication tables"""
        self._table1_dataset_summary()
        self._table2_severity_statistics()
        self._table3_top_risk_drugs()
        self._table4_network_statistics()
        self._table5_atc_distribution()
        self._table6_phenotype_summary()
        self._table7_sample_recommendations()
    
    def _save_table(self, df: pd.DataFrame, name: str, caption: str = ""):
        """Save table in multiple formats"""
        # CSV
        df.to_csv(self.dirs['tables'] / f'{name}.csv', index=False)
        
        # LaTeX (manual generation to avoid jinja2 dependency)
        latex_lines = [
            f"\\begin{{table}}[htbp]",
            f"\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:{name}}}",
            f"\\begin{{tabular}}{{{'l' * len(df.columns)}}}",
            "\\toprule",
            " & ".join(df.columns) + " \\\\",
            "\\midrule"
        ]
        for _, row in df.iterrows():
            latex_lines.append(" & ".join(str(v) for v in row.values) + " \\\\")
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        with open(self.dirs['tables'] / f'{name}.tex', 'w') as f:
            f.write('\n'.join(latex_lines))
        
        # Markdown
        try:
            md = df.to_markdown(index=False)
        except ImportError:
            # Fallback if tabulate not installed
            md = df.to_string(index=False)
        with open(self.dirs['tables'] / f'{name}.md', 'w') as f:
            f.write(f"# {caption}\n\n{md}")
        
        print(f"     ✓ Saved {name}.[csv/tex/md]")
    
    def _table1_dataset_summary(self):
        """Table 1: Dataset Summary Statistics"""
        print("\n  → Table 1: Dataset Summary")
        
        unique_drugs_1 = self.df['drug_name_1'].nunique()
        unique_drugs_2 = self.df['drug_name_2'].nunique()
        
        data = {
            'Metric': [
                'Total DDI Records',
                'Unique Drug Pairs',
                'Unique Drugs (Drug 1)',
                'Unique Drugs (Drug 2)',
                'Total Unique Drugs (Network)',
                'Network Edges',
                'Cardiovascular Drugs',
                'Non-Cardiovascular Drugs'
            ],
            'Value': [
                f"{len(self.df):,}",
                f"{len(self.network.edges):,}",
                f"{unique_drugs_1:,}",
                f"{unique_drugs_2:,}",
                f"{len(self.network.nodes):,}",
                f"{len(self.network.edges):,}",
                f"{sum(1 for n in self.network.nodes.values() if n.is_cardiovascular):,}",
                f"{sum(1 for n in self.network.nodes.values() if not n.is_cardiovascular):,}"
            ]
        }
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table1_dataset_summary', 
                        'Dataset Summary Statistics')
    
    def _table2_severity_statistics(self):
        """Table 2: DDI Severity Statistics"""
        print("\n  → Table 2: Severity Statistics")
        
        severity_counts = self.df['severity_label'].value_counts()
        total = len(self.df)
        
        data = {
            'Severity Level': [],
            'Count': [],
            'Percentage': [],
            'Severity Weight': []
        }
        
        for severity in ['Contraindicated interaction', 'Major interaction', 
                        'Moderate interaction', 'Minor interaction']:
            count = severity_counts.get(severity, 0)
            data['Severity Level'].append(severity.replace(' interaction', ''))
            data['Count'].append(f"{count:,}")
            data['Percentage'].append(f"{count/total*100:.2f}%")
            data['Severity Weight'].append(self.network.SEVERITY_WEIGHTS.get(severity, 0))
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table2_severity_statistics',
                        'Distribution of DDI Severity Levels')
    
    def _table3_top_risk_drugs(self):
        """Table 3: Top 20 Highest Risk Drugs"""
        print("\n  → Table 3: Top Risk Drugs")
        
        top_drugs = sorted(self.network.nodes.items(), 
                          key=lambda x: -x[1].pri_score)[:20]
        
        data = {
            'Rank': [],
            'Drug Name': [],
            'ATC Code': [],
            'PRI Score': [],
            'Degree Centrality': [],
            'Weighted Degree': [],
            'Contraindicated': [],
            'Major': []
        }
        
        for i, (name, node) in enumerate(top_drugs, 1):
            data['Rank'].append(i)
            data['Drug Name'].append(name.title())
            data['ATC Code'].append(node.atc_code or 'N/A')
            data['PRI Score'].append(f"{node.pri_score:.4f}")
            data['Degree Centrality'].append(f"{node.degree_centrality:.4f}")
            data['Weighted Degree'].append(f"{node.weighted_degree:.4f}")
            data['Contraindicated'].append(node.contraindicated_count)
            data['Major'].append(node.major_count)
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table3_top_risk_drugs',
                        'Top 20 Drugs by Polypharmacy Risk Index (PRI)')
    
    def _table4_network_statistics(self):
        """Table 4: Network Statistics"""
        print("\n  → Table 4: Network Statistics")
        
        nodes = list(self.network.nodes.values())
        
        data = {
            'Metric': [
                'Number of Nodes',
                'Number of Edges',
                'Network Density',
                'Mean Degree Centrality',
                'Max Degree Centrality',
                'Mean Weighted Degree',
                'Max Weighted Degree',
                'Mean Betweenness Centrality',
                'Max Betweenness Centrality',
                'Mean PRI Score',
                'Max PRI Score',
                'Std PRI Score'
            ],
            'Value': [
                f"{len(self.network.nodes):,}",
                f"{len(self.network.edges):,}",
                f"{2*len(self.network.edges)/(len(self.network.nodes)*(len(self.network.nodes)-1)):.6f}",
                f"{np.mean([n.degree_centrality for n in nodes]):.4f}",
                f"{max(n.degree_centrality for n in nodes):.4f}",
                f"{np.mean([n.weighted_degree for n in nodes]):.4f}",
                f"{max(n.weighted_degree for n in nodes):.4f}",
                f"{np.mean([n.betweenness_centrality for n in nodes]):.4f}",
                f"{max(n.betweenness_centrality for n in nodes):.4f}",
                f"{np.mean([n.pri_score for n in nodes]):.4f}",
                f"{max(n.pri_score for n in nodes):.4f}",
                f"{np.std([n.pri_score for n in nodes]):.4f}"
            ]
        }
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table4_network_statistics',
                        'Drug Risk Network Statistics')
    
    def _table5_atc_distribution(self):
        """Table 5: ATC Classification Distribution"""
        print("\n  → Table 5: ATC Distribution")
        
        atc_labels = {
            'A': 'Alimentary tract and metabolism',
            'B': 'Blood and blood forming organs',
            'C': 'Cardiovascular system',
            'D': 'Dermatologicals',
            'G': 'Genitourinary system',
            'H': 'Systemic hormones',
            'J': 'Anti-infectives for systemic use',
            'L': 'Antineoplastic agents',
            'M': 'Musculoskeletal system',
            'N': 'Nervous system',
            'P': 'Antiparasitic products',
            'R': 'Respiratory system',
            'S': 'Sensory organs',
            'V': 'Various'
        }
        
        atc_counts = defaultdict(int)
        atc_pri = defaultdict(list)
        
        for node in self.network.nodes.values():
            if node.atc_code:
                atc_l1 = node.atc_code[0]
                atc_counts[atc_l1] += 1
                atc_pri[atc_l1].append(node.pri_score)
        
        data = {
            'ATC Code': [],
            'Category': [],
            'Drug Count': [],
            'Percentage': [],
            'Mean PRI': [],
            'Max PRI': []
        }
        
        total = sum(atc_counts.values())
        for code in sorted(atc_counts.keys()):
            data['ATC Code'].append(code)
            data['Category'].append(atc_labels.get(code, 'Unknown'))
            data['Drug Count'].append(atc_counts[code])
            data['Percentage'].append(f"{atc_counts[code]/total*100:.1f}%")
            data['Mean PRI'].append(f"{np.mean(atc_pri[code]):.4f}")
            data['Max PRI'].append(f"{max(atc_pri[code]):.4f}")
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table5_atc_distribution',
                        'Drug Distribution by ATC Classification')
    
    def _table6_phenotype_summary(self):
        """Table 6: Interaction Phenotype Summary"""
        print("\n  → Table 6: Phenotype Summary")
        
        phenotype_data = defaultdict(lambda: {'total': 0, 'contraindicated': 0, 'major': 0})
        
        for edge in self.network.edges:
            for phenotype in edge.phenotypes:
                phenotype_data[phenotype]['total'] += 1
                if edge.severity_label == 'Contraindicated interaction':
                    phenotype_data[phenotype]['contraindicated'] += 1
                elif edge.severity_label == 'Major interaction':
                    phenotype_data[phenotype]['major'] += 1
        
        data = {
            'Phenotype': [],
            'Total Interactions': [],
            'Contraindicated': [],
            'Major': [],
            'High-Risk %': []
        }
        
        for phenotype, counts in sorted(phenotype_data.items(), key=lambda x: -x[1]['total']):
            data['Phenotype'].append(phenotype.replace('_', ' ').title())
            data['Total Interactions'].append(f"{counts['total']:,}")
            data['Contraindicated'].append(f"{counts['contraindicated']:,}")
            data['Major'].append(f"{counts['major']:,}")
            high_risk_pct = (counts['contraindicated'] + counts['major']) / counts['total'] * 100
            data['High-Risk %'].append(f"{high_risk_pct:.1f}%")
        
        df = pd.DataFrame(data)
        self._save_table(df, 'table6_phenotype_summary',
                        'DDI Phenotype Summary')
    
    def _table7_sample_recommendations(self):
        """Table 7: Sample Drug Recommendations"""
        print("\n  → Table 7: Sample Recommendations")
        
        # Test with cardiovascular combo
        test_drugs = ['warfarin', 'aspirin', 'metoprolol', 'clopidogrel']
        valid_drugs = [d for d in test_drugs if d in self.network.nodes]
        
        if len(valid_drugs) >= 2:
            rec = self.recommender.recommend_for_polypharmacy(valid_drugs)
            
            data = {
                'Target Drug': [],
                'Risk Contribution': [],
                'Recommended Alternative': [],
                'Multi-Obj Score': [],
                'PRI Reduction': [],
                'ATC Match': []
            }
            
            for r in rec.get('recommendations', []):
                data['Target Drug'].append(r.get('target_drug', 'N/A'))
                data['Risk Contribution'].append(f"{r.get('risk_contribution', 0):.4f}")
                
                best = r.get('best_alternative', {})
                if best:
                    data['Recommended Alternative'].append(best.get('drug_name', 'N/A'))
                    data['Multi-Obj Score'].append(f"{best.get('multi_objective_score', 0):.4f}")
                    data['PRI Reduction'].append(f"{best.get('risk_metrics', {}).get('pri_reduction', 0):.4f}")
                    data['ATC Match'].append(best.get('atc_match_type', 'N/A'))
                else:
                    data['Recommended Alternative'].append('None found')
                    data['Multi-Obj Score'].append('N/A')
                    data['PRI Reduction'].append('N/A')
                    data['ATC Match'].append('N/A')
            
            df = pd.DataFrame(data)
            self._save_table(df, 'table7_sample_recommendations',
                            f'Sample Drug Recommendations for {", ".join([d.title() for d in valid_drugs])}')
    
    # =========================================================================
    # DATA FILES
    # =========================================================================
    
    def generate_data(self):
        """Generate processed data files"""
        self._data_all_drugs()
        self._data_all_interactions()
        self._data_network_metrics()
        self._data_pri_scores()
    
    def _data_all_drugs(self):
        """Export all drugs with metrics"""
        print("\n  → Exporting all drug data")
        
        data = []
        for name, node in self.network.nodes.items():
            data.append({
                'drug_name': name,
                'drugbank_id': node.drugbank_id,
                'atc_code': node.atc_code,
                'atc_level_3': node.atc_level_3,
                'atc_level_4': node.atc_level_4,
                'is_cardiovascular': node.is_cardiovascular,
                'degree_centrality': node.degree_centrality,
                'weighted_degree': node.weighted_degree,
                'betweenness_centrality': node.betweenness_centrality,
                'pri_score': node.pri_score,
                'contraindicated_count': node.contraindicated_count,
                'major_count': node.major_count,
                'moderate_count': node.moderate_count,
                'minor_count': node.minor_count
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('pri_score', ascending=False)
        
        df.to_csv(self.dirs['data'] / 'all_drugs_with_metrics.csv', index=False)
        df.to_json(self.dirs['data'] / 'all_drugs_with_metrics.json', orient='records', indent=2)
        print(f"     ✓ Saved all_drugs_with_metrics.[csv/json] ({len(df):,} drugs)")
    
    def _data_all_interactions(self):
        """Export all interactions with metadata"""
        print("\n  → Exporting all interaction data")
        
        data = []
        for edge in self.network.edges:
            data.append({
                'drug1': edge.drug1,
                'drug2': edge.drug2,
                'severity_label': edge.severity_label,
                'severity_weight': edge.severity_weight,
                'description': edge.description,
                'confidence': edge.confidence,
                'phenotypes': ','.join(edge.phenotypes),
                'drug1_pri': self.network.nodes[edge.drug1].pri_score,
                'drug2_pri': self.network.nodes[edge.drug2].pri_score,
                'combined_pri': (self.network.nodes[edge.drug1].pri_score + 
                                self.network.nodes[edge.drug2].pri_score) / 2
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('combined_pri', ascending=False)
        
        df.to_csv(self.dirs['data'] / 'all_interactions_with_metrics.csv', index=False)
        print(f"     ✓ Saved all_interactions_with_metrics.csv ({len(df):,} interactions)")
    
    def _data_network_metrics(self):
        """Export network-level metrics"""
        print("\n  → Exporting network metrics")
        
        nodes = list(self.network.nodes.values())
        
        metrics = {
            'network_info': {
                'num_nodes': len(self.network.nodes),
                'num_edges': len(self.network.edges),
                'density': 2*len(self.network.edges)/(len(self.network.nodes)*(len(self.network.nodes)-1))
            },
            'degree_centrality': {
                'mean': np.mean([n.degree_centrality for n in nodes]),
                'std': np.std([n.degree_centrality for n in nodes]),
                'min': min(n.degree_centrality for n in nodes),
                'max': max(n.degree_centrality for n in nodes),
                'median': np.median([n.degree_centrality for n in nodes])
            },
            'weighted_degree': {
                'mean': np.mean([n.weighted_degree for n in nodes]),
                'std': np.std([n.weighted_degree for n in nodes]),
                'min': min(n.weighted_degree for n in nodes),
                'max': max(n.weighted_degree for n in nodes),
                'median': np.median([n.weighted_degree for n in nodes])
            },
            'betweenness_centrality': {
                'mean': np.mean([n.betweenness_centrality for n in nodes]),
                'std': np.std([n.betweenness_centrality for n in nodes]),
                'min': min(n.betweenness_centrality for n in nodes),
                'max': max(n.betweenness_centrality for n in nodes),
                'median': np.median([n.betweenness_centrality for n in nodes])
            },
            'pri_score': {
                'mean': np.mean([n.pri_score for n in nodes]),
                'std': np.std([n.pri_score for n in nodes]),
                'min': min(n.pri_score for n in nodes),
                'max': max(n.pri_score for n in nodes),
                'median': np.median([n.pri_score for n in nodes])
            },
            'pri_weights': self.network.PRI_WEIGHTS,
            'severity_weights': self.network.SEVERITY_WEIGHTS
        }
        
        with open(self.dirs['data'] / 'network_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"     ✓ Saved network_metrics.json")
    
    def _data_pri_scores(self):
        """Export PRI scores for all drugs"""
        print("\n  → Exporting PRI scores")
        
        data = []
        for name, node in sorted(self.network.nodes.items(), key=lambda x: -x[1].pri_score):
            data.append({
                'drug_name': name,
                'pri_score': round(node.pri_score, 6),
                'degree_component': round(node.degree_centrality * self.network.PRI_WEIGHTS['degree'], 6),
                'weighted_component': round(node.weighted_degree * self.network.PRI_WEIGHTS['weighted_degree'], 6),
                'betweenness_component': round(node.betweenness_centrality * self.network.PRI_WEIGHTS['betweenness'], 6),
                'severity_component': round(node.pri_score - (
                    node.degree_centrality * self.network.PRI_WEIGHTS['degree'] +
                    node.weighted_degree * self.network.PRI_WEIGHTS['weighted_degree'] +
                    node.betweenness_centrality * self.network.PRI_WEIGHTS['betweenness']
                ), 6)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.dirs['data'] / 'pri_scores_detailed.csv', index=False)
        print(f"     ✓ Saved pri_scores_detailed.csv")
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def generate_statistics(self):
        """Generate statistical analysis"""
        self._stats_descriptive()
        self._stats_correlation()
        self._stats_hypothesis_tests()
    
    def _stats_descriptive(self):
        """Descriptive statistics"""
        print("\n  → Computing descriptive statistics")
        
        nodes = list(self.network.nodes.values())
        
        stats = {
            'dataset': {
                'total_records': len(self.df),
                'unique_drugs': len(self.network.nodes),
                'unique_interactions': len(self.network.edges),
                'severity_distribution': self.df['severity_label'].value_counts().to_dict()
            },
            'centrality_metrics': {
                'degree': {
                    'mean': float(np.mean([n.degree_centrality for n in nodes])),
                    'std': float(np.std([n.degree_centrality for n in nodes])),
                    'q25': float(np.percentile([n.degree_centrality for n in nodes], 25)),
                    'q50': float(np.percentile([n.degree_centrality for n in nodes], 50)),
                    'q75': float(np.percentile([n.degree_centrality for n in nodes], 75))
                },
                'weighted_degree': {
                    'mean': float(np.mean([n.weighted_degree for n in nodes])),
                    'std': float(np.std([n.weighted_degree for n in nodes])),
                    'q25': float(np.percentile([n.weighted_degree for n in nodes], 25)),
                    'q50': float(np.percentile([n.weighted_degree for n in nodes], 50)),
                    'q75': float(np.percentile([n.weighted_degree for n in nodes], 75))
                },
                'betweenness': {
                    'mean': float(np.mean([n.betweenness_centrality for n in nodes])),
                    'std': float(np.std([n.betweenness_centrality for n in nodes])),
                    'q25': float(np.percentile([n.betweenness_centrality for n in nodes], 25)),
                    'q50': float(np.percentile([n.betweenness_centrality for n in nodes], 50)),
                    'q75': float(np.percentile([n.betweenness_centrality for n in nodes], 75))
                },
                'pri': {
                    'mean': float(np.mean([n.pri_score for n in nodes])),
                    'std': float(np.std([n.pri_score for n in nodes])),
                    'q25': float(np.percentile([n.pri_score for n in nodes], 25)),
                    'q50': float(np.percentile([n.pri_score for n in nodes], 50)),
                    'q75': float(np.percentile([n.pri_score for n in nodes], 75))
                }
            },
            'drug_categories': {
                'cardiovascular': sum(1 for n in nodes if n.is_cardiovascular),
                'non_cardiovascular': sum(1 for n in nodes if not n.is_cardiovascular)
            }
        }
        
        with open(self.dirs['statistics'] / 'descriptive_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"     ✓ Saved descriptive_statistics.json")
    
    def _stats_correlation(self):
        """Correlation analysis"""
        print("\n  → Computing correlations")
        
        nodes = list(self.network.nodes.values())
        
        metrics = {
            'degree_centrality': [n.degree_centrality for n in nodes],
            'weighted_degree': [n.weighted_degree for n in nodes],
            'betweenness_centrality': [n.betweenness_centrality for n in nodes],
            'pri_score': [n.pri_score for n in nodes],
            'contraindicated_count': [n.contraindicated_count for n in nodes],
            'major_count': [n.major_count for n in nodes]
        }
        
        df = pd.DataFrame(metrics)
        
        correlations = {
            'pearson': df.corr(method='pearson').to_dict(),
            'spearman': df.corr(method='spearman').to_dict()
        }
        
        with open(self.dirs['statistics'] / 'correlation_analysis.json', 'w') as f:
            json.dump(correlations, f, indent=2)
        
        # Also save as CSV
        df.corr().to_csv(self.dirs['statistics'] / 'correlation_matrix.csv')
        print(f"     ✓ Saved correlation_analysis.json, correlation_matrix.csv")
    
    def _stats_hypothesis_tests(self):
        """Statistical hypothesis tests"""
        print("\n  → Running hypothesis tests")
        
        try:
            from scipy import stats as scipy_stats
            
            nodes = list(self.network.nodes.values())
            
            # Compare PRI between cardiovascular and non-cardiovascular
            cardio_pri = [n.pri_score for n in nodes if n.is_cardiovascular]
            non_cardio_pri = [n.pri_score for n in nodes if not n.is_cardiovascular]
            
            results = {
                'cardiovascular_vs_other': {
                    'test': 'Mann-Whitney U',
                    'n_cardiovascular': len(cardio_pri),
                    'n_other': len(non_cardio_pri),
                    'mean_cardiovascular': float(np.mean(cardio_pri)) if cardio_pri else None,
                    'mean_other': float(np.mean(non_cardio_pri)) if non_cardio_pri else None,
                    'statistic': float(scipy_stats.mannwhitneyu(cardio_pri, non_cardio_pri)[0]) if cardio_pri and non_cardio_pri else None,
                    'p_value': float(scipy_stats.mannwhitneyu(cardio_pri, non_cardio_pri)[1]) if cardio_pri and non_cardio_pri else None
                },
                'normality_test_pri': {
                    'test': 'Shapiro-Wilk (sample)',
                    'statistic': float(scipy_stats.shapiro([n.pri_score for n in nodes[:1000]])[0]),
                    'p_value': float(scipy_stats.shapiro([n.pri_score for n in nodes[:1000]])[1])
                }
            }
            
            with open(self.dirs['statistics'] / 'hypothesis_tests.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"     ✓ Saved hypothesis_tests.json")
            
        except ImportError:
            print("     ⚠️ scipy not available, skipping hypothesis tests")
    
    def print_summary(self):
        """Print summary of generated files"""
        print(f"\n📁 Output Location: {self.output_dir.absolute()}")
        print("\n📊 Generated Files:")
        
        for dir_name, dir_path in self.dirs.items():
            files = list(dir_path.glob('*'))
            print(f"\n  {dir_name}/ ({len(files)} files)")
            for f in sorted(files)[:5]:
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication materials')
    parser.add_argument('--output', '-o', default='publication',
                       help='Output directory (default: publication)')
    parser.add_argument('--data', '-d', help='Path to DDI data file')
    
    args = parser.parse_args()
    
    generator = PublicationGenerator(
        data_path=args.data,
        output_dir=args.output
    )
    generator.generate_all()


if __name__ == '__main__':
    main()
