#!/usr/bin/env python3
"""
Circos Plot Generator for DDI Network
=====================================
Creates publication-quality circular visualizations of drug-drug interactions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from collections import defaultdict
import warnings
import os

warnings.filterwarnings('ignore')


class CircosPlotGenerator:
    """Generate circos-style plots for DDI visualization."""
    
    def __init__(self, data_path: str, output_dir: str = "publication/figures_circos"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 70)
        print("🔄 CIRCOS PLOT GENERATOR")
        print("=" * 70)
        
        # Load data
        print(f"\n📂 Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"   ✓ Loaded {len(self.df):,} interactions")
        
        # Severity colors
        self.severity_colors = {
            'Contraindicated': '#d62728',  # Red
            'Major': '#ff7f0e',             # Orange
            'Moderate': '#ffbb78',          # Light orange
            'Minor': '#2ca02c'              # Green
        }
        
    def _save_figure(self, fig, name: str):
        """Save figure in multiple formats."""
        for fmt in ['png', 'pdf', 'svg']:
            path = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        print(f"     ✓ Saved {name}.[png/pdf/svg]")
        plt.close(fig)
    
    def _bezier_curve(self, start_angle, end_angle, radius=0.8, n_points=50):
        """Create a bezier curve between two points on a circle."""
        # Start and end points on the circle
        x1 = radius * np.cos(start_angle)
        y1 = radius * np.sin(start_angle)
        x2 = radius * np.cos(end_angle)
        y2 = radius * np.sin(end_angle)
        
        # Control point at center (creates nice arc)
        # Adjust control point based on angular distance
        angular_dist = abs(end_angle - start_angle)
        if angular_dist > np.pi:
            angular_dist = 2 * np.pi - angular_dist
        
        # Closer to center for longer arcs
        control_radius = 0.1 + 0.3 * (1 - angular_dist / np.pi)
        mid_angle = (start_angle + end_angle) / 2
        
        cx = control_radius * np.cos(mid_angle)
        cy = control_radius * np.sin(mid_angle)
        
        # Quadratic bezier
        t = np.linspace(0, 1, n_points)
        x = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        y = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
        
        return x, y
    
    def generate_atc_circos(self):
        """Generate circos plot showing interactions between ATC categories."""
        print("\n  → ATC Category Circos Plot")
        
        import ast
        
        # Helper to extract ATC first letter
        def get_atc_category(atc_str):
            if pd.isna(atc_str):
                return 'Unknown'
            try:
                # Handle string representation of list
                if atc_str.startswith('['):
                    atc_list = ast.literal_eval(atc_str)
                    if atc_list and len(atc_list) > 0:
                        return atc_list[0][0]  # First letter of first code
                elif len(atc_str) > 0:
                    return atc_str[0]
            except:
                pass
            return 'Unknown'
        
        # Extract ATC first letter (main category)
        df = self.df.copy()
        df['atc_cat_1'] = df['atc_1'].apply(get_atc_category)
        df['atc_cat_2'] = df['atc_2'].apply(get_atc_category)
        
        # ATC category names
        atc_names = {
            'A': 'Alimentary',
            'B': 'Blood',
            'C': 'Cardiovascular',
            'D': 'Dermatologicals',
            'G': 'Genito-urinary',
            'H': 'Hormones',
            'J': 'Anti-infectives',
            'L': 'Antineoplastic',
            'M': 'Musculo-skeletal',
            'N': 'Nervous System',
            'P': 'Antiparasitic',
            'R': 'Respiratory',
            'S': 'Sensory Organs',
            'V': 'Various',
            'Unknown': 'Unknown'
        }
        
        # Count interactions between categories
        interaction_matrix = defaultdict(lambda: defaultdict(int))
        severity_matrix = defaultdict(lambda: defaultdict(list))
        
        for _, row in df.iterrows():
            cat1, cat2 = row['atc_cat_1'], row['atc_cat_2']
            interaction_matrix[cat1][cat2] += 1
            if cat1 != cat2:
                interaction_matrix[cat2][cat1] += 1
            severity_matrix[cat1][cat2].append(row['severity_label'])
        
        # Get categories with significant interactions
        categories = sorted([c for c in interaction_matrix.keys() if c != 'Unknown'])
        n_cats = len(categories)
        
        if n_cats < 2:
            print("     ⚠ Not enough ATC categories for circos plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': 'polar'})
        
        # Calculate segment sizes based on total interactions
        cat_totals = {}
        for cat in categories:
            total = sum(interaction_matrix[cat].values())
            cat_totals[cat] = total
        
        total_all = sum(cat_totals.values())
        
        # Gap between segments
        gap = 0.02  # radians
        total_gap = gap * n_cats
        available_space = 2 * np.pi - total_gap
        
        # Calculate angles for each category
        cat_angles = {}
        current_angle = 0
        
        for cat in categories:
            proportion = cat_totals[cat] / total_all if total_all > 0 else 1/n_cats
            segment_size = proportion * available_space
            cat_angles[cat] = {
                'start': current_angle,
                'end': current_angle + segment_size,
                'mid': current_angle + segment_size / 2
            }
            current_angle += segment_size + gap
        
        # Color map for categories
        cmap = plt.cm.tab20
        cat_colors = {cat: cmap(i / n_cats) for i, cat in enumerate(categories)}
        
        # Draw outer arcs (category segments)
        for cat in categories:
            angles = cat_angles[cat]
            theta = np.linspace(angles['start'], angles['end'], 100)
            
            # Outer arc
            ax.fill_between(theta, 0.92, 1.0, color=cat_colors[cat], alpha=0.8)
            
            # Category label
            mid_angle = angles['mid']
            rotation = np.degrees(mid_angle) - 90
            if mid_angle > np.pi/2 and mid_angle < 3*np.pi/2:
                rotation += 180
            
            label = atc_names.get(cat, cat)
            if len(label) > 12:
                label = label[:10] + '...'
            
            ax.text(mid_angle, 1.08, label, ha='center', va='center',
                   fontsize=9, fontweight='bold', rotation=rotation,
                   rotation_mode='anchor')
        
        # Draw connections (chords)
        # Get top interactions
        interactions = []
        for cat1 in categories:
            for cat2 in categories:
                if cat1 < cat2:  # Avoid duplicates
                    count = interaction_matrix[cat1][cat2]
                    if count > 0:
                        # Get dominant severity
                        severities = severity_matrix[cat1][cat2]
                        if severities:
                            from collections import Counter
                            dominant = Counter(severities).most_common(1)[0][0]
                        else:
                            dominant = 'Major'
                        interactions.append((cat1, cat2, count, dominant))
        
        # Sort by count and take top interactions
        interactions.sort(key=lambda x: -x[2])
        top_interactions = interactions[:50]  # Top 50 connections
        
        max_count = max(i[2] for i in top_interactions) if top_interactions else 1
        
        for cat1, cat2, count, severity in top_interactions:
            angle1 = cat_angles[cat1]['mid']
            angle2 = cat_angles[cat2]['mid']
            
            # Line width based on interaction count
            lw = 0.5 + 4 * (count / max_count)
            
            # Color based on severity
            color = self.severity_colors.get(severity, '#888888')
            alpha = 0.3 + 0.4 * (count / max_count)
            
            # Draw bezier curve
            x, y = self._bezier_curve(angle1, angle2, radius=0.85)
            
            # Convert to polar coordinates for plotting
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            ax.plot(theta, r, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
        
        # Remove polar grid
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        # Add legend for severity
        legend_elements = [
            mpatches.Patch(facecolor=self.severity_colors['Contraindicated'], 
                          label='Contraindicated', alpha=0.7),
            mpatches.Patch(facecolor=self.severity_colors['Major'], 
                          label='Major', alpha=0.7),
            mpatches.Patch(facecolor=self.severity_colors['Moderate'], 
                          label='Moderate', alpha=0.7),
            mpatches.Patch(facecolor=self.severity_colors['Minor'], 
                          label='Minor', alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='lower right', 
                 bbox_to_anchor=(1.15, -0.05), title='Severity')
        
        plt.title('Drug-Drug Interactions by ATC Category\n(Chord width ∝ interaction count)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        self._save_figure(fig, 'circos_atc_categories')
    
    def generate_top_drugs_circos(self, n_drugs: int = 30):
        """Generate circos plot for top interacting drugs."""
        print(f"\n  → Top {n_drugs} Drugs Circos Plot")
        
        # Count interactions per drug
        drug_counts = defaultdict(int)
        for _, row in self.df.iterrows():
            drug_counts[row['drug_name_1']] += 1
            drug_counts[row['drug_name_2']] += 1
        
        # Get top drugs
        top_drugs = sorted(drug_counts.items(), key=lambda x: -x[1])[:n_drugs]
        drug_names = [d[0] for d in top_drugs]
        
        # Filter interactions to only those between top drugs
        mask = (self.df['drug_name_1'].isin(drug_names)) & (self.df['drug_name_2'].isin(drug_names))
        filtered_df = self.df[mask]
        
        if len(filtered_df) == 0:
            print("     ⚠ No interactions between top drugs")
            return
        
        # Build interaction matrix
        interaction_counts = defaultdict(lambda: defaultdict(int))
        severity_dominant = defaultdict(lambda: defaultdict(str))
        
        for _, row in filtered_df.iterrows():
            d1, d2 = row['drug_name_1'], row['drug_name_2']
            interaction_counts[d1][d2] += 1
            severity_dominant[d1][d2] = row['severity_label']
        
        n_drugs_actual = len(drug_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={'projection': 'polar'})
        
        # Calculate segment sizes
        gap = 0.03
        total_gap = gap * n_drugs_actual
        segment_size = (2 * np.pi - total_gap) / n_drugs_actual
        
        # Drug angles
        drug_angles = {}
        for i, drug in enumerate(drug_names):
            start = i * (segment_size + gap)
            drug_angles[drug] = {
                'start': start,
                'end': start + segment_size,
                'mid': start + segment_size / 2
            }
        
        # Color drugs by category (cardiovascular includes antithrombotic)
        drug_categories = {}
        for drug in drug_names:
            drug_row = self.df[self.df['drug_name_1'] == drug].iloc[0] if len(self.df[self.df['drug_name_1'] == drug]) > 0 else None
            if drug_row is not None:
                if drug_row['is_cardiovascular_1'] or drug_row['is_antithrombotic_1']:
                    drug_categories[drug] = 'Cardiovascular'
                else:
                    drug_categories[drug] = 'Other'
            else:
                drug_categories[drug] = 'Other'
        
        category_colors = {
            'Cardiovascular': '#e41a1c',
            'Other': '#4daf4a'
        }
        
        # Draw outer arcs
        for drug in drug_names:
            angles = drug_angles[drug]
            theta = np.linspace(angles['start'], angles['end'], 50)
            
            cat = drug_categories.get(drug, 'Other')
            color = category_colors[cat]
            
            ax.fill_between(theta, 0.9, 1.0, color=color, alpha=0.8)
            
            # Drug label
            mid_angle = angles['mid']
            rotation = np.degrees(mid_angle) - 90
            if mid_angle > np.pi/2 and mid_angle < 3*np.pi/2:
                rotation += 180
            
            # Truncate long names
            label = drug[:15] + '...' if len(drug) > 15 else drug
            
            ax.text(mid_angle, 1.08, label, ha='center', va='center',
                   fontsize=7, rotation=rotation, rotation_mode='anchor')
        
        # Draw connections
        connections = []
        for d1 in drug_names:
            for d2 in drug_names:
                if d1 < d2:
                    count = interaction_counts[d1][d2] + interaction_counts[d2][d1]
                    if count > 0:
                        sev = severity_dominant[d1][d2] or severity_dominant[d2][d1] or 'Major'
                        connections.append((d1, d2, count, sev))
        
        if not connections:
            print("     ⚠ No connections to draw")
            plt.close(fig)
            return
        
        max_count = max(c[2] for c in connections)
        
        for d1, d2, count, severity in connections:
            angle1 = drug_angles[d1]['mid']
            angle2 = drug_angles[d2]['mid']
            
            lw = 0.3 + 3 * (count / max_count)
            color = self.severity_colors.get(severity, '#888888')
            alpha = 0.2 + 0.5 * (count / max_count)
            
            x, y = self._bezier_curve(angle1, angle2, radius=0.85)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            ax.plot(theta, r, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
        
        # Clean up
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        # Legends
        severity_legend = [
            mpatches.Patch(facecolor=c, label=s, alpha=0.7)
            for s, c in self.severity_colors.items()
        ]
        
        category_legend = [
            mpatches.Patch(facecolor=c, label=cat, alpha=0.8)
            for cat, c in category_colors.items()
        ]
        
        leg1 = ax.legend(handles=severity_legend, loc='lower right',
                        bbox_to_anchor=(1.2, 0), title='Severity')
        ax.add_artist(leg1)
        ax.legend(handles=category_legend, loc='lower right',
                 bbox_to_anchor=(1.2, 0.25), title='Drug Category')
        
        plt.title(f'Top {n_drugs_actual} Most Interacting Drugs\n(Connections colored by severity)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        self._save_figure(fig, f'circos_top_{n_drugs}_drugs')
    
    def generate_cardiovascular_circos(self):
        """Generate circos specifically for cardiovascular drugs (includes former antithrombotic)."""
        print("\n  → Cardiovascular Drugs Circos Plot")
        
        # Collect all cardiovascular drugs (combining CV and AT)
        cv_drugs = set()
        
        for _, row in self.df.iterrows():
            if row['is_cardiovascular_1'] or row['is_antithrombotic_1']:
                cv_drugs.add(row['drug_name_1'])
            if row['is_cardiovascular_2'] or row['is_antithrombotic_2']:
                cv_drugs.add(row['drug_name_2'])
        
        # Get interactions among CV drugs
        cv_interactions = self.df[
            ((self.df['drug_name_1'].isin(cv_drugs)) & (self.df['drug_name_2'].isin(cv_drugs)))
        ]
        
        if len(cv_interactions) == 0:
            print("     ⚠ No cardiovascular interactions found")
            return
        
        # Count per drug
        drug_interaction_count = defaultdict(int)
        for _, row in cv_interactions.iterrows():
            drug_interaction_count[row['drug_name_1']] += 1
            drug_interaction_count[row['drug_name_2']] += 1
        
        # Top CV drugs
        cv_sorted = sorted([(d, drug_interaction_count[d]) for d in cv_drugs], key=lambda x: -x[1])[:35]
        top_cv = [d[0] for d in cv_sorted]
        
        if len(top_cv) < 2:
            print("     ⚠ Not enough cardiovascular drugs")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={'projection': 'polar'})
        
        # Distribute drugs around the circle
        gap = 0.02
        total_space = 2 * np.pi - len(top_cv) * gap
        segment = total_space / len(top_cv)
        
        drug_angles = {}
        for i, drug in enumerate(top_cv):
            start = i * (segment + gap)
            drug_angles[drug] = {
                'start': start,
                'end': start + segment * 0.9,
                'mid': start + segment * 0.45
            }
        
        # Draw segments (all cardiovascular = red)
        for drug, angles in drug_angles.items():
            theta = np.linspace(angles['start'], angles['end'], 50)
            ax.fill_between(theta, 0.88, 1.0, color='#e41a1c', alpha=0.7)
            
            # Label
            mid = angles['mid']
            rotation = np.degrees(mid) - 90
            if mid > np.pi/2 and mid < 3*np.pi/2:
                rotation += 180
            
            label = drug[:12] + '..' if len(drug) > 12 else drug
            ax.text(mid, 1.08, label, ha='center', va='center',
                   fontsize=7, rotation=rotation, rotation_mode='anchor')
        
        # Draw connections
        connections = []
        for _, row in cv_interactions.iterrows():
            d1, d2 = row['drug_name_1'], row['drug_name_2']
            if d1 in drug_angles and d2 in drug_angles and d1 != d2:
                connections.append((d1, d2, row['severity_label']))
        
        # Aggregate
        conn_counts = defaultdict(lambda: {'count': 0, 'severity': 'Major'})
        for d1, d2, sev in connections:
            key = tuple(sorted([d1, d2]))
            conn_counts[key]['count'] += 1
            conn_counts[key]['severity'] = sev
        
        if not conn_counts:
            print("     ⚠ No connections to draw")
            plt.close(fig)
            return
        
        max_count = max(v['count'] for v in conn_counts.values())
        
        for (d1, d2), data in conn_counts.items():
            angle1 = drug_angles[d1]['mid']
            angle2 = drug_angles[d2]['mid']
            
            lw = 0.3 + 2.5 * (data['count'] / max_count)
            color = self.severity_colors.get(data['severity'], '#888888')
            alpha = 0.15 + 0.5 * (data['count'] / max_count)
            
            x, y = self._bezier_curve(angle1, angle2, radius=0.82)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            ax.plot(theta, r, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
        
        # Clean up
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        # Add center label
        ax.text(0, 0.3, 'CARDIOVASCULAR\nDRUG\nINTERACTIONS', ha='center', va='center',
               fontsize=12, fontweight='bold', color='#e41a1c', alpha=0.5)
        
        # Legend
        severity_legend = [
            mpatches.Patch(facecolor=c, label=s, alpha=0.7)
            for s, c in self.severity_colors.items()
        ]
        ax.legend(handles=severity_legend, loc='lower right',
                 bbox_to_anchor=(1.15, 0), title='Severity')
        
        plt.title('Cardiovascular Drug Interactions\n(Top drugs by interaction count)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        self._save_figure(fig, 'circos_cardiovascular_drugs')
    
    def generate_severity_circos(self):
        """Generate circos grouped by severity level."""
        print("\n  → Severity-Grouped Circos Plot")
        
        # Group drugs by their most common severity
        drug_severity = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            drug_severity[row['drug_name_1']][row['severity_label']] += 1
            drug_severity[row['drug_name_2']][row['severity_label']] += 1
        
        # Assign each drug to dominant severity
        drug_dominant_severity = {}
        for drug, severities in drug_severity.items():
            dominant = max(severities.items(), key=lambda x: x[1])[0]
            # Normalize severity names
            if 'contraindicated' in dominant.lower():
                dominant = 'Contraindicated'
            elif 'major' in dominant.lower():
                dominant = 'Major'
            elif 'moderate' in dominant.lower():
                dominant = 'Moderate'
            elif 'minor' in dominant.lower():
                dominant = 'Minor'
            drug_dominant_severity[drug] = dominant
        
        # Get top drugs per severity
        severity_order = ['Contraindicated', 'Major', 'Moderate', 'Minor']
        drugs_per_severity = {s: [] for s in severity_order}
        
        for drug, sev in drug_dominant_severity.items():
            if sev in drugs_per_severity:
                total_interactions = sum(drug_severity[drug].values())
                drugs_per_severity[sev].append((drug, total_interactions))
        
        # Sort and take top 10 per category
        for sev in severity_order:
            drugs_per_severity[sev] = sorted(drugs_per_severity[sev], key=lambda x: -x[1])[:10]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={'projection': 'polar'})
        
        # Assign angles
        drug_angles = {}
        current_angle = 0
        gap_between_groups = 0.15
        
        for sev in severity_order:
            drugs = drugs_per_severity[sev]
            if not drugs:
                continue
            
            segment_size = (2 * np.pi - len(severity_order) * gap_between_groups) / 4
            drug_gap = segment_size / len(drugs) if drugs else 0
            
            for i, (drug, _) in enumerate(drugs):
                start = current_angle + i * drug_gap
                drug_angles[drug] = {
                    'start': start,
                    'end': start + drug_gap * 0.85,
                    'mid': start + drug_gap * 0.425,
                    'severity': sev
                }
            
            current_angle += segment_size + gap_between_groups
        
        # Draw segments
        for drug, angles in drug_angles.items():
            theta = np.linspace(angles['start'], angles['end'], 30)
            color = self.severity_colors[angles['severity']]
            
            ax.fill_between(theta, 0.88, 1.0, color=color, alpha=0.8)
            
            mid = angles['mid']
            rotation = np.degrees(mid) - 90
            if mid > np.pi/2 and mid < 3*np.pi/2:
                rotation += 180
            
            label = drug[:10] + '..' if len(drug) > 10 else drug
            ax.text(mid, 1.08, label, ha='center', va='center',
                   fontsize=6, rotation=rotation, rotation_mode='anchor')
        
        # Draw connections
        all_drugs_set = set(drug_angles.keys())
        mask = (self.df['drug_name_1'].isin(all_drugs_set)) & (self.df['drug_name_2'].isin(all_drugs_set))
        filtered = self.df[mask]
        
        conn_counts = defaultdict(lambda: {'count': 0, 'severity': 'Major'})
        for _, row in filtered.iterrows():
            key = tuple(sorted([row['drug_name_1'], row['drug_name_2']]))
            conn_counts[key]['count'] += 1
            conn_counts[key]['severity'] = row['severity_label']
        
        if conn_counts:
            max_count = max(v['count'] for v in conn_counts.values())
            
            for (d1, d2), data in conn_counts.items():
                if d1 not in drug_angles or d2 not in drug_angles:
                    continue
                    
                angle1 = drug_angles[d1]['mid']
                angle2 = drug_angles[d2]['mid']
                
                lw = 0.2 + 2 * (data['count'] / max_count)
                color = self.severity_colors.get(data['severity'], '#888888')
                alpha = 0.1 + 0.4 * (data['count'] / max_count)
                
                x, y = self._bezier_curve(angle1, angle2, radius=0.82)
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                
                ax.plot(theta, r, color=color, alpha=alpha, linewidth=lw)
        
        # Clean up
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.severity_colors[s], label=s, alpha=0.8)
            for s in severity_order
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                 bbox_to_anchor=(1.15, 0), title='Severity Group')
        
        plt.title('Drugs Grouped by Dominant Interaction Severity\n(Top 10 drugs per severity category)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        self._save_figure(fig, 'circos_severity_grouped')
    
    def generate_all(self):
        """Generate all circos plots."""
        print("\n" + "=" * 70)
        print("📈 GENERATING CIRCOS PLOTS")
        print("=" * 70)
        
        self.generate_atc_circos()
        self.generate_top_drugs_circos(30)
        self.generate_cardiovascular_circos()
        self.generate_severity_circos()
        
        print("\n" + "=" * 70)
        print(f"✅ CIRCOS PLOTS COMPLETE")
        print(f"📁 Output: {os.path.abspath(self.output_dir)}")
        print("=" * 70)


def main():
    # Try data/ folder first, then root
    data_path = "data/ddi_cardio_or_antithrombotic_labeled (1).csv"
    if not os.path.exists(data_path):
        data_path = "ddi_cardio_or_antithrombotic_labeled (1).csv"
    generator = CircosPlotGenerator(data_path)
    generator.generate_all()


if __name__ == "__main__":
    main()
