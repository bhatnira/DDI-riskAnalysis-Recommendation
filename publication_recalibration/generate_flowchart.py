#!/usr/bin/env python3
"""
Generate Recalibration Flowchart for Publication

Accurate representation of methodology from validate_against_ddinter.py:
1. Data Sources → Matching → Train/Test Split
2. Training Set: Derive keyword weights and percentile thresholds
3. Test Set: ALL 4 methods evaluated (Zero-Shot, Confidence-Weighted, Rule-Based, Evidence-Based)
4. Selection: Rule-Based chosen for best accuracy + parsimony

Author: DDI Research Team
Date: February 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def create_flowchart():
    """
    Create flowchart showing the complete methodology:
    - Training: weight derivation and threshold calibration
    - Testing: ALL 4 methods evaluated on held-out test set
    - Selection: Rule-Based chosen based on test performance + parsimony
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # ==========================================================================
    # Style Definitions
    # ==========================================================================
    box_data = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    box_process = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    box_output = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    box_split = dict(boxstyle='round,pad=0.3', facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
    box_method = dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5)
    box_selected = dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=3)
    box_result = dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FBC02D', linewidth=2)
    
    arrow_main = dict(arrowstyle='->', color='#37474F', linewidth=2)
    arrow_train = dict(arrowstyle='->', color='#1565C0', linewidth=2)  # Blue for training flow
    arrow_test = dict(arrowstyle='->', color='#C2185B', linewidth=2)   # Pink for test flow
    
    # ==========================================================================
    # Row 1: Data Sources (y=10)
    # ==========================================================================
    ax.annotate('DrugBank v5.1.9\n759,774 DDI pairs\n(descriptions)', 
                xy=(2.5, 10), fontsize=9, ha='center', va='center', bbox=box_data)
    
    ax.annotate('DDInter Database\n(severity labels)', 
                xy=(5.5, 10), fontsize=9, ha='center', va='center', bbox=box_data)
    
    # Arrow: DrugBank → DDInter (matching)
    ax.annotate('', xy=(4.2, 10), xytext=(3.5, 10), arrowprops=arrow_main)
    
    # Matching process
    ax.annotate('Drug Name\nMatching', 
                xy=(8, 10), fontsize=9, ha='center', va='center', bbox=box_process)
    ax.annotate('', xy=(6.8, 10), xytext=(6.3, 10), arrowprops=arrow_main)
    
    # Matched dataset
    ax.annotate('Matched Dataset\nn=37,164', 
                xy=(11, 10), fontsize=9, ha='center', va='center', bbox=box_process)
    ax.annotate('', xy=(9.5, 10), xytext=(8.9, 10), arrowprops=arrow_main)
    
    # ==========================================================================
    # Row 2: Train/Test Split (y=8.5)
    # ==========================================================================
    ax.annotate('70/30 Stratified Split', 
                xy=(11, 8.5), fontsize=9, ha='center', va='center', bbox=box_split)
    ax.annotate('', xy=(11, 9.2), xytext=(11, 9.6), arrowprops=arrow_main)
    
    # ==========================================================================
    # Row 3: Training and Test Sets (y=7)
    # ==========================================================================
    # Training set
    ax.annotate('Training Set\nn=26,014 (70%)', 
                xy=(5, 7), fontsize=9, ha='center', va='center', bbox=box_split,
                fontweight='bold')
    ax.annotate('', xy=(8, 7.8), xytext=(10, 8.3), arrowprops=arrow_train)
    
    # Test set
    ax.annotate('Test Set\nn=11,150 (30%)', 
                xy=(11, 7), fontsize=9, ha='center', va='center', bbox=box_split,
                fontweight='bold')
    ax.annotate('', xy=(11, 7.8), xytext=(11, 8.2), arrowprops=arrow_test)
    
    # Arrow from split to training
    ax.annotate('', xy=(7, 7.8), xytext=(10, 8.3), arrowprops=arrow_train)
    
    # ==========================================================================
    # Row 4: Training Process (y=5.5) - LEFT SIDE
    # ==========================================================================
    ax.annotate('Keyword Weight\nDerivation\n(log-likelihood ratios)', 
                xy=(3, 5.5), fontsize=9, ha='center', va='center', bbox=box_process)
    ax.annotate('', xy=(4, 6.5), xytext=(4.5, 6.7), arrowprops=arrow_train)
    
    ax.annotate('Percentile Threshold\nCalibration\n(P96, P78, P4)', 
                xy=(6.5, 5.5), fontsize=9, ha='center', va='center', bbox=box_process)
    ax.annotate('', xy=(5.5, 6.5), xytext=(5.2, 6.7), arrowprops=arrow_train)
    
    # Arrow connecting training processes
    ax.annotate('', xy=(5.2, 5.5), xytext=(4.3, 5.5), arrowprops=arrow_train)
    
    # ==========================================================================
    # Row 5: 4-Method Comparison Grid (y=3.5) - CENTER
    # ==========================================================================
    # Title for method comparison
    ax.annotate('ALL 4 METHODS EVALUATED ON TEST SET', 
                xy=(7, 4.5), fontsize=10, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=1))
    
    # Arrow from test set to methods
    ax.annotate('', xy=(8, 6.3), xytext=(10.5, 6.7), arrowprops=arrow_test)
    
    # Method boxes with results
    methods = [
        ('Zero-Shot\nBART-MNLI', '22.1%', 2.5),
        ('Confidence-\nWeighted', '22.1%', 5.5),
        ('Rule-Based', '66.4%', 8.5),
        ('Evidence-\nBased', '66.4%', 11.5),
    ]
    
    for name, acc, x in methods:
        if name == 'Rule-Based':
            # Selected method - highlighted
            ax.annotate(f'{name}\nAcc: {acc}', 
                       xy=(x, 3.2), fontsize=9, ha='center', va='center', 
                       bbox=box_selected, fontweight='bold')
        else:
            ax.annotate(f'{name}\nAcc: {acc}', 
                       xy=(x, 3.2), fontsize=9, ha='center', va='center', 
                       bbox=box_method)
    
    # Arrows from trained parameters to methods (Rule-Based and Evidence-Based use them)
    ax.annotate('', xy=(7.5, 4), xytext=(6.8, 5.1), arrowprops=arrow_train)
    ax.annotate('', xy=(10.5, 4), xytext=(6.8, 5.1), arrowprops=arrow_train)
    
    # ==========================================================================
    # Row 6: Selection Process (y=1.8)
    # ==========================================================================
    ax.annotate('Method Selection\n(best accuracy + parsimony)', 
                xy=(7, 1.8), fontsize=9, ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='#E1F5FE', edgecolor='#0288D1', linewidth=2))
    
    # Arrows from methods to selection
    for x in [2.5, 5.5, 8.5, 11.5]:
        ax.annotate('', xy=(7, 2.3), xytext=(x, 2.7), 
                   arrowprops=dict(arrowstyle='->', color='#9E9E9E', linewidth=1, alpha=0.5))
    
    # Highlight arrow from Rule-Based
    ax.annotate('', xy=(7.5, 2.3), xytext=(8.5, 2.7), 
               arrowprops=dict(arrowstyle='->', color='#2E7D32', linewidth=2.5))
    
    # ==========================================================================
    # Row 7: Final Output (y=0.5)
    # ==========================================================================
    ax.annotate('SELECTED: Rule-Based Classifier\nExact Acc: 66.4% | Adjacent: 99.3%\nKappa: +0.096', 
                xy=(7, 0.5), fontsize=10, ha='center', va='center', 
                bbox=box_output, fontweight='bold')
    ax.annotate('', xy=(7, 1.1), xytext=(7, 1.5), arrowprops=arrow_main)
    
    # ==========================================================================
    # Key insight annotation
    # ==========================================================================
    ax.annotate('Severity Priority:\nAssign highest\nlevel matched', 
                xy=(1, 3.2), fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    # ==========================================================================
    # Legend
    # ==========================================================================
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Data Source'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='Processing'),
        mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label='Train/Test'),
        mpatches.Patch(facecolor='#F3E5F5', edgecolor='#7B1FA2', label='Method'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Selected'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    # Color legend for arrows
    ax.annotate('Blue arrows: Training flow', xy=(0.5, 0.2), fontsize=7, color='#1565C0')
    ax.annotate('Pink arrows: Test flow', xy=(0.5, -0.1), fontsize=7, color='#C2185B')
    
    # ==========================================================================
    # Title
    # ==========================================================================
    ax.set_title('DDI Severity Recalibration Framework', 
                 fontsize=14, fontweight='bold', loc='left', pad=10)
    
    plt.tight_layout()
    
    # Save figures
    fig.savefig(OUTPUT_DIR / 'fig_recalibration_flowchart.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_recalibration_flowchart.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("✓ Flowchart saved:")
    print(f"  - {OUTPUT_DIR / 'fig_recalibration_flowchart.pdf'}")
    print(f"  - {OUTPUT_DIR / 'fig_recalibration_flowchart.png'}")


if __name__ == '__main__':
    create_flowchart()
