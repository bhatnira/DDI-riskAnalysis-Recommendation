#!/usr/bin/env python3
"""
Generate Updated Workflow Figure for DDI Severity Classification
Reflects actual methodology: Rule-Based classifier with DDInter validation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path('/home/nbhatta1/Desktop/copyOfOriginal/publication_recalibration/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def create_workflow_figure():
    """
    Create workflow diagram reflecting actual methodology:
    1. DrugBank DDI descriptions
    2. DDInter matching for severity labels
    3. 70/30 train/test split
    4. Keyword weight derivation (log-likelihood)
    5. Score calculation + Percentile thresholds
    6. Severity classification
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Style definitions
    box_data = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    box_process = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    box_output = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    box_split = dict(boxstyle='round,pad=0.3', facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
    arrow_props = dict(arrowstyle='->', color='#37474F', linewidth=2, connectionstyle='arc3,rad=0')
    
    # =========================================================================
    # ROW 1: DATA SOURCES
    # =========================================================================
    
    # DrugBank
    ax.annotate('DrugBank v5.1.9\n759,774 DDI pairs\n(descriptions)', 
                xy=(2, 7), fontsize=9, ha='center', va='center', bbox=box_data)
    
    # DDInter
    ax.annotate('DDInter Database\n(severity labels,\nno descriptions)', 
                xy=(5, 7), fontsize=9, ha='center', va='center', bbox=box_data)
    
    # Matching arrow
    ax.annotate('', xy=(3.3, 7), xytext=(2.8, 7), arrowprops=arrow_props)
    ax.annotate('', xy=(4.2, 7), xytext=(3.7, 7), arrowprops=arrow_props)
    
    # Matched dataset
    ax.annotate('Matched Dataset\n37,164 DDI pairs\n(descriptions + labels)', 
                xy=(8, 7), fontsize=9, ha='center', va='center', bbox=box_process)
    
    ax.annotate('', xy=(6.5, 7), xytext=(5.8, 7), arrowprops=arrow_props)
    
    # =========================================================================
    # ROW 2: TRAIN/TEST SPLIT
    # =========================================================================
    
    # Split box
    ax.annotate('70/30 Train/Test Split\n(stratified)', 
                xy=(8, 5.5), fontsize=9, ha='center', va='center', bbox=box_split)
    
    ax.annotate('', xy=(8, 6.3), xytext=(8, 6.7), arrowprops=arrow_props)
    
    # Training set
    ax.annotate('Training Set\nn=26,014', 
                xy=(5.5, 4.2), fontsize=9, ha='center', va='center', bbox=box_split)
    
    # Test set
    ax.annotate('Test Set\nn=11,150', 
                xy=(10.5, 4.2), fontsize=9, ha='center', va='center', bbox=box_split)
    
    # Split arrows
    ax.annotate('', xy=(6.5, 4.8), xytext=(7.3, 5.3), arrowprops=arrow_props)
    ax.annotate('', xy=(9.5, 4.8), xytext=(8.7, 5.3), arrowprops=arrow_props)
    
    # =========================================================================
    # ROW 3: WEIGHT DERIVATION (from training)
    # =========================================================================
    
    ax.annotate('Keyword Weight\nDerivation\n$w_k = \\ln\\frac{P(k|Major)}{P(k|Mod)}$', 
                xy=(5.5, 2.5), fontsize=9, ha='center', va='center', bbox=box_process)
    
    ax.annotate('', xy=(5.5, 3.5), xytext=(5.5, 3.9), arrowprops=arrow_props)
    
    # Example weights
    ax.annotate('Examples:\nbleeding: +1.03\ntherapeutic: −1.20', 
                xy=(2.5, 2.5), fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.annotate('', xy=(4.0, 2.5), xytext=(3.3, 2.5), arrowprops=dict(arrowstyle='-', color='gray', linewidth=1))
    
    # =========================================================================
    # ROW 4: CLASSIFICATION
    # =========================================================================
    
    # Score calculation
    ax.annotate('Score Calculation\n$S = \\sum_k w_k \\cdot \\mathbb{1}[k \\in d]$', 
                xy=(8, 2.5), fontsize=9, ha='center', va='center', bbox=box_process)
    
    ax.annotate('', xy=(6.8, 2.5), xytext=(6.2, 2.5), arrowprops=arrow_props)
    
    # Percentile thresholds
    ax.annotate('Percentile Thresholds\n(from training scores)\nP₉₆, P₇₈, P₄', 
                xy=(11, 2.5), fontsize=9, ha='center', va='center', bbox=box_process)
    
    ax.annotate('', xy=(9.5, 2.5), xytext=(8.9, 2.5), arrowprops=arrow_props)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    # Apply to test set
    ax.annotate('', xy=(10.5, 3.5), xytext=(10.5, 3.9), arrowprops=arrow_props)
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    ax.annotate('Severity Classification\nContra (4%) | Major (18%)\nModerate (74%) | Minor (4%)', 
                xy=(11, 0.8), fontsize=9, ha='center', va='center', bbox=box_output)
    
    ax.annotate('', xy=(11, 1.6), xytext=(11, 2.1), arrowprops=arrow_props)
    
    # =========================================================================
    # VALIDATION RESULTS BOX
    # =========================================================================
    
    ax.annotate('DDInter Validation (test set)\nExact Acc: 66.4%\nAdjacent Acc: 99.3%\nCohen\'s κ: +0.096', 
                xy=(13, 4.2), fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#FFFDE7', edgecolor='#FBC02D', linewidth=2))
    
    ax.annotate('', xy=(11.8, 4.2), xytext=(11.2, 4.2), arrowprops=dict(arrowstyle='->', color='#FBC02D', linewidth=2))
    
    # =========================================================================
    # TITLE
    # =========================================================================
    
    ax.set_title('Rule-Based DDI Severity Classification Workflow', 
                 fontsize=14, fontweight='bold', loc='left')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Data Source'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='Processing'),
        mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label='Train/Test Split'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, ncol=4)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6_workflow.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig6_workflow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Updated workflow figure saved:")
    print(f"  - {OUTPUT_DIR / 'fig6_workflow.pdf'}")
    print(f"  - {OUTPUT_DIR / 'fig6_workflow.png'}")

if __name__ == '__main__':
    create_workflow_figure()
