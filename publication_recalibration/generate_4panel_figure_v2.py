#!/usr/bin/env python3
"""
Generate 4-Panel Validation Figure for Publication (Version 2)

Four DISTINCT analyses - no redundancy:
(A) Method Comparison - Exact accuracy bar chart
(B) Confusion Matrix - Where Rule-Based gets it right/wrong
(C) Per-Class Recall - Clinical performance by severity
(D) Score Discrimination - Mechanistic basis for classification

DATA SOURCE: All values from validate_against_ddinter.py output
- Test set: n=11,150 (30% of 37,164 matched pairs)

Author: DDI Research Team
Date: February 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# =============================================================================
# VERIFIED DATA FROM validate_against_ddinter.py
# =============================================================================

# Method comparison (test set n=11,150)
METHODS = {
    'Zero-Shot\nBART-MNLI': {'exact': 22.1, 'adjacent': 95.6, 'kappa': 0.000, 'type': 'baseline'},
    'Confidence\nWeighted': {'exact': 22.1, 'adjacent': 95.6, 'kappa': 0.000, 'type': 'baseline'},
    'Rule-Based': {'exact': 66.4, 'adjacent': 99.3, 'kappa': 0.096, 'type': 'calibrated'},
    'Evidence-Based': {'exact': 66.4, 'adjacent': 99.4, 'kappa': 0.096, 'type': 'calibrated'},
}

# Score statistics by DDInter ground truth severity
SCORE_STATS = {
    'Major': {'mean': 0.926, 'std': 1.328, 'n': 2464},
    'Moderate': {'mean': 0.065, 'std': 1.477, 'n': 8193},
    'Minor': {'mean': -0.281, 'std': 1.250, 'n': 493},
}

# Per-class recall for Rule-Based (from test set)
PER_CLASS_RECALL = {
    'Major': 20.4,
    'Moderate': 83.5,
    'Minor': 11.4,
}

# Per-class precision for Rule-Based (computed from confusion matrix)
# Ground truth: Major 22.1%, Moderate 73.5%, Minor 4.4%
PER_CLASS_PRECISION = {
    'Major': 50.1,  # (502 correct / 1002 predicted as Major)
    'Moderate': 74.6,  # (6844 correct / 9170 predicted as Moderate) 
    'Minor': 5.7,  # (56 correct / 978 predicted as Minor)
}

# Confusion matrix (Rule-Based, test set n=11,150)
# Rows = True labels, Cols = Predicted labels
# Order: Major, Moderate, Minor
# VERIFIED from validate_against_ddinter.py execution
CONFUSION_MATRIX = np.array([
    [502, 1895, 67],     # True Major (n=2464): 502 correct, 1895→Mod, 67→Minor
    [494, 6844, 855],    # True Moderate (n=8193): 494→Major, 6844 correct, 855→Minor
    [6, 431, 56],        # True Minor (n=493): 6→Major, 431→Mod, 56 correct
])

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
BASELINE_COLOR = '#E74C3C'  # Red
CALIBRATED_COLOR = '#27AE60'  # Green
COLORS = {'Major': '#e74c3c', 'Moderate': '#f39c12', 'Minor': '#2ecc71'}


def create_4panel_figure():
    """Create 4-panel validation figure with distinct analyses."""
    
    fig = plt.figure(figsize=(14, 11), dpi=100)
    
    # =========================================================================
    # Panel A: Method Comparison Bar Chart (Exact Accuracy)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    
    methods = list(METHODS.keys())
    exact_accs = [METHODS[m]['exact'] for m in methods]
    adjacent_accs = [METHODS[m]['adjacent'] for m in methods]
    kappas = [METHODS[m]['kappa'] for m in methods]
    colors = [BASELINE_COLOR if METHODS[m]['type'] == 'baseline' else CALIBRATED_COLOR for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, exact_accs, width, label='Exact Accuracy', color=colors, edgecolor='black')
    bars2 = ax1.bar(x + width/2, adjacent_accs, width, label='Adjacent Accuracy', color=colors, alpha=0.5, edgecolor='black', hatch='//')
    
    # Add value labels
    for bar, val in zip(bars1, exact_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add kappa values below
    for i, k in enumerate(kappas):
        ax1.text(x[i], -8, f'κ={k:.3f}', ha='center', fontsize=9, color='gray')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_ylim(-12, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_title('(A) Method Performance Comparison', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # =========================================================================
    # Panel B: Confusion Matrix (Rule-Based classifier)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Normalize by row (recall-oriented)
    cm_normalized = CONFUSION_MATRIX.astype(float) / CONFUSION_MATRIX.sum(axis=1, keepdims=True) * 100
    
    labels = ['Major', 'Moderate', 'Minor']
    
    im = ax2.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = cm_normalized[i, j]
            count = CONFUSION_MATRIX[i, j]
            color = 'white' if val > 50 else 'black'
            ax2.text(j, i, f'{val:.1f}%\n({count:,})', ha='center', va='center', 
                    fontsize=10, color=color, fontweight='bold' if i == j else 'normal')
    
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    ax2.set_ylabel('True Label (DDInter)', fontsize=11)
    ax2.set_title('(B) Confusion Matrix (Rule-Based, n=11,150)', fontweight='bold', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Row-Normalized %', fontsize=10)
    
    # =========================================================================
    # Panel C: Per-Class Recall and Precision
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    
    severities = ['Major', 'Moderate', 'Minor']
    recalls = [PER_CLASS_RECALL[s] for s in severities]
    precisions = [PER_CLASS_PRECISION[s] for s in severities]
    colors_bars = [COLORS[s] for s in severities]
    
    x = np.arange(len(severities))
    width = 0.35
    
    bars_recall = ax3.bar(x - width/2, recalls, width, label='Recall (Sensitivity)', 
                          color=colors_bars, edgecolor='black')
    bars_precision = ax3.bar(x + width/2, precisions, width, label='Precision (PPV)', 
                             color=colors_bars, alpha=0.5, edgecolor='black', hatch='//')
    
    # Add sample sizes
    ns = [SCORE_STATS[s]['n'] for s in severities]
    for i, (bar, n) in enumerate(zip(bars_recall, ns)):
        ax3.text(x[i], -8, f'n={n:,}', ha='center', fontsize=9, color='gray')
    
    # Add value labels
    for bar, val in zip(bars_recall, recalls):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars_precision, precisions):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    ax3.set_ylabel('Performance (%)', fontsize=11)
    ax3.set_ylim(-12, 100)
    ax3.set_xticks(x)
    ax3.set_xticklabels(severities, fontsize=11)
    ax3.set_xlabel('DDInter Ground Truth Severity', fontsize=11)
    ax3.set_title('(C) Per-Class Recall and Precision', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.axhline(0, color='black', linewidth=0.5)
    
    # Add interpretation note
    ax3.text(0.02, 0.98, 'Moderate: High recall (83.5%)\nMajor/Minor: Class imbalance challenge',
            transform=ax3.transAxes, ha='left', va='top', fontsize=8,
            style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel D: Score Discrimination by Severity Class
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    
    severities = ['Minor', 'Moderate', 'Major']
    means = [SCORE_STATS[s]['mean'] for s in severities]
    stds = [SCORE_STATS[s]['std'] for s in severities]
    ns = [SCORE_STATS[s]['n'] for s in severities]
    colors_bars = [COLORS[s] for s in severities]
    
    x = np.arange(len(severities))
    bars = ax4.bar(x, means, yerr=stds, capsize=10, color=colors_bars,
                   edgecolor='black', linewidth=1, alpha=0.8,
                   error_kw={'linewidth': 2, 'capthick': 2})
    
    ax4.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels with n and stats
    for i, (bar, s) in enumerate(zip(bars, severities)):
        mean = SCORE_STATS[s]['mean']
        std = SCORE_STATS[s]['std']
        n = SCORE_STATS[s]['n']
        label = f'{mean:+.2f} ± {std:.2f}'
        y_pos = mean + std + 0.15 if mean > 0 else mean - std - 0.4
        va = 'bottom' if mean > 0 else 'top'
        ax4.text(bar.get_x() + bar.get_width()/2, mean + std + 0.2, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(severities, fontsize=11)
    ax4.set_xlabel('DDInter Ground Truth Severity', fontsize=11)
    ax4.set_ylabel('Keyword Score', fontsize=11)
    ax4.set_ylim(-2.5, 3.0)
    ax4.set_title('(D) Score Discrimination by Severity', fontweight='bold', fontsize=12)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add note about mechanism
    ax4.text(0.98, 0.02, 'Empirically-derived keywords\nfrom DDInter training data\nError bars = ±1 SD',
            transform=ax4.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color='gray', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Finalize
    # =========================================================================
    plt.tight_layout()
    
    # Save at publication quality
    fig.savefig(OUTPUT_DIR / 'fig_recalibration_4panel.png', dpi=1200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig_recalibration_4panel.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_recalibration_4panel.png'} (1200 DPI)")
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_recalibration_4panel.pdf'}")


if __name__ == "__main__":
    create_4panel_figure()
