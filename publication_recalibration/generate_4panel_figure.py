#!/usr/bin/env python3
"""
Generate 4-Panel Validation Figure for Publication

DATA SOURCE: All values from validate_against_ddinter.py output
- Test set: n=11,150 (30% of 37,164 matched pairs)
- Metrics confirmed via actual script execution

Author: DDI Research Team
Date: February 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =============================================================================
# VERIFIED DATA FROM validate_against_ddinter.py
# =============================================================================

# Method comparison (test set n=11,150)
METHODS = {
    'Zero-Shot BART-MNLI': {'exact': 22.1, 'adjacent': 95.6, 'kappa': 0.000, 'type': 'baseline'},
    'Confidence-Weighted': {'exact': 22.1, 'adjacent': 95.6, 'kappa': 0.000, 'type': 'baseline'},
    'Rule-Based': {'exact': 66.4, 'adjacent': 99.3, 'kappa': 0.096, 'type': 'calibrated'},
    'Evidence-Based': {'exact': 66.4, 'adjacent': 99.4, 'kappa': 0.096, 'type': 'calibrated'},
}

# Score statistics by DDInter ground truth severity
SCORE_STATS = {
    'Major': {'mean': 0.926, 'std': 1.328, 'n': 2464},
    'Moderate': {'mean': 0.065, 'std': 1.477, 'n': 8193},
    'Minor': {'mean': -0.281, 'std': 1.250, 'n': 493},
}

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
BASELINE_COLOR = '#E74C3C'  # Red
CALIBRATED_COLOR = '#27AE60'  # Green
COLORS = {'Major': '#e74c3c', 'Moderate': '#f39c12', 'Minor': '#2ecc71'}


def create_4panel_figure():
    """Create 4-panel validation figure matching methods_brief.tex description."""
    
    fig = plt.figure(figsize=(15, 11.25), dpi=100)  # 1200 DPI on save
    
    # =========================================================================
    # Panel A: Slope Chart (Baseline → Calibrated accuracy improvement)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    
    baseline_acc = 22.1
    calibrated_acc = 66.4
    improvement = calibrated_acc - baseline_acc
    
    # Draw slope lines
    x = [0, 1]
    ax1.plot(x, [baseline_acc, calibrated_acc], 'o-', color=CALIBRATED_COLOR, 
             linewidth=3, markersize=12, label=f'Calibrated: {calibrated_acc}%')
    ax1.axhline(baseline_acc, color=BASELINE_COLOR, linestyle='--', linewidth=2, alpha=0.8)
    
    # Method dots at baseline
    ax1.scatter([0, 0], [22.1, 22.1], s=100, c=BASELINE_COLOR, zorder=5)
    ax1.scatter([1, 1], [66.4, 66.4], s=100, c=CALIBRATED_COLOR, zorder=5)
    
    # Labels
    ax1.annotate('Zero-Shot\nConf-Weighted\n22.1%', xy=(0, 22.1), xytext=(-0.15, 22.1),
                ha='right', va='center', fontsize=10)
    ax1.annotate('Rule-Based\nEvidence-Based\n66.4%', xy=(1, 66.4), xytext=(1.15, 66.4),
                ha='left', va='center', fontsize=10)
    
    # Improvement arrow
    ax1.annotate('', xy=(0.5, calibrated_acc), xytext=(0.5, baseline_acc),
                arrowprops=dict(arrowstyle='->', color=CALIBRATED_COLOR, lw=2.5))
    ax1.text(0.55, (baseline_acc + calibrated_acc) / 2, f'+{improvement:.1f} pp',
            fontsize=14, fontweight='bold', color=CALIBRATED_COLOR)
    
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Exact Accuracy (%)', fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Baseline Methods', 'Calibrated Methods'], fontsize=11)
    ax1.set_title('(A) Accuracy Improvement: Baseline → Calibrated', fontweight='bold', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel B: Radar Plot (Multi-metric comparison)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    
    # Metrics: Exact Acc, Adjacent Acc, Cohen's κ (scaled)
    categories = ['Exact\nAccuracy', 'Adjacent\nAccuracy', "Cohen's κ\n(×100)"]
    n_cats = len(categories)
    
    # Angles for radar
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Close the loop
    
    # Baseline values (scaled to 0-100)
    baseline_vals = [22.1, 95.6, 0.0]  # κ=0.000 → 0
    baseline_vals += baseline_vals[:1]
    
    # Calibrated values (scaled to 0-100)
    calibrated_vals = [66.4, 99.3, 9.6]  # κ=0.096 → 9.6
    calibrated_vals += calibrated_vals[:1]
    
    ax2.plot(angles, baseline_vals, 'o-', linewidth=2, color=BASELINE_COLOR, label='Baseline')
    ax2.fill(angles, baseline_vals, alpha=0.2, color=BASELINE_COLOR)
    ax2.plot(angles, calibrated_vals, 'o-', linewidth=2, color=CALIBRATED_COLOR, label='Calibrated')
    ax2.fill(angles, calibrated_vals, alpha=0.2, color=CALIBRATED_COLOR)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('(B) Performance Radar: Baseline vs Calibrated', fontweight='bold', 
                  fontsize=12, y=1.08)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # =========================================================================
    # Panel C: Bar Chart (All 4 methods comparison)
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    
    methods = ['Zero-Shot\nBART-MNLI', 'Confidence\nWeighted', 'Rule-Based', 'Evidence-Based']
    exact_accs = [22.1, 22.1, 66.4, 66.4]
    colors = [BASELINE_COLOR, BASELINE_COLOR, CALIBRATED_COLOR, CALIBRATED_COLOR]
    
    bars = ax3.bar(methods, exact_accs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, exact_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Exact Accuracy (%)', fontsize=12)
    ax3.set_ylim(0, 85)
    ax3.set_title('(C) Method Comparison (Test Set n=11,150)', fontweight='bold', fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.axhline(y=66.4, color=CALIBRATED_COLOR, linestyle=':', alpha=0.5)
    ax3.axhline(y=22.1, color=BASELINE_COLOR, linestyle=':', alpha=0.5)
    
    # Legend
    baseline_patch = mpatches.Patch(color=BASELINE_COLOR, label='Baseline (22.1%)')
    calibrated_patch = mpatches.Patch(color=CALIBRATED_COLOR, label='Calibrated (66.4%)')
    ax3.legend(handles=[baseline_patch, calibrated_patch], loc='upper left')
    
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
        label = f'{mean:+.2f} ± {std:.2f}\nn={n:,}'
        ax4.text(bar.get_x() + bar.get_width()/2, mean + std + 0.2, label,
                ha='center', va='bottom', fontsize=10)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(severities, fontsize=11)
    ax4.set_xlabel('DDInter Ground Truth Severity', fontsize=12)
    ax4.set_ylabel('Keyword Score', fontsize=12)
    ax4.set_ylim(-2.5, 3.5)
    ax4.set_title('(D) Score Discrimination by Severity Class', fontweight='bold', fontsize=12)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add note about variance
    ax4.text(0.98, 0.02, 'Error bars = ±1 SD\nHigh σ limits classification accuracy',
            transform=ax4.transAxes, ha='right', va='bottom', fontsize=9,
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


if __name__ == '__main__':
    print("="*60)
    print("GENERATING 4-PANEL VALIDATION FIGURE")
    print("="*60)
    print("\nData source: validate_against_ddinter.py")
    print(f"Test set: n=11,150")
    print(f"Baseline accuracy: 22.1%")
    print(f"Calibrated accuracy: 66.4% (+44.3 pp)")
    print()
    create_4panel_figure()
    print("\n" + "="*60)
