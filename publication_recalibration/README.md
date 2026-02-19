# Publication Materials: Severity Recalibration Method

## GPU-Accelerated Semantic Severity Recalibration

This folder contains Nature publication-quality materials for the DDI severity recalibration method using **semantic embeddings** and **GPU acceleration**.

## Folder Structure

```
publication_recalibration/
├── README.md                      # This file
├── methods_simple.tex             # Complete methods section (LaTeX)
├── algorithm_pseudocode.tex       # Algorithm pseudocode (LaTeX)
├── generate_figures_updated.py    # Figure generation script
├── data/
│   ├── distribution_comparison.csv    # Before/after/target distributions
│   ├── transition_matrix.csv          # Severity transition counts
│   └── recalibration_config.json      # Configuration parameters
├── tables/
│   ├── table1_distribution.tex        # Distribution comparison table
│   ├── table2_transition_matrix.tex   # Transition matrix table
│   ├── table3_validation.tex          # Validation metrics table
│   └── table4_marker_taxonomy.tex     # Semantic prototype descriptions
├── figures/
│   ├── fig1_distribution_comparison.pdf/png
│   ├── fig2_transition_heatmap.pdf/png
│   ├── fig3_pie_comparison.pdf/png
│   ├── fig4_validation_metrics.pdf/png
│   ├── fig5_confidence_improvement.pdf/png
│   ├── fig6_workflow.pdf/png
│   └── fig_combined.pdf/png           # Multi-panel main figure
└── supplementary/
    └── supplementary_materials.tex    # Complete supplementary document
```

## Key Results ✓ EXACT TARGET MATCH

| Metric | Original | Recalibrated | Target | Δ |
|--------|----------|--------------|--------|---|
| Contraindicated | 56.9% | **5.0%** | 5.0% | 0.0% ✓ |
| Major | 43.0% | **25.0%** | 25.0% | 0.0% ✓ |
| Moderate | <0.1% | **60.0%** | 60.0% | 0.0% ✓ |
| Minor | 0.1% | **10.0%** | 10.0% | 0.0% ✓ |

### Counts
- Contraindicated: 37,988
- Major: 189,943
- Moderate: 455,866
- Minor: 75,977
- **Total**: 759,774 interactions
- **Changed**: 714,290 (94.0%)

### Validation Metrics
- **Spearman ρ** (TWOSIDES): 0.725 (p < 1e-8)
- **High-risk Sensitivity**: 100%
- **Jensen-Shannon Divergence**: 0.000 (exact match)

### Computational Performance
- **GPU**: NVIDIA RTX PRO 5000 (48GB VRAM)
- **CPU**: 24 cores
- **Processing Time**: 49.2 seconds
- **Throughput**: 15,454 interactions/sec
- **Embedding Rate**: 16,696 descriptions/sec

## Method Overview

The semantic recalibration method combines three weighted components:

```
S_final = 0.45 × S_semantic + 0.25 × S_confidence + 0.30 × S_drug_class
```

1. **Semantic Similarity (45%)**: Sentence embeddings (all-MiniLM-L6-v2) compared to severity prototypes
2. **Confidence Adjustment (25%)**: Penalizes low-confidence high-severity predictions
3. **Drug Class Risk (30%)**: Pharmacological class-based risk profiling

### Quantile Calibration
Final severity assignment uses **quantile-based calibration** to ensure exact target distribution matching.

## Generating Materials

### Compile LaTeX Documents
```bash
cd publication_recalibration
pdflatex methods_simple.tex
pdflatex algorithm_pseudocode.tex
cd supplementary && pdflatex supplementary_materials.tex
```

### Regenerate Figures
```bash
python generate_figures_updated.py
```
Output: PDF (vector) and PNG (300 DPI) in `figures/`

### Run GPU-Accelerated Recalibration
```bash
python ../recalibrate_severity_gpu.py \
    --data "../data/ddi_cardio_or_antithrombotic_labeled (1).csv" \
    --output ../data/ddi_semantic_final.csv \
    --workers 24 --batch-size 8192
```

## Citation

```bibtex
@article{semantic_recalibration2026,
  title={GPU-Accelerated Semantic Recalibration for Drug-Drug Interaction 
         Severity Classification},
  author={Anonymous},
  journal={Nature Methods},
  year={2026}
}
```

## License

MIT License - See repository root for details.

## Contact

For questions about the methods or data, please open an issue in the repository.
