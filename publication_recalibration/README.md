# Publication Materials: DDI Severity Classification

## Evidence-Based Severity Validation Pipeline

This folder contains publication-quality materials for the DDI severity classification method using **external validation** against the DDInter database.

## Folder Structure

```
publication_recalibration/
├── README.md                               # This file
├── SEVERITY_CLASSIFICATION_PIPELINE.md    # Complete pipeline documentation
├── methods_publication_grade.tex          # Publication methods (LaTeX)
├── data/
│   ├── complete_method_comparison.csv     # All methods compared
│   └── validation_results.csv             # Validation metrics
└── figures/
    └── [validation figures]
```

## Key Results ✓ EXTERNALLY VALIDATED

### Method Comparison

| Method | DDInter Exact | DDInter Adj | DDInter κ |
|--------|--------------|-------------|----------|
| Zero-Shot BART (Baseline) | 12.4% | 92.4% | -0.000 |
| Confidence-Weighted | 71.1% | 99.9% | -0.080 |
| Rule-Based Keywords | 73.4% | 99.9% | -0.002 |
| **Evidence-Based (Final)** | **71.6%** | **98.8%** | **+0.107** |

### Final Severity Distribution (Evidence-Based)

| Severity | Count | Percentage |
|----------|-------|------------|
| Contraindicated | 28,614 | 3.8% |
| Major | 179,568 | 23.6% |
| Moderate | 550,145 | 72.4% |
| Minor | 1,447 | 0.2% |
| **Total** | **759,774** | **100%** |
### Validation Source

| Source | n | Description |
|--------|---|-------------|
| **DDInter** | 6,381 | Literature-curated severity (Xiong et al., 2022, NAR) |

### Validation Metrics
- **DDInter Exact Accuracy**: 71.6%
- **DDInter Adjacent Accuracy**: 98.8%
- **DDInter Cohen's κ**: +0.107 (only positive κ)

## Method Overview

The Evidence-Based Classifier uses hierarchical clinical rules:

```
Priority 1: FDA Black Box Warnings → Contraindicated
Priority 2: QT Prolongation Risk → Major (CredibleMeds)
Priority 3: Dual Antithrombotic → Major (CHEST Guidelines)
Priority 4: ACC/AHA Guidelines → Class-specific
Priority 5: Clinical Pattern Matching → Moderate (default)
```

### Why Evidence-Based Won

| Criterion | Evidence-Based | Other Methods |
|-----------|---------------|---------------|
| Cohen's κ > 0 | **+0.107** | All negative |
| External validation | DDInter (n=6,381) | DDInter only |
| Distribution RMSE | **5.3%** | 7.4-68.5% |

## Running the Pipeline

### Generate Method Comparison
```bash
python publication_evidence_based_classifier.py
```

### Compile LaTeX Methods
```bash
cd publication_recalibration
pdflatex methods_publication_grade.tex
```

## Citation

```bibtex
@article{evidence_based_ddi_2026,
  title={Evidence-Based Drug-Drug Interaction Severity Classification 
         with External Validation},
  author={Anonymous},
  journal={Journal of Managed Care & Specialty Pharmacy},
  year={2026}
}
```

## References

1. Xiong G, et al. (2022). DDInter: an online drug–drug interaction database. NAR.
2. Tatonetti NP, et al. (2012). Data-driven prediction of drug effects. Sci Transl Med.
3. January CT, et al. (2019). AHA/ACC/HRS Atrial Fibrillation Guideline. Circulation.
4. Kearon C, et al. (2016). CHEST Antithrombotic Guidelines. CHEST.

*Last Updated: February 20, 2026*
