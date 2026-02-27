# Distribution of DDI Severity Levels (Evidence-Based Classification)

## Validation: DDInter 71.6% exact accuracy, 98.8% adjacent, κ=+0.107

 Severity Level   Count Percentage  Severity Weight
Contraindicated  28,614      3.77%             10.0
          Major 179,568     23.63%              7.0
       Moderate 550,145     72.41%              4.0
          Minor   1,447      0.19%              1.0

## Comparison: Baseline vs Evidence-Based

| Severity | Zero-Shot (Baseline) | Evidence-Based (Final) | DDInter Ground Truth |
|----------|---------------------|------------------------|---------------------|
| Contraindicated | 56.9% | 3.8% | 4.2% |
| Major | 43.0% | 23.6% | 24.7% |
| Moderate | 0.0% | 72.4% | 71.1% |
| Minor | 0.1% | 0.2% | 0.0% |

## Validation Metrics (DDInter, n=6,381)

| Metric | Value |
|--------|-------|
| Exact Accuracy | 71.6% |
| Adjacent Accuracy | 98.8% |
| Cohen's κ | +0.107 |
| Macro F1 | 0.366 |