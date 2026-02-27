# Table 3: Evidence-Based Classifier Performance Results

## DDInter Validation (n=6,381)

| Metric                       | Value     | Note                     |
|:-----------------------------|:----------|:-------------------------|
| Exact Accuracy               | 71.6%     | Literature validation    |
| Adjacent Accuracy (±1)       | 98.8%     | Within one level         |
| Macro F1                     | 0.366     | Class-balanced           |
| Cohen's Kappa                | **+0.107**| Only positive κ          |

## Method Comparison Summary

| Method                  | DDInter Exact | DDInter Adj | Cohen's κ |
|:------------------------|:--------------|:------------|:----------|
| Zero-Shot BART          | 12.4%         | 92.4%       | -0.000    |
| Confidence-Weighted     | 71.1%         | 99.9%       | -0.080    |
| Rule-Based Keywords     | 73.4%         | 99.9%       | -0.002    |
| **Evidence-Based**      | **71.6%**     | **98.8%**   | **+0.107**|

## Distribution Alignment

| Severity        | Evidence-Based | DDInter | RMSE  |
|:----------------|:---------------|:--------|:------|
| Contraindicated | 3.8%           | 4.2%    |       |
| Major           | 23.6%          | 24.7%   |       |
| Moderate        | 72.4%          | 71.1%   |       |
| Minor           | 0.2%           | 0.0%    |       |
| **Total RMSE**  |                |         | **5.3%** |

## Improvement Over Baseline

| Metric              | Baseline | Final   | Δ           |
|:--------------------|:---------|:--------|:------------|
| DDInter Accuracy    | 12.4%    | 71.6%   | +59.2 pp    |
| Adjacent Accuracy   | 92.4%    | 98.8%   | +6.4 pp     |
| Cohen's κ           | -0.000   | +0.107  | Now positive|
| Distribution RMSE   | 68.5%    | 5.3%    | -63.2 pp    |
