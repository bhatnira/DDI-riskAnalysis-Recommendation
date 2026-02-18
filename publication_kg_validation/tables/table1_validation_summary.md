| Metric                        | Value          | Interpretation                                 |
|:------------------------------|:---------------|:-----------------------------------------------|
| Sample Size                   | 87             | Drugs with ≥100 FAERS reports                  |
| Spearman ρ (vs Serious Count) | 0.296          | KG risk vs log(serious event count)            |
| 95% CI                        | [0.090, 0.483] | Bootstrap confidence interval (n=1000)         |
| P-value                       | 5.37e-03       | Statistical significance                       |
| Spearman ρ (vs Serious Ratio) | -0.182         | KG risk vs serious/total ratio (biased metric) |
| AUC-ROC                       | 0.625          | Discriminative ability for high-risk drugs     |