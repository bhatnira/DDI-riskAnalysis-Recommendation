# Publication Materials - Summary

Generated on: 2026-02-12

## Directory Structure

```
publication/
├── figures/          # 24 files (8 figures × 3 formats)
├── figures_advanced/ # 45 files (15 figures × 3 formats)
├── tables/           # 21 files (7 tables × 3 formats)
├── data/             # 5 files
├── statistics/       # 4 files
└── README.md
```

**Total: 100 files**

---

## Basic Figures

| Figure | Description | Formats |
|--------|-------------|---------|
| Fig 1 | DDI Severity Distribution | PNG, PDF, SVG |
| Fig 2 | Network Centrality Metrics (Degree, Weighted, Betweenness, PRI) | PNG, PDF, SVG |
| Fig 3 | PRI Distribution by Drug Category | PNG, PDF, SVG |
| Fig 4 | ATC Classification Analysis | PNG, PDF, SVG |
| Fig 5 | DDI Heatmap (Top 20 Risk Drugs) | PNG, PDF, SVG |
| Fig 6 | Risk Metrics Correlation Matrix | PNG, PDF, SVG |
| Fig 7 | Multi-Objective Recommender Performance | PNG, PDF, SVG |
| Fig 8 | Interaction Phenotype Analysis | PNG, PDF, SVG |

---

## Advanced Figures (Network Visualizations)

| Figure | Description | Formats |
|--------|-------------|---------|
| Fig Adv 1 | Force-Directed Network Visualization | PNG, PDF, SVG |
| Fig Adv 2 | Network Community Detection & Clustering | PNG, PDF, SVG |
| Fig Adv 3 | Degree Distribution Analysis (Power Law) | PNG, PDF, SVG |
| Fig Adv 4 | PRI Components Radar Chart | PNG, PDF, SVG |
| Fig Adv 5 | Severity Hierarchy Sunburst | PNG, PDF, SVG |
| Fig Adv 6 | Drug Embedding t-SNE Visualization | PNG, PDF, SVG |
| Fig Adv 7 | Cardiovascular Drug Interaction Flow | PNG, PDF, SVG |
| Fig Adv 8 | Risk Stratification Analysis | PNG, PDF, SVG |
| Fig Adv 9 | Severity Confidence Analysis | PNG, PDF, SVG |
| Fig Adv 10 | ATC Hierarchy Visualization | PNG, PDF, SVG |
| Fig Adv 11 | Phenotype Co-occurrence Network | PNG, PDF, SVG |
| Fig Adv 12 | Centrality Measures Comparison | PNG, PDF, SVG |
| Fig Adv 13 | Recommendation Impact Analysis | PNG, PDF, SVG |
| Fig Adv 14 | Network Robustness Under Attack | PNG, PDF, SVG |
| Fig Adv 15 | High-Risk Drug PRI Heatmap | PNG, PDF, SVG |

---

## Tables

| Table | Description | Formats |
|-------|-------------|---------|
| Table 1 | Dataset Summary Statistics | CSV, TEX, MD |
| Table 2 | DDI Severity Distribution Statistics | CSV, TEX, MD |
| Table 3 | Top 20 Highest Risk Drugs by PRI | CSV, TEX, MD |
| Table 4 | Drug Risk Network Statistics | CSV, TEX, MD |
| Table 5 | ATC Classification Distribution | CSV, TEX, MD |
| Table 6 | Interaction Phenotype Summary | CSV, TEX, MD |
| Table 7 | Sample Multi-Objective Recommendations | CSV, TEX, MD |

## Data Files

| File | Description |
|------|-------------|
| `all_drugs_with_metrics.csv` | All 4,314 drugs with network metrics |
| `all_drugs_with_metrics.json` | Same data in JSON format |
| `all_interactions_with_metrics.csv` | All 379,917 interactions with PRI |
| `network_metrics.json` | Network-level statistics |
| `pri_scores_detailed.csv` | PRI component breakdown |

## Statistics

| File | Description |
|------|-------------|
| `descriptive_statistics.json` | Dataset & metric summaries |
| `correlation_analysis.json` | Pearson & Spearman correlations |
| `correlation_matrix.csv` | Correlation matrix |
| `hypothesis_tests.json` | Statistical hypothesis test results |

## Key Findings

### Dataset Summary
- **Total DDI Records**: 759,774
- **Unique Drugs**: 4,314
- **Unique Interactions**: 379,917
- **Cardiovascular Drugs**: 450 (includes antithrombotic drugs)
- **Non-Cardiovascular Drugs**: 3,864

### Severity Distribution
- Contraindicated: 432,226 (56.9%)
- Major: 326,716 (43.0%)
- Moderate: 24 (0.003%)
- Minor: 808 (0.1%)

### Network Statistics
- Network Density: 0.000041
- Mean PRI Score: 0.254
- Max PRI Score: 0.843 (Procaine)

### Top 5 Risk Drugs by PRI
1. Procaine (PRI: 0.8434)
2. Indomethacin (PRI: 0.7173)
3. Lidocaine (PRI: 0.7137)
4. Benzocaine (PRI: 0.6661)
5. Quinidine (PRI: 0.6401)

## Usage

### Regenerate Materials
```bash
python generate_publication_materials.py
```

### Custom Output Directory
```bash
python generate_publication_materials.py --output custom_dir
```

### Custom Data Path
```bash
python generate_publication_materials.py --data path/to/ddi_data.csv
```

## LaTeX Integration

Include tables in your LaTeX document:
```latex
\input{publication/tables/table1_dataset_summary.tex}
\input{publication/tables/table2_severity_statistics.tex}
```

Include figures:
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{publication/figures/fig1_severity_distribution.pdf}
  \caption{Distribution of DDI Severity Levels}
  \label{fig:severity}
\end{figure}
```
