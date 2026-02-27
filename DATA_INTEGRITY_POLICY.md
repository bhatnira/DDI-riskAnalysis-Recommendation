# Data Integrity Policy

**Last Updated:** February 22, 2026

## Core Principle

**All data in this repository must come from verifiable sources. No synthetic, fake, dummy, or simulated data is permitted for publication figures, tables, or reported results.**

---

## ✅ ALLOWED Uses of Random

1. **Bootstrap sampling** for confidence intervals (e.g., `np.random.choice(..., replace=True)`)
2. **Train/test splitting** with random seeds for reproducibility
3. **Graph layout positioning** for visualizations (positions, not data)
4. **Random baseline classifiers** explicitly labeled as comparison baselines
5. **Jitter for visualization** of overlapping data points (small offsets for display only)
6. **Stratified sampling** of real data for validation subsets

---

## ❌ PROHIBITED

1. **Synthetic sample generation** (`np.random.normal()`, `np.random.beta()`) to create fake data points
2. **Hardcoded "example" values** passed off as computed results
3. **"Hypothetical" or "simulated" values** in figures or tables
4. **Placeholder numbers** in publication materials
5. **Fabricated distributions** matching reported summary statistics

---

## Data Sources

All data in this repository should be traceable to:

| Source | Description | Files |
|--------|-------------|-------|
| DrugBank | Drug interactions, descriptions | `data/full database.xml`, knowledge graph CSVs |
| DDInter | Expert-validated severity labels | `external_data/ddinter/*.csv` |
| FAERS | FDA adverse event reports | `external_data/faers/` |
| SIDER | Side effect data | `external_data/sider/` |
| STRING | Protein interactions | `external_data/string/` |

---

## Validation Checklist

Before committing any analysis code or figures:

- [ ] All plotted data points come from actual computed values
- [ ] Summary statistics are computed from real data, not assumed
- [ ] Any "example" values shown (e.g., keyword weights) are actual derived values
- [ ] Figure captions accurately describe data source
- [ ] No `np.random.normal/beta/uniform` generating publication data
- [ ] Baseline comparisons are clearly labeled as such

---

## Deleted Files (Violated Policy)

The following files were removed for generating synthetic data:
- `publication_recalibration/generate_figures.py` - fake confidence distributions
- `publication_recalibration/generate_figures_updated.py` - same
- `publication_recalibration/generate_score_severity_figure.py` - synthetic scores

---

## Contact

For questions about data provenance, contact the repository maintainer.
