# DDI Risk Analysis Application - Publication Materials

This folder contains publication-ready documentation and sample outputs for the DDI Risk Analysis Application with Alternative Recommendations.

## Folder Structure

```
publication_app/
├── README.md                    # This file
├── data/                        # Sample data and analysis outputs
│   └── sample_analyses.json     # JSON outputs from sample runs
├── figures/                     # Screenshots and visualizations
│   └── app_screenshots/         # Application interface captures
├── tables/                      # Generated tables in various formats
├── method_brief/                # Methods documentation
│   └── methods_brief.tex        # LaTeX methods brief
└── supplementary/               # Supplementary materials
    └── supplementary_materials.tex  # LaTeX supplementary doc
```

## Application Overview

The DDI Risk Analysis Application is a comprehensive web-based tool for analyzing drug-drug interactions built on a knowledge graph of 759,774 interactions from DrugBank, DDInter, and FAERS databases.

### Key Features

1. **Multi-Modal Drug Input**
   - Text entry (comma/plus/newline separated)
   - Clinical narrative extraction
   - Prescription image OCR

2. **Risk Analysis**
   - Overall polypharmacy risk score
   - Individual Polypharmacy Risk Index (PRI)
   - Severity-classified interactions

3. **Alternative Recommendations**
   - Alternative Recommendation Score (ARS)
   - Therapeutic substitution suggestions
   - Interactive "what-if" analysis

4. **LLM Chat Assistant**
   - Context-aware clinical Q&A
   - LLaMA 7B model via Ollama (local inference)

## Sample Drug Combinations Analyzed

### High-Risk Cardiovascular Regimen
- **Drugs**: Warfarin, Aspirin, Atorvastatin, Lisinopril, Metoprolol
- **Risk Level**: HIGH
- **Key Interactions**: Warfarin-Aspirin (Major), Warfarin-Atorvastatin (Moderate)

### Diabetes Management Regimen
- **Drugs**: Metformin, Glipizide, Lisinopril, Atorvastatin
- **Risk Level**: MODERATE
- **Key Interactions**: Metformin-Lisinopril (Minor)

### Pain Management Regimen
- **Drugs**: Tramadol, Sertraline, Alprazolam
- **Risk Level**: HIGH
- **Key Interactions**: Tramadol-Sertraline (Major - Serotonin Syndrome Risk)

## Running the Application

```bash
# From the repository root
source .venv/bin/activate
python ddi_app.py

# Access at http://localhost:7860
```

## Generating Sample Outputs

```bash
# Run the sample analysis script
python publication_app/generate_sample_outputs.py
```

## Building Documentation

```bash
cd publication_app/method_brief
pdflatex methods_brief.tex
pdflatex methods_brief.tex  # Run twice for TOC

cd ../supplementary
pdflatex supplementary_materials.tex
pdflatex supplementary_materials.tex
```

## Citation

If using this application or documentation in your research, please cite:

```bibtex
@software{ddi_risk_analyzer,
  title={DDI Risk Analysis Application with Alternative Recommendations},
  author={Drug-Drug Interaction Knowledge Graph Project},
  year={2026},
  url={https://github.com/bhatnira/knowledge-graph}
}
```

## License

This project is provided for academic and research purposes.
