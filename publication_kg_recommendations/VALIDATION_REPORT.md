# KG-Based Recommendation System Validation Report

Generated: 2026-02-16 22:29

## 1. Substitution Quality Metrics

Validated against clinical guideline substitution pairs.

| Metric | Value |
|--------|-------|
| Drugs Evaluated | 27 |
| Precision@1 | 0.148 |
| Precision@5 | 0.133 |
| Precision@10 | 0.085 |
| Recall@5 | 0.580 |
| Recall@10 | 0.765 |
| MRR | 0.352 |
| NDCG@10 | 0.449 |

## 2. Risk Reduction Validation

Validates that recommendations reduce polypharmacy risk.

| Metric | Value |
|--------|-------|
| Test Cases | 7 |
| Cases Improved | 6 |
| Improvement Rate | 85.7% |
| Mean Risk Reduction | 0.201 |
| Max Risk Reduction | 0.431 |

### Test Case Details

| Original Regimen | Risk | Optimized | New Risk | Reduction |
|-----------------|------|-----------|----------|-----------|
| warfarin, aspirin, metoprolol | 0.634 | acenocoumarol, aspirin, metoprolol | 0.517 | 0.117 |
| clopidogrel, omeprazole, atorvastatin | 0.587 | prasugrel, omeprazole, atorvastatin | 0.404 | 0.183 |
| simvastatin, amiodarone, lisinopril | 0.576 | ezetimibe, amiodarone, indapamide | 0.541 | 0.035 |
| methotrexate, ibuprofen, prednisone | 0.653 | cytarabine, diclofenac, prednisone | 0.408 | 0.245 |
| lithium, ibuprofen, hydrochlorothiazide | 0.669 | lithium, acetylsalicylic acid, hydrochlorothiazide | 0.473 | 0.195 |
| fluoxetine, tramadol, alprazolam | 0.408 | fluvoxamine, tramadol, alprazolam | 0.408 | -0.000 |
| digoxin, amiodarone, furosemide | 0.772 | digoxin, amiodarone, sulfamethizole | 0.340 | 0.431 |