# Project Handover Document

## Project Overview

This project reproduces causal inference analysis from:
- **Paper:** "Step-by-step causal analysis of EHRs to ground decision-making"
- **Authors:** Doutreligne et al., PLOS Digital Health, 2025
- **Data:** MIMIC-IV v3.1

## What's Included

### Code
- `01_cohort_creation.py` - Cohort extraction
- `02_propensity_matching.py` - Propensity models, IPW, matching
- `03_aipw_analysis.py` - Doubly robust estimation
- `04_cate_analysis.py` - Heterogeneous effects
- `05_summary_results.py` - Summary tables
- `06_generate_report.py` - Auto-report generation
- `cohort_counts.py` - Cohort audit

### Data
- `cohort_sepsis.parquet` - Final cohort (~23,000 patients)
- `cohort_audit.csv` - Step-by-step counts
- `ate_results.csv` - ATE estimates
- `balance_smd.csv` - Covariate balance
- `overlap_summary.csv` - Overlap statistics
- `matching_audit.csv` - Matching pairs

### Visualizations
- `propensity_distribution.png`
- `calibration_plot.png`
- `love_plot.png`
- `bootstrap_ipw.png`
- `bootstrap_aipw.png`
- `cate_*.png`
- `forest_plot_*.png`

### Reports
- `FINAL_REPORT.md` - Auto-generated results
- `README.md` - Project documentation
- `STATUS_REPORT.md` - Detailed methodology
- `IMPLEMENTATION_PLAN.md` - Implementation plan

## How to Reproduce

1. Install dependencies: `pip install -r requirements.txt`
2. Configure paths: Edit `config.yaml`
3. Run pipeline: `bash run_pipeline.sh`
4. View results: Open `FINAL_REPORT.md`

## Key Results

| Method | ATE (28-day mortality) | 95% CI |
|--------|------------------------|--------|
| IPW | -2.62% | [-4.34%, -0.97%] |
| Matching | -7.05% | [-9.09%, -4.91%] |
| AIPW | -9.87% | [-11.03%, -8.52%] |

**Interpretation:** All methods show benefit from albumin (negative ATE = reduced mortality).

## Known Limitations

1. LOS filter uses future information (selection bias)
2. ~45% patients outside overlap region
3. Residual imbalance after IPW (18% with |SMD|>0.1)
4. CATE for septic shock not significant (CI includes 0)

## Methodological Fixes Implemented

1. **LOS Bias Documented:** Added warning comments in code and README
2. **Bootstrap Fixed:** Now retrains propensity model on each iteration
3. **Cross-Fitting:** Propensity scores are now out-of-fold predictions
4. **Matching Audit:** Added stay_id mapping and repeated controls tracking
5. **Cohort Audit:** Real unique counts at each step

## Configuration

All parameters are in `config.yaml`:
- Random seeds
- Bootstrap iterations
- Propensity thresholds
- Matching caliper
- Cohort definitions

## Contact

For questions about this project, refer to:
- `README.md` for methodology
- `STATUS_REPORT.md` for detailed discussion
- `FINAL_REPORT.md` for results summary
