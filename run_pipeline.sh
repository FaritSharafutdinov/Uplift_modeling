#!/bin/bash
# Causal Inference Pipeline - Full Run

set -e  # Exit on error

echo "=========================================="
echo "Causal Inference Pipeline"
echo "=========================================="

# Step 1: Cohort Creation
echo "[1/6] Creating cohort..."
python 01_cohort_creation.py
python cohort_counts.py

# Step 2: Propensity Matching + IPW
echo "[2/6] Propensity score matching + IPW..."
python 02_propensity_matching.py

# Step 3: AIPW Analysis
echo "[3/6] AIPW (Doubly Robust) analysis..."
python 03_aipw_analysis.py

# Step 4: CATE Analysis
echo "[4/6] CATE (Heterogeneous effects)..."
python 04_cate_analysis.py

# Step 5: Summary Results
echo "[5/6] Generating summary..."
python 05_summary_results.py

# Step 6: Generate Report
echo "[6/6] Generating final report..."
python 06_generate_report.py

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - cohort_sepsis.csv"
echo "  - ps_matching_results.pkl"
echo "  - aipw_results.pkl"
echo "  - cate_results.pkl"
echo "  - FINAL_REPORT.md"
echo ""
echo "Visualizations:"
echo "  - propensity_distribution.png"
echo "  - calibration_plot.png"
echo "  - love_plot.png"
echo "  - bootstrap_ipw.png"
echo "  - bootstrap_aipw.png"
echo "  - cate_*.png"
echo "  - forest_plot_*.png"
