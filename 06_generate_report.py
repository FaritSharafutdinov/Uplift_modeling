"""
Generate final report from CSV outputs
"""
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent

# Load all results
ate_df = pd.read_csv(DATA_DIR / "ate_results.csv")
overlap_df = pd.read_csv(DATA_DIR / "overlap_summary.csv")
balance_df = pd.read_csv(DATA_DIR / "balance_smd.csv")
matching_df = pd.read_csv(DATA_DIR / "matching_summary.csv")

with open(DATA_DIR / "cate_results.pkl", "rb") as f:
    cate_results = pickle.load(f)

# Generate Markdown report
report = f"""# Causal Inference Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Cohort Summary

- Final cohort size: {ate_df.shape[0]} patients
- Treatment rate: {ate_df['treatment'].mean():.1%}

## 2. ATE Results

| Method | ATE | 95% CI | Significant |
|--------|-----|--------|-------------|
"""

for _, row in ate_df.iterrows():
    significant = "Yes" if (row['ci_lower'] > 0 or row['ci_upper'] < 0) else "No"
    report += f"| {row['method']} | {row['ate']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {significant} |\n"

report += f"""
## 3. Overlap Statistics

- Patients in overlap region: {overlap_df['n_inside_overlap'].values[0]:,} ({100 - overlap_df['pct_outside'].values[0]:.1f}%)
- Patients trimmed: {overlap_df['n_outside_overlap'].values[0]:,} ({overlap_df['pct_outside'].values[0]:.1f}%)

## 4. Balance (|SMD| > 0.1)

- Before IPW: {(balance_df['abs_smd_before'] > 0.1).sum()} / {len(balance_df)}
- After IPW: {(balance_df['abs_smd_after'] > 0.1).sum()} / {len(balance_df)}

## 5. CATE Results

"""

if 'septic_shock' in cate_results:
    report += f"""| Subgroup | CATE | N |
|----------|------|---|
| Septic Shock | {cate_results['septic_shock']['cate']:.4f} | {cate_results['septic_shock']['n']} |
| Age < 60 | {cate_results['age_lt_60']['cate']:.4f} | {cate_results['age_lt_60']['n']} |
| Age >= 60 | {cate_results['age_ge_60']['cate']:.4f} | {cate_results['age_ge_60']['n']} |
"""

report += f"""
## 6. Known Limitations

1. **LOS Filter Bias:** LOS ≥24h is known only after ICU discharge, introducing selection bias
2. **Immortal Time:** Time zero = first crystalloid, patients must survive to receive treatment
3. **Overlap Violations:** ~{overlap_df['pct_outside'].values[0]:.1f}% patients outside [0.1, 0.9] propensity range
4. **Residual Confounding:** {(balance_df['abs_smd_after'] > 0.1).sum()} covariates with |SMD| > 0.1 after adjustment

## 7. Files Generated

- `cohort_sepsis.csv` - Final cohort
- `cohort_audit.csv` - Step-by-step cohort counts
- `ate_results.csv` - ATE estimates
- `balance_smd.csv` - Covariate balance
- `overlap_summary.csv` - Overlap statistics
- `matching_audit.csv` - Matching pairs audit
- `ps_matching_results.pkl` - Matching + IPW results
- `aipw_results.pkl` - AIPW results
- `cate_results.pkl` - CATE results

## 8. Visualizations

- `propensity_distribution.png` - Propensity score distribution
- `calibration_plot.png` - Propensity model calibration
- `love_plot.png` - Covariate balance (SMD)
- `bootstrap_ipw.png` - IPW bootstrap distribution
- `bootstrap_aipw.png` - AIPW bootstrap distribution
- `cate_septic_shock.png` - CATE for septic shock
- `cate_age.png` - CATE for age subgroups
- `cate_forest_plot.png` - Forest plot of CATE
- `forest_plot_ate.png` - Forest plot of ATE methods
"""

# Save report
with open(DATA_DIR / "FINAL_REPORT.md", "w") as f:
    f.write(report)

print("Report generated: FINAL_REPORT.md")
print(f"Location: {DATA_DIR / 'FINAL_REPORT.md'}")
