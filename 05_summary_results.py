"""
Summary Results - генерация итоговых таблиц
Запускается после всех анализов
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")

print("="*70)
print("SUMMARY RESULTS - генерация итоговых таблиц")
print("="*70)

ate_results_rows = []

try:
    with open(DATA_DIR / "ps_matching_results.pkl", "rb") as f:
        matching_results = pickle.load(f)
    
    ate_results_rows.append({
        "method": "Matching",
        "ate": matching_results["ate_matching"],
        "ci_lower": matching_results["ci_matching"][0],
        "ci_upper": matching_results["ci_matching"][1],
        "significant": "Yes" if (matching_results["ci_matching"][0] > 0 or matching_results["ci_matching"][1] < 0) else "No",
    })
    
    ate_results_rows.append({
        "method": "IPW",
        "ate": matching_results["ate_ipw"],
        "ci_lower": matching_results["ci_ipw"][0],
        "ci_upper": matching_results["ci_ipw"][1],
        "significant": "Yes" if (matching_results["ci_ipw"][0] > 0 or matching_results["ci_ipw"][1] < 0) else "No",
    })
    
    print(f"\nMatching/IPW результаты загружены")
except Exception as e:
    print(f"\nWARNING: Не удалось загрузить ps_matching_results.pkl: {e}")

try:
    with open(DATA_DIR / "aipw_results.pkl", "rb") as f:
        aipw_results = pickle.load(f)
    
    ate_results_rows.append({
        "method": "AIPW",
        "ate": aipw_results["ate_aipw"],
        "ci_lower": aipw_results["ci_aipw"][0],
        "ci_upper": aipw_results["ci_aipw"][1],
        "significant": "Yes" if (aipw_results["ci_aipw"][0] > 0 or aipw_results["ci_aipw"][1] < 0) else "No",
    })
    print(f"AIPW результаты загружены")
except Exception as e:
    print(f"WARNING: Не удалось загрузить aipw_results.pkl: {e}")

ate_results_df = pd.DataFrame(ate_results_rows)
ate_results_df.to_csv(DATA_DIR / "ate_results.csv", index=False)
print(f"\nate_results.csv сохранен: {DATA_DIR / 'ate_results.csv'}")

print("\n" + "="*70)
print("ATE RESULTS - Сводная таблица")
print("="*70)
print(ate_results_df.to_string(index=False))

try:
    overlap_df = pd.read_csv(DATA_DIR / "overlap_summary.csv")
    print(f"\nOverlap statistics загружены")
    
    pct_outside = overlap_df["pct_outside"].values[0]
    print(f"\n{'='*70}")
    print("OVERLAP LIMITATION WARNING")
    print("="*70)
    print(f"Пациентов вне overlap region [0.1, 0.9]: {int(overlap_df['n_outside_overlap'].values[0]):,} ({pct_outside:.1f}%)")
    print(f"Пациентов в overlap region: {int(overlap_df['n_inside_overlap'].values[0]):,} ({100-pct_outside:.1f}%)")
    
    if pct_outside > 40:
        print(f"\n⚠️  SERIOUS LIMITATION: {pct_outside:.1f}% когорты вне overlap region")
        print("Это может указывать на систематические различия между treated и control")
        print("Результаты могут быть не generalizable на всю популяцию")
    
    print(f"\nMean propensity score:")
    print(f"  Treated: {overlap_df['mean_ps_treated'].values[0]:.3f}")
    print(f"  Control: {overlap_df['mean_ps_control'].values[0]:.3f}")
    
except Exception as e:
    print(f"\nWARNING: Не удалось загрузить overlap_summary.csv: {e}")

print("\n" + "="*70)
print("ВСЕ ВЫХОДНЫЕ ФАЙЛЫ")
print("="*70)

output_files = [
    "cohort_audit.csv",
    "balance_smd.csv",
    "overlap_summary.csv",
    "ate_results.csv",
    "matching_summary.csv",
]

for fname in output_files:
    fpath = DATA_DIR / fname
    if fpath.exists():
        print(f"✓ {fname}")
    else:
        print(f"✗ {fname} - НЕ НАЙДЕН")

print("="*70)
