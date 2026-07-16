"""
Cohort Counts - генерация cohort_audit.csv для отчета
Запускается после 01_cohort_creation.py
"""

import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")

cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")

with open(DATA_DIR / "cohort_counts.json", "r") as f:
    cohort_counts = json.load(f)

print("="*70)
print("COHORT AUDIT - генерация cohort_audit.csv")
print("="*70)

# Check if cohort_audit.csv already exists from 01_cohort_creation.py
audit_csv_path = DATA_DIR / "cohort_audit.csv"
if audit_csv_path.exists():
    print(f"cohort_audit.csv уже существует, используем данные из 01_cohort_creation.py")
    audit_df_existing = pd.read_csv(audit_csv_path)
    print(f"Загружено {len(audit_df_existing)} шагов аудита")
    audit_rows = audit_df_existing.to_dict('records')
else:
    print("cohort_audit.csv не найден, генерируем из cohort_counts.json")
    audit_rows = []

n_final = cohort_counts.get("final_cohort", len(cohort))
n_treatment = cohort["treatment"].sum()
n_control = (cohort["treatment"] == 0).sum()
n_late_albumin = cohort_counts.get("exclude_late_albumin", 0)
n_missing_lactate = cohort["lactate_missing"].sum() if "lactate_missing" in cohort.columns else 0

pct_missing_cols = {}
for col in cohort.columns:
    if col not in ["stay_id", "subject_id", "hadm_id"]:
        pct_missing = 100 * cohort[col].isnull().sum() / len(cohort)
        if pct_missing > 0:
            pct_missing_cols[col] = pct_missing

pct_missing_str = "; ".join([f"{k}={v:.1f}%" for k, v in pct_missing_cols.items()]) if pct_missing_cols else "0%"

steps = [
    ("all_icu_stays", cohort_counts.get("all_icu_stays")),
    ("adults_age_ge_18", cohort_counts.get("adults_age_ge_18")),
    ("los_ge_24h", cohort_counts.get("los_ge_24h")),
    ("first_stay", cohort_counts.get("first_stay")),
    ("crystalloids_in_24h", cohort_counts.get("crystalloids_in_24h")),
    ("with_time_zero", cohort_counts.get("with_time_zero")),
    ("with_demographics", cohort_counts.get("with_demographics")),
    ("sepsis_proxy", cohort_counts.get("sepsis_proxy")),
    ("with_treatment", cohort_counts.get("with_treatment")),
    ("exclude_late_albumin", cohort_counts.get("exclude_late_albumin")),
    ("final_cohort", cohort_counts.get("final_cohort")),
]

# Only generate from scratch if not already created
if not audit_rows:
    for step_name, n_rows in steps:
        if n_rows is None:
            continue
        
        audit_rows.append({
            "step_name": step_name,
            "n_rows": n_rows,
            "n_unique_stay_id": n_rows if step_name in ["all_icu_stays", "first_stay", "crystalloids_in_24h", "with_time_zero", "sepsis_proxy", "with_treatment"] else None,
            "n_unique_hadm_id": None,
            "n_unique_subject_id": None,
            "n_treatment": n_treatment if step_name == "final_cohort" else None,
            "n_control": n_control if step_name == "final_cohort" else None,
            "n_late_albumin": n_late_albumin if step_name in ["exclude_late_albumin", "final_cohort"] else None,
            "n_missing_lactate": n_missing_lactate if step_name == "final_cohort" else None,
            "pct_missing_per_column": pct_missing_str if step_name == "final_cohort" else None,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(DATA_DIR / "cohort_audit.csv", index=False)
    print(f"\ncohort_audit.csv сохранен: {DATA_DIR / 'cohort_audit.csv'}")
    print(audit_df.to_string(index=False))
else:
    # Add final cohort stats to existing audit
    audit_df = pd.DataFrame(audit_rows)
    audit_df["n_treatment"] = audit_df.apply(lambda row: n_treatment if row["step_name"] == "final_cohort" else None, axis=1)
    audit_df["n_control"] = audit_df.apply(lambda row: n_control if row["step_name"] == "final_cohort" else None, axis=1)
    audit_df["n_late_albumin"] = audit_df.apply(lambda row: n_late_albumin if row["step_name"] in ["exclude_late_albumin", "final_cohort"] else None, axis=1)
    audit_df["n_missing_lactate"] = audit_df.apply(lambda row: n_missing_lactate if row["step_name"] == "final_cohort" else None, axis=1)
    audit_df["pct_missing_per_column"] = audit_df.apply(lambda row: pct_missing_str if row["step_name"] == "final_cohort" else None, axis=1)
    audit_df.to_csv(DATA_DIR / "cohort_audit.csv", index=False)
    print(f"\ncohort_audit.csv обновлен: {DATA_DIR / 'cohort_audit.csv'}")
    print(audit_df.to_string(index=False))

print("\n" + "="*70)
print("COHORT COUNTS - Таблица для отчета")
print("="*70)

print(f"\n{'Этап':.<50} {'N':>10}")
print("-" * 62)
for step_name, n_rows in steps:
    if n_rows is not None:
        print(f"{step_name:.<50} {n_rows:>10,}")

print("\n" + "="*70)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*70)
print(f"Финальная когорта: {n_final:,}")
print(f"Лечение (альбумин): {n_treatment:,} ({100*n_treatment/n_final:.1f}%)")
print(f"Контроль: {n_control:,}")
print(f"28-day mortality: {cohort['mortality_28days'].sum():,} ({100*cohort['mortality_28days'].mean():.1f}%)")
print(f"Late albumin исключено: {n_late_albumin:,}")
print(f"Lactate missing: {n_missing_lactate:,} ({100*n_missing_lactate/n_final:.1f}%)")
