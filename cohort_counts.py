"""
Cohort Counts - генерация таблицы для отчета
Запускается после 01_cohort_creation.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")

# Загружаем когорту
cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")

print("="*70)
print("COHORT COUNTS - Таблица для отчета")
print("="*70)

# Базовая статистика
print(f"\n{'Этап':.<50} {'N':>10}")
print("-" * 62)
print(f"{'Финальная когорта':.<50} {len(cohort):>10,}")
print(f"{'Лечение (альбумин)':.<50} {cohort['treatment'].sum():>10,}")
print(f"{'Контроль':.<50} {(cohort['treatment'] == 0).sum():>10,}")
print(f"{'28-day mortality':.<50} {cohort['mortality_28days'].sum():>10,}")

# Демография
print("\n" + "="*70)
print("ДЕМОГРАФИЯ")
print("="*70)

print(f"\n{'Возраст, mean (std)':.<50} {cohort['admission_age'].mean():.1f} ({cohort['admission_age'].std():.1f})")
print(f"{'Пол (Female), n (%)':.<50} {cohort['Female'].sum()} ({100*cohort['Female'].mean():.1f}%)")
print(f"{'Раса (White), n (%)':.<50} {cohort['White'].sum()} ({100*cohort['White'].mean():.1f}%)")
print(f"{'Emergency admission, n (%)':.<50} {cohort['emergency_admission'].sum()} ({100*cohort['emergency_admission'].mean():.1f}%)")

# Сепсис
print("\n" + "="*70)
print("СЕПСИС")
print("="*70)

if 'septic_shock' in cohort.columns:
    print(f"{'Септический шок, n (%)':.<50} {cohort['septic_shock'].sum()} ({100*cohort['septic_shock'].mean():.1f}%)")

# Лечение
print("\n" + "="*70)
print("ЛЕЧЕНИЕ")
print("="*70)

print(f"{'Альбумин (treatment), n (%)':.<50} {cohort['treatment'].sum()} ({100*cohort['treatment'].mean():.1f}%)")

# Исходы
print("\n" + "="*70)
print("ИСХОДЫ")
print("="*70)

print(f"{'28-day mortality, n (%)':.<50} {cohort['mortality_28days'].sum()} ({100*cohort['mortality_28days'].mean():.1f}%)")

# Пропуски
print("\n" + "="*70)
print("ПРОПУСКИ (до импутации)")
print("="*70)

if 'lactate_missing' in cohort.columns:
    print(f"{'Lactate missing, n (%)':.<50} {cohort['lactate_missing'].sum()} ({100*cohort['lactate_missing'].mean():.1f}%)")

print("\n" + "="*70)
