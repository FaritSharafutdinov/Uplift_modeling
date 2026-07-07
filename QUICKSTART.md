# Quick Start

## Требования

```bash
pip install -r requirements.txt
```

## Данные

MIMIC-IV v3.1 должны быть расположены по пути:
```
/Users/faritsharafutdinov/untitled folder/mimic-iv-3.1/mimiciv_as_parquet/
```

## Запуск

```bash
# 1. Создание когорты
python 01_cohort_creation.py

# 2. Propensity score matching + IPW
python 02_propensity_matching.py

# 3. AIPW (Doubly Robust)
python 03_aipw_analysis.py

# 4. CATE анализ
python 04_cate_analysis.py

# 5. Сводка результатов
python 05_summary_results.py
```

## Выходные файлы

### Данные
- `cohort_sepsis.csv` - финальная когорта
- `ps_matching_results.pkl` - matching + IPW
- `aipw_results.pkl` - AIPW
- `cate_results.pkl` - CATE

### Визуализации
- `propensity_distribution.png`
- `calibration_plot.png`
- `love_plot.png`
- `bootstrap_ipw.png`
- `bootstrap_aipw.png`
- `cate_forest_plot.png`
- `cate_septic_shock.png`
- `cate_age.png`
- `forest_plot_ate.png`

## Интерпретация результатов

См. таблицу в выводе `05_summary_results.py`:
- Naive difference
- IPW ATE
- Matching ATT
- AIPW ATE

Все оценки с 95% bootstrap CI.
