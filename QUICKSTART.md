# Quick Start - запуск пайплайна

## Требования

```bash
pip install -r requirements.txt
```

## Данные

Данные MIMIC-IV v3.1 должны быть расположены по пути:
```
/Users/faritsharafutdinov/untitled folder/mimic-iv-3.1/mimiciv_as_parquet/
```

## Запуск пайплайна

```bash
# 1. Создание когорты (time_zero = first crystalloid)
python 01_cohort_creation.py

# 2. Propensity score matching + IPW
python 02_propensity_matching.py

# 3. AIPW (Doubly Robust)
python 03_aipw_analysis.py

# 4. CATE анализ (подгруппы)
python 04_cate_analysis.py

# 5. Сводка результатов
python 05_summary_results.py
```

## Выходные файлы

После запуска создаются:

### Данные
- `cohort_sepsis.csv` - финальная когорта с конфаундерами
- `ps_matching_results.pkl` - результаты matching + IPW
- `aipw_results.pkl` - результаты AIPW
- `cate_results.pkl` - результаты CATE

### Визуализации
- `propensity_distribution.png` - overlap propensity scores
- `calibration_plot.png` - калибровка propensity model
- `love_plot.png` - баланс ковариат (SMD)
- `bootstrap_ipw.png` - bootstrap распределение IPW
- `bootstrap_aipw.png` - bootstrap распределение AIPW
- `cate_forest_plot.png` - CATE по подгруппам
- `cate_septic_shock.png` - CATE для septic shock
- `cate_age.png` - CATE для возраста
- `forest_plot_ate.png` - сравнение всех методов

## Отчет

Для генерации cohort counts:
```bash
python cohort_counts.py
```

## Интерпретация результатов

См. таблицу в конце вывода `02_propensity_matching.py` и `05_summary_results.py`:
- Naive difference
- IPW ATE
- Matching ATT
- AIPW ATE

Все оценки приведены с 95% bootstrap CI.
