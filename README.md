# Causal Inference Analysis for Albumin in Sepsis (MIMIC-IV v3.1)

Проект воспроизводит causal inference анализ из статьи **Doutreligne et al. "Step-by-step causal analysis of EHRs to ground decision-making"** (PLOS Digital Health, 2025) на данных **MIMIC-IV v3.1**.

## PICOT

| Компонент | Определение |
|-----------|-------------|
| **P** (Population) | ICU пациенты с sepsis proxy, возраст >= 18, LOS >= 24ч |
| **I** (Intervention) | Кристаллоиды + альбумин в течение 24ч после time zero |
| **C** (Control) | Кристаллоиды без альбумина в течение 24ч после time zero |
| **O** (Outcome) | Смерть в течение 28 дней после time zero |
| **T** (Time zero) | **Первое введение кристаллоидов** |

### Target Trial Эмуляция

- **Inclusion**: Первый ICU stay, возраст >= 18, LOS >= 24ч, кристаллоиды в первые 24ч
- **Exclusion**: Альбумин > 24ч после time zero (late albumin)
- **Confounding adjustment**: Propensity score matching, IPW, AIPW

### Known Biases & Limitations

1. **Immortal Time Bias:**
   - Time zero = first crystalloid
   - Patients must survive to receive crystalloids
   - This excludes patients who die before crystalloid administration

2. **Selection Bias (LOS Filter):**
   - LOS ≥24h is known only after ICU discharge
   - Patients who die <24h are excluded
   - This may bias toward healthier patients

3. **Confounding by Indication:**
   - Sicker patients more likely to receive albumin
   - Adjusted via 22 confounders + IPW/AIPW
   - Residual confounding possible

4. **Overlap Violations:**
   - ~45% of patients outside [0.1, 0.9] propensity range
   - Results generalizable only to overlap population

---

## Оригинальная статья

- **Название:** Step-by-step causal analysis of EHRs to ground decision-making
- **Авторы:** Doutreligne M., Struja T., Abecassis J., Morgand C., Celi L., Varoquaux G.
- **Журнал:** PLOS Digital Health, 2025
- **Код авторов:** https://github.com/soda-inria/causal_ehr_mimic
- **Версия MIMIC у авторов:** v2.2

## Ключевые отличия от оригинала

| Аспект | Оригинал (авторы) | Эта реализация |
|--------|-------------------|----------------|
| **MIMIC-IV версия** | v2.2 | v3.1 |
| **SAPSII score** | Доступен | Charlson Comorbidity Index |
| **Лактат** | Из dedicated таблиц | first_day_sofa + labevents + missing indicator |
| **Time zero** | ICU intime | **First crystalloid** |
| **Treatment window** | Не указано | [time_zero, time_zero + 24h] |
| **Late albumin** | Не указано | Исключен из когорты |
| **Propensity model** | Не указана | LogReg + GradientBoosting + LightGBM |
| **Overlap check** | Не указан | Порог [0.1, 0.9] |
| **IPW weights** | Не указано | Stabilized + clipping (>10) |

## Структура пайплайна

```
notebook_new/
├── requirements.txt            # Зависимости Python
├── config.yaml                 # Конфигурация
├── run_pipeline.sh             # Script для запуска
├── Makefile                    # Make команды
├── 01_cohort_creation.py       # Создание когорты
├── 02_propensity_matching.py   # Propensity score matching + IPW
├── 03_aipw_analysis.py         # Doubly Robust (AIPW) анализ
├── 04_cate_analysis.py         # CATE (heterogeneous effects)
├── 05_summary_results.py       # Сводка результатов
└── 06_generate_report.py       # Авто-генерация отчета
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths

Edit `config.yaml`:
```yaml
MIMIC_DIR: "/path/to/your/mimic-iv-3.1/mimiciv_as_parquet"
OUTPUT_DIR: "/path/to/your/output/directory"
```

### 3. Run full pipeline

```bash
# Option A: Using shell script
bash run_pipeline.sh

# Option B: Using Makefile
make all

# Option C: Manual step-by-step
python 01_cohort_creation.py
python 02_propensity_matching.py
python 03_aipw_analysis.py
python 04_cate_analysis.py
python 05_summary_results.py
python 06_generate_report.py
```

### 4. View results

```bash
# Open final report
open FINAL_REPORT.md

# View ATE results
cat ate_results.csv

# View cohort counts
cat cohort_audit.csv
```

### 5. Reproducibility

All random seeds are fixed in `config.yaml`. For exact reproduction:
```bash
export PYTHONHASHSEED=42
python 01_cohort_creation.py
# ... rest of pipeline
```

### Требования

```bash
pip install -r requirements.txt
```

### Порядок выполнения

1. **01_cohort_creation.py** - Создание когорты
   - Time zero = first crystalloid
   - Фильтрация: возраст >= 18, LOS >= 24ч
   - Исключение: late albumin (>24ч после time zero)
   - Определение сепсиса (suspicion + organ dysfunction)
   - Конфаундеры за 24ч ДО time_zero
   - Missing indicators для lactate
   - **Выход:** `cohort_sepsis.csv`

2. **02_propensity_matching.py** - Propensity score анализ
   - Три модели: LogReg + GradientBoosting + LightGBM
   - Overlap check: порог [0.1, 0.9]
   - Matching 1:1 nearest neighbor
   - SMD до/после matching и IPW
   - IPW: stabilized weights + clipping
   - Effective Sample Size (ESS)
   - Bootstrap CI (500 повторов)
   - **Выход:** `ps_matching_results.pkl`

3. **03_aipw_analysis.py** - Doubly Robust анализ
   - Outcome models (T-learner)
   - AIPW формула
   - Bootstrap CI (1000 повторов)
   - **Выход:** `aipw_results.pkl`

4. **04_cate_analysis.py** - Гетерогенность эффектов
   - CATE для septic shock
   - CATE для возраста (<60, >=60)
   - **Выход:** `cate_results.pkl`

5. **05_summary_results.py** - Сводка
   - Forest plot всех оценок
   - **Выход:** Визуализации и summary

## Ожидаемые результаты

Согласно статье авторов:

| Метод | Ожидаемый ATE | Интерпретация |
|-------|---------------|---------------|
| **AIPW** | ~0% (CI включает 0) | Нет значимого эффекта |
| **IPW** | ~0% (CI включает 0) | Нет значимого эффекта |
| **Matching** | ~0% (CI включает 0) | Нет значимого эффекта |
| **CATE Septic Shock** | -2% до -5% | Benefit от альбумина |
| **CATE Age < 60** | ~0% | Нет эффекта |
| **CATE Age >= 60** | ~0% | Нет эффекта |

## Конфаундеры

### Демография
- admission_age, Female, Male
- White, Black, Hispanic
- emergency_admission
- insurance_Medicare, insurance_Medicaid

### Severity scores
- Charlson Comorbidity Index
- Lactate (с импутацией)

### Витальные признаки (mean за 24ч ДО time_zero)
- Heart rate, SpO2, Mean BP, Temperature, Respiratory rate

### Лекарства (флаги за 24ч ДО time_zero)
- Carbapenems, Aminoglycosides, Beta-lactams, Glycopeptides

### Процедуры
- RRT, Ventilation

### Missing indicators
- `lactate_missing`

## Критерии успеха

1. ATE оценки (AIPW, IPW, Matching) имеют CI включающий 0
2. CATE для septic shock < 0 (показывает benefit)
3. Overlap: доля вне [0.1, 0.9] < 5%
4. <10% ковариат имеют |SMD| > 0.1 после matching/IPW

---

## Ограничения

1. **Immortal time bias** — частично решено правильным time_zero
2. **Confounding by indication** — residual confounding возможен
3. **Weak overlap** — проверяется, trimming по [0.1, 0.9]
4. **Missingness** — KNN импутация + missing indicators
5. **SAPS-II недоступен** — заменен на Charlson (v3.1 limitation)

### Отличия от RCT

- Наблюдательные данные (не рандомизированные)
- Остаточное смещение возможно даже после PS adjustment
- Требуется внешняя валидация

## Визуализации

- `propensity_distribution.png` - Распределение propensity scores
- `calibration_plot.png` - Калибровка propensity model
- `love_plot.png` - Баланс ковариат до/после matching
- `bootstrap_ipw.png` - Bootstrap распределение IPW ATE
- `bootstrap_aipw.png` - Bootstrap распределение AIPW ATE
- `cate_forest_plot.png` - Forest plot CATE по подгруппам
- `cate_septic_shock.png` - CATE для septic shock
- `cate_age.png` - CATE для возраста
- `forest_plot_ate.png` - Сравнение всех методов
- `forest_plot_cate.png` - CATE по подгруппам

## Ссылки

- [Оригинальная статья](https://journals.plos.org/plosdigitalhealth/article?id=10.1371/journal.pdig.0000721)
- [Код авторов](https://github.com/soda-inria/causal_ehr_mimic)
- [MIMIC-IV документация](https://mimic.mit.edu/)
