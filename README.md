# Causal Inference Analysis for Albumin in Sepsis (MIMIC-IV v3.1)

**ВЕРСИЯ 2.0** - Исправлено по требованиям ментора

Этот проект воспроизводит causal inference анализ из статьи **Doutreligne et al. "Step-by-step causal analysis of EHRs to ground decision-making"** (PLOS Digital Health, 2025) на данных **MIMIC-IV v3.1**.

---

## PICOT (обновлено)

| Компонент | Определение |
|-----------|-------------|
| **P** (Population) | ICU пациенты с sepsis proxy, возраст >= 18, LOS >= 24ч |
| **I** (Intervention) | Кристаллоиды + альбумин в течение 24ч после time zero |
| **C** (Control) | Кристаллоиды без альбумина в течение 24ч после time zero |
| **O** (Outcome) | Смерть в течение 28 дней после time zero |
| **T** (Time zero) | **Первое введение кристаллоидов** (НЕ ICU intime!) |

### Target Trial Эмуляция

- **Inclusion**: Первый ICU stay, возраст >= 18, LOS >= 24ч, кристаллоиды в первые 24ч
- **Exclusion**: Альбумин > 24ч после time zero (late albumin)
- **Treatment assignment**: Наблюдательное (не рандомизированное)
- **Confounding adjustment**: Propensity score matching, IPW, AIPW

---

## Оригинальная статья

- **Название:** Step-by-step causal analysis of EHRs to ground decision-making
- **Авторы:** Doutreligne M., Struja T., Abecassis J., Morgand C., Celi L., Varoquaux G.
- **Журнал:** PLOS Digital Health, 2025
- **Код авторов:** https://github.com/soda-inria/causal_ehr_mimic
- **Версия MIMIC у авторов:** v2.2

## Ключевые отличия от оригинала

| Аспект | Оригинал (авторы) | Эта реализация (v2.0) |
|--------|-------------------|----------------------|
| **MIMIC-IV версия** | v2.2 | v3.1 |
| **SAPSII score** | Доступен | Charlson Comorbidity Index |
| **Лактат** | Из dedicated таблиц | first_day_sofa + labevents + missing indicator |
| **Time zero** | ICU intime | **First crystalloid** (исправлено!) |
| **Treatment window** | Не указано | [time_zero, time_zero + 24h] |
| **Late albumin** | Не указано | Исключен из когорты |
| **Propensity model** | Не указана | **LogReg baseline** + GradientBoosting |
| **Overlap check** | Не указан | Порог [0.1, 0.9] |
| **IPW weights** | Не указано | Stabilized + clipping (>10) |

## Структура пайплайна

```
notebook_new/
├── requirements.txt            # Зависимости Python
├── 01_cohort_creation.py       # Создание когорты (time_zero = crystalloid)
├── 02_propensity_matching.py   # LogReg PS, Matching, IPW, SMD, ESS
├── 03_aipw_analysis.py         # Doubly Robust (AIPW) анализ
├── 04_cate_analysis.py         # CATE (heterogeneous effects)
└── 05_summary_results.py       # Сводка результатов
```

## Запуск

### Требования

```bash
pip install polars pandas scikit-learn scipy matplotlib seaborn statsmodels joblib
```

### Порядок выполнения

1. **01_cohort_creation.py** - Создание когорты
   - **Time zero = first crystalloid** (НЕ ICU intime!)
   - Фильтрация: возраст >= 18, LOS >= 24ч, кристаллоиды в первые 24ч
   - Исключение: late albumin (>24ч после time zero)
   - Определение сепсиса (suspicion + organ dysfunction)
   - Конфаундеры за 24ч ДО time_zero
   - Missing indicators для lactate
   - **Выход:** `cohort_sepsis.csv` + cohort counts

2. **02_propensity_matching.py** - Propensity score анализ
   - **Propensity model: LogReg baseline** + GradientBoosting
   - Overlap check: порог [0.1, 0.9]
   - Calibration plot
   - Matching 1:1 nearest neighbor (without replacement)
   - **SMD до/после matching и до/после IPW**
   - IPW: stabilized weights + clipping (>10)
   - **Effective Sample Size (ESS)**
   - Bootstrap CI (500 повторов)
   - **Выход:** `ps_matching_results.pkl`

3. **03_aipw_analysis.py** - Doubly Robust анализ
   - Outcome models (T-learner)
   - AIPW формула
   - Bootstrap CI
   - Сравнение с IPW и Matching
   - **Выход:** `aipw_results.pkl`

4. **04_cate_analysis.py** - Гетерогенность эффектов
   - CATE для septic shock
   - CATE для возраста (<60, >=60)
   - CATE для пола и расы
   - Bootstrap CI для подгрупп
   - **Выход:** `cate_results.pkl`

5. **05_summary_results.py** - Сводка
   - Forest plot всех оценок
   - Сравнение с результатами авторов
   - Проверка критериев успеха
   - **Выход:** Визуализации и summary

## Ожидаемые результаты

Согласно статье авторов:

| Метод | Ожидаемый ATE | Интерпретация |
|-------|---------------|---------------|
| **AIPW** | ~0% (CI включает 0) | Нет значимого эффекта в средней популяции |
| **IPW** | ~0% (CI включает 0) | Нет значимого эффекта |
| **Matching** | ~0% (CI включает 0) | Нет значимого эффекта |
| **CATE Septic Shock** | -2% до -5% | Benefit от альбумина |
| **CATE Age < 60** | ~0% | Нет эффекта |
| **CATE Age >= 60** | ~0% | Нет эффекта |

## Конфаундеры

Используется полный набор из методологии авторов:

### Демография
- admission_age
- Female, Male
- White, Black, Hispanic
- emergency_admission
- insurance_Medicare, insurance_Medicaid

### Severity scores
- SOFA
- Charlson Comorbidity Index
- Lactate (с импутацией)

### Витальные признаки (mean за 24ч ДО time_zero)
- Heart rate, SpO2, Mean BP, Temperature, Respiratory rate

### Лекарства (флаги за 24ч ДО time_zero)
- Carbapenems, Aminoglycosides, Beta-lactams, Glycopeptides
- **has_vasopressors УБРАН** (это часть определения сепсиса!)

### Процедуры (за 24ч ДО time_zero)
- RRT (renal replacement therapy), Ventilation

### Missing indicators
- `lactate_missing` (для >50% пропусков)

## Критерии успеха

Анализ считается успешным если:

1. ✅ ATE оценки (AIPW, IPW, Matching) имеют CI включающий 0
2. ✅ CATE для septic shock < 0 (показывает benefit)
3. ✅ Propensity model: LogReg baseline (не гонимся за AUC)
4. ✅ Overlap: доля вне [0.1, 0.9] < 5%
5. ✅ <10% ковариат имеют |SMD| > 0.1 после matching/IPW

---

## Ограничения (честно!)

### Не решено на текущий момент:

1. **Immortal time bias** — частично решено правильным time_zero
2. **Confounding by indication** — residual confounding возможен
3. **Weak overlap** — проверяется, trimming по [0.1, 0.9]
4. **Missingness** — KNN импутация + missing indicators
5. **Late albumin** — исключен из когорты
6. **SAPS-II недоступен** — заменен на Charlson (v3.1 limitation)

### Отличия от RCT:

- Наблюдательные данные (не рандомизированные)
- Остаточное смещение возможно даже после PS adjustment
- Требуется внешняя валидация

## Визуализации

Пайплайн генерирует следующие фигуры:

- `propensity_distribution.png` - Распределение propensity scores
- `love_plot.png` - Баланс ковариат до/после matching
- `bootstrap_ipw.png` - Bootstrap распределение IPW ATE
- `bootstrap_aipw.png` - Bootstrap распределение AIPW ATE
- `cate_forest_plot.png` - Forest plot CATE по подгруппам
- `cate_septic_shock.png` - CATE для septic shock
- `cate_age.png` - CATE для возраста
- `forest_plot_ate.png` - Сравнение всех методов
- `forest_plot_cate.png` - CATE по подгруппам

## Отличия в результатах

Если ваши результаты отличаются от ожидаемых:

1. **Проверьте определение сепсиса** - должно быть SOFA >= 2 + suspicion in [-24h, +24h]
2. **Проверьте определение септического шока** - сепсис + вазопрессоры + lactate > 2
3. **Убедитесь что лактат импутирован** - >50% пропусков в оригинале
4. **Проверьте баланс ковариат** - |SMD| < 0.1 для ≥90% признаков
5. **Настройте гиперпараметры** - используйте больше итераций в RandomizedSearchCV

## Ссылки

- [Оригинальная статья](https://journals.plos.org/plosdigitalhealth/article?id=10.1371/journal.pdig.0000721)
- [Код авторов](https://github.com/soda-inria/causal_ehr_mimic)
- [MIMIC-IV документация](https://mimic.mit.edu/)
- [Causal inference методология](https://arxiv.org/abs/2308.01605)
