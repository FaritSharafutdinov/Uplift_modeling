# Causal Inference Analysis for Albumin in Sepsis (MIMIC-IV v3.1)

Этот проект воспроизводит causal inference анализ из статьи **Doutreligne et al. "Step-by-step causal analysis of EHRs to ground decision-making"** (PLOS Digital Health, 2025) на данных **MIMIC-IV v3.1**.

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
| **SAPSII score** | Доступен | Использован Charlson Comorbidity Index |
| **Лактат** | Из dedicated таблиц | Из vitalsign/first_day_sofa |
| **Антибиотики** | По ATC кодам | По названиям препаратов |
| **Импутация** | Не указана | KNNImputer (k=10) |

## Структура пайплайна

```
notebook_new/
├── 01_cohort_creation.py       # Создание когорты с конфаундерами
├── 02_propensity_matching.py   # Propensity score matching + IPW
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
   - Фильтрация ICU stays (возраст >= 18, LOS >= 1 день)
   - Определение сепсиса (SOFA >= 2 + suspicion of infection)
   - Определение септического шока (сепсис + вазопрессоры + лактат > 2)
   - Извлечение конфаундеров (демография, витальные, лабораторные, коморбидность)
   - Импутация пропусков (KNN)
   - **Выход:** `cohort_sepsis.csv`

2. **02_propensity_matching.py** - Propensity score анализ
   - Обучение propensity model (GradientBoosting с RandomizedSearchCV)
   - Проверка positivity assumption
   - Matching с калипером (0.2 * std)
   - IPW (Inverse Propensity Weighting)
   - Bootstrap доверительные интервалы
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

### Витальные признаки (mean за 24ч до ICU)
- Heart rate
- SpO2
- Mean BP
- Temperature
- Respiratory rate

### Лекарства (флаги до ICU)
- Carbapenems
- Aminoglycosides
- Beta-lactams
- Glycopeptides
- Vasopressors

### Процедуры (флаги до ICU)
- RRT (renal replacement therapy)
- Ventilation

## Критерии успеха

Анализ считается успешным если:

1. ✅ ATE оценки (AIPW, IPW, Matching) имеют CI включающий 0
2. ✅ CATE для septic shock < 0 (показывает benefit)
3. ✅ Propensity model AUC-ROC > 0.75
4. ✅ <10% ковариат имеют |SMD| > 0.1 после matching

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
