# Быстрый старт

## 1. Проверка зависимостей

```bash
pip install polars pandas scikit-learn scipy matplotlib seaborn statsmodels joblib
```

## 2. Запуск пайплайна

Выполняй ноутбуки последовательно:

```bash
# 1. Создание когорты (5-10 мин)
python 01_cohort_creation.py

# 2. Propensity score matching (10-15 мин)
python 02_propensity_matching.py

# 3. AIPW анализ (15-20 мин)
python 03_aipw_analysis.py

# 4. CATE анализ (10-15 мин)
python 04_cate_analysis.py

# 5. Сводка результатов (1 мин)
python 05_summary_results.py
```

Или в Jupyter:
```
%run 01_cohort_creation.py
%run 02_propensity_matching.py
%run 03_aipw_analysis.py
%run 04_cate_analysis.py
%run 05_summary_results.py
```

## 3. Проверка результатов

После выполнения проверь файлы:
- `cohort_sepsis.csv` - финальная когорта
- `ps_matching_results.pkl` - результаты matching/IPW
- `aipw_results.pkl` - результаты AIPW
- `cate_results.pkl` - результаты CATE
- `*.png` - визуализации

## 4. Ожидаемые результаты

Сравни свои результаты с таблицей:

| Метод | Ожидается | Твои |
|-------|-----------|------|
| AIPW ATE | ~0% (CI включает 0) | ? |
| CATE Septic Shock | < 0% (benefit) | ? |
| Propensity AUC | > 0.75 | ? |

## 5. Если что-то не так

1. **ATE значительно отличается от 0:**
   - Проверь определение сепсиса/септического шока
   - Убедись что лактат импутирован правильно
   - Проверь баланс ковариат после matching

2. **CATE не показывает benefit для septic shock:**
   - Проверь что septic_shock = сепсис + вазопрессоры + lactate > 2
   - Увеличь количество итераций в RandomizedSearchCV

3. **Плохой баланс ковариат:**
   - Уменьши калипер (сейчас 0.2 * std)
   - Попробуй другое соотношение treated:control (1:1, 1:2)

## 6. Контакты

Если возникли вопросы - смотри:
- `README.md` - полная документация
- `AI_AGENT_PROMPT.md` - детальное описание методологии
- Оригинал: https://github.com/soda-inria/causal_ehr_mimic
