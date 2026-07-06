# План доработок - Осталось сделать

## ✅ Уже сделано:

1. **PICOT** - обновлен в README
2. **Propensity models** - LogReg + GBM + LightGBM добавлены
3. **Overlap/ESS/SMD** - всё реализовано
4. **Bootstrap CI** - есть для всех методов
5. **AIPW** - готов
6. **CATE** - готов
7. **Reproducibility** - README, requirements.txt, QUICKSTART

---

## ⚠️ Нужно запустить заново:

### 1. Пересоздать когорту (ОБЯЗАТЕЛЬНО)

```bash
python3 01_cohort_creation.py
```

**Что исправится:**
- ✅ Time zero = first crystalloid (НЕ ICU intime)
- ✅ Treatment window [time_zero, time_zero + 24h]
- ✅ LOS >= 24 часа (явный фильтр)
- ✅ Late albumin исключен
- ✅ lactate_missing indicator добавлен
- ✅ has_vasopressors убран из конфаундеров
- ✅ Cohort counts на каждом этапе

**Выход:** `cohort_sepsis.csv` (новая версия)

---

### 2. Запустить propensity analysis

```bash
python3 02_propensity_matching.py
```

**Что будет:**
- ✅ Три модели: LogReg + GBM + LightGBM
- ✅ Сравнение AUC-ROC
- ✅ Overlap [0.1, 0.9]
- ✅ Calibration plot
- ✅ ESS (Effective Sample Size)
- ✅ SMD до/после IPW и matching
- ✅ Bootstrap CI 95%

**Выход:** `ps_matching_results.pkl`, визуализации

---

### 3. Запустить AIPW

```bash
python3 03_aipw_analysis.py
```

**Важно:** Предварительно проверить что в 03_aipw_analysis.py используется правильный confounder_vars (без has_vasopressors)

---

### 4. Запустить CATE

```bash
python3 04_cate_analysis.py
```

---

### 5. Сводка

```bash
python3 05_summary_results.py
```

---

## 📊 Для финального отчета:

```bash
# Сгенерировать cohort counts
python3 cohort_counts.py
```

---

## 🔧 Критичные проверки перед отправкой:

1. **Cohort counts** - убедиться что числа адекватные
   - Все ICU stays -> взрослые -> LOS >=24ч -> сепсис -> treatment/control

2. **Overlap** - доля вне [0.1, 0.9] должна быть < 5%

3. **Balance** - |SMD| < 0.1 для ≥90% ковариат после matching/IPW

4. **ATE оценки** - сравнить с оригиналом (Doutreligne et al.)
   - Ожидается: ATE ≈ 0% (CI включает 0)
   - CATE Septic Shock < 0 (benefit)

5. **Feature set** - убедиться что has_vasopressors нет в конфаундерах

---

## 📁 Файлы для проверки:

- [x] `README.md` - PICOT, методология
- [x] `QUICKSTART.md` - инструкция
- [x] `requirements.txt` - зависимости
- [x] `01_cohort_creation.py` - исправлен
- [x] `02_propensity_matching.py` - исправлен
- [ ] `03_aipw_analysis.py` - проверить confounder_vars
- [ ] `04_cate_analysis.py` - проверить confounder_vars
- [ ] `05_summary_results.py` - готов

---

## 🚀 Команды для запуска:

```bash
cd "/Users/faritsharafutdinov/untitled folder/notebook_new"

# 1. Когорта
python3 01_cohort_creation.py

# 2. Propensity
python3 02_propensity_matching.py

# 3. AIPW
python3 03_aipw_analysis.py

# 4. CATE
python3 04_cate_analysis.py

# 5. Summary
python3 05_summary_results.py

# 6. Cohort counts для отчета
python3 cohort_counts.py
```

---

## 📝 Заметки:

- **LightGBM** установится автоматически если есть (`pip install lightgbm`)
- Если LightGBM не нужен - просто не устанавливай, скрипт продолжит работу без него
- **Лактат** - в текущей когорте уже импутирован (0 пропусков), missing indicator будет в новой версии
