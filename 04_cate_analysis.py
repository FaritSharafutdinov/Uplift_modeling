# %%
"""
CATE (Conditional Average Treatment Effect) анализ
Адаптировано из causal_ehr_mimic авторов под MIMIC-IV v3.1

Анализируем гетерогенность эффекта лечения для разных подгрупп:
- Septic shock (ключевой effect modifier!)
- Возраст (<60 vs >=60)
- Пол (Female vs Male)
- Раса (White vs Non-White)
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Пути
DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")

# Загружаем когорту
cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")
print(f"Когорта: {cohort.shape}")

# %% [markdown]
# ## Шаг 1: Создаем effect modifiers

# %%
# Ключевые effect modifiers из методологии авторов
# 1. Septic shock (сепсис + вазопрессоры + лактат > 2)
# Проверяем есть ли уже septic_shock
if "septic_shock" not in cohort.columns:
    cohort["septic_shock"] = (
        (cohort.get("sepsis", False) == True) &
        (cohort.get("has_vasopressors", 0) == 1) &
        (cohort.get("lactate_final", 0) > 2.0)
    ).astype(int)

# 2. Возраст (порог 60 лет, не 65!)
age_threshold = 60
cohort["age_ge_60"] = (cohort["admission_age"] >= age_threshold).astype(int)

# 3. Пол
cohort["female"] = cohort["Female"]

# 4. Раса
cohort["white"] = cohort["White"]

print("Effect modifiers:")
print(f"  Septic shock: {cohort['septic_shock'].mean():.2%}")
print(f"  Age >= 60: {cohort['age_ge_60'].mean():.2%}")
print(f"  Female: {cohort['female'].mean():.2%}")
print(f"  White: {cohort['white'].mean():.2%}")

# %% [markdown]
# ## Шаг 2: CATE через causal forest / T-learner

# %%
# Готовим данные
confounder_vars = [
    "admission_age", "Female", "White", "Black", "Hispanic",
    "emergency_admission", "insurance_Medicare", "insurance_Medicaid",
    "sofa", "lactate_final", "charlson_comorbidity_index",
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    # "has_vasopressors",  # УБРАН - это часть определения сепсиса!
    "rrt_flag", "ventilation_flag",
    "lactate_missing",  # Missing indicator
]

available_confounders = [col for col in confounder_vars if col in cohort.columns]
X = cohort[available_confounders].values
t = cohort["treatment"].values
y = cohort["mortality_28days"].values

# Effect modifiers для стратификации
effect_modifiers = ["septic_shock", "age_ge_60", "female", "white"]

# %% [markdown]
# ## Шаг 3: T-Learner для CATE

# %%
def t_learner_cate(X, t, y, X_modifiers=None, random_state=42):
    """
    T-Learner для оценки CATE
    
    1. Обучаем outcome model для treated: mu_1(X)
    2. Обучаем outcome model для control: mu_0(X)
    3. CATE(X) = mu_1(X) - mu_0(X)
    """
    np.random.seed(random_state)
    
    # Outcome model для treated
    treated_mask = t == 1
    X_treated = X[treated_mask]
    y_treated = y[treated_mask]
    
    outcome_treated = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=random_state,
    )
    outcome_treated.fit(X_treated, y_treated)
    
    # Outcome model для control
    control_mask = t == 0
    X_control = X[control_mask]
    y_control = y[control_mask]
    
    outcome_control = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=random_state,
    )
    outcome_control.fit(X_control, y_control)
    
    # Предсказываем CATE для всех
    mu_1 = outcome_treated.predict(X)
    mu_0 = outcome_control.predict(X)
    cate = mu_1 - mu_0
    
    return cate, outcome_treated, outcome_control

# Вычисляем CATE
cate_scores, outcome_t, outcome_c = t_learner_cate(X, t, y)

print(f"CATE statistics:")
print(f"  Mean: {cate_scores.mean():.4f}")
print(f"  Std: {cate_scores.std():.4f}")
print(f"  Min: {cate_scores.min():.4f}")
print(f"  Max: {cate_scores.max():.4f}")
print(f"  Median: {np.median(cate_scores):.4f}")

# %% [markdown]
# ## Шаг 4: CATE по подгруппам

# %%
def cate_by_subgroup(cate, subgroup, subgroup_name):
    """CATE для бинарной подгруппы"""
    mask_0 = subgroup == 0
    mask_1 = subgroup == 1
    
    cate_0 = cate[mask_0].mean()
    cate_1 = cate[mask_1].mean()
    
    print(f"\n{subgroup_name}:")
    print(f"  {subgroup_name}=0 (n={mask_0.sum()}): CATE = {cate_0:.4f} ({100*cate_0:.2f}%)")
    print(f"  {subgroup_name}=1 (n={mask_1.sum()}): CATE = {cate_1:.4f} ({100*cate_1:.2f}%)")
    print(f"  Difference: {cate_1 - cate_0:.4f}")
    
    return {
        "subgroup": subgroup_name,
        "cate_0": cate_0,
        "n_0": mask_0.sum(),
        "cate_1": cate_1,
        "n_1": mask_1.sum(),
        "diff": cate_1 - cate_0,
    }

print("=== CATE ПО ПОДГРУППАМ ===")
cate_results = []

for modifier in effect_modifiers:
    if modifier in cohort.columns:
        result = cate_by_subgroup(cate_scores, cohort[modifier].values, modifier)
        cate_results.append(result)

# %% [markdown]
# ## Шаг 5: Визуализация CATE по подгруппам

# %%
plt.figure(figsize=(12, 6))

# Forest plot
y_pos = np.arange(len(cate_results) * 2)
cate_values = []
cate_labels = []
cate_errors = []

for i, result in enumerate(cate_results):
    # Group 0
    cate_values.extend([result["cate_0"], result["cate_1"]])
    cate_labels.extend([
        f"{result['subgroup']}=0 (n={result['n_0']})",
        f"{result['subgroup']}=1 (n={result['n_1']})"
    ])
    
# Горизонтальный forest plot
colors = ["blue" if v < 0 else "red" for v in cate_values]
plt.barh(np.arange(len(cate_values)), cate_values, color=colors, alpha=0.7)
plt.yticks(np.arange(len(cate_values)), cate_labels)
plt.axvline(x=0, color="black", linestyle="-", linewidth=2)
plt.xlabel("CATE (Treatment Effect on Mortality)")
plt.title("Conditional Average Treatment Effects by Subgroup\n(Negative = Benefit from Albumin)")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(DATA_DIR / "cate_forest_plot.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Шаг 6: CATE для septic shock (ключевой анализ!)

# %%
# Детальный анализ для septic shock
septic_shock_mask = cohort["septic_shock"].values == 1
no_septic_shock_mask = cohort["septic_shock"].values == 0

cate_septic_shock = cate_scores[septic_shock_mask].mean()
cate_no_septic_shock = cate_scores[no_septic_shock_mask].mean()

print("\n=== DETAIL: Septic Shock CATE ===")
print(f"Без септического шока (n={no_septic_shock_mask.sum()}): CATE = {cate_no_septic_shock:.4f}")
print(f"С септическим шоком (n={septic_shock_mask.sum()}): CATE = {cate_septic_shock:.4f}")
print(f"Разница: {cate_septic_shock - cate_no_septic_shock:.4f}")

# Ожидаемый результат из статьи:
# - Septic shock: CATE < 0 (benefit от альбумина)
# - No septic shock: CATE ≈ 0 (нет эффекта)

# %%
# Визуализация
plt.figure(figsize=(8, 5))

subgroups = ["No Septic Shock", "Septic Shock"]
cate_vals = [cate_no_septic_shock, cate_septic_shock]
ns = [no_septic_shock_mask.sum(), septic_shock_mask.sum()]
colors = ["orange" if v > 0 else "blue" for v in cate_vals]

plt.bar(subgroups, cate_vals, color=colors, alpha=0.7, edgecolor="black")
plt.axhline(y=0, color="black", linestyle="-", linewidth=2)
plt.ylabel("CATE (Treatment Effect)")
plt.title(f"CATE: Septic Shock Subgroups\n(Albumin effect on 28-day mortality)")
plt.grid(alpha=0.3, axis="y")

# Добавляем значения на бары
for i, (v, n) in enumerate(zip(cate_vals, ns)):
    plt.text(i, v + np.sign(v) * 0.01, f"{v:.3f}\n(n={n})", ha="center", va="bottom" if v > 0 else "top")

plt.tight_layout()
plt.savefig(DATA_DIR / "cate_septic_shock.png", dpi=150)


# %% [markdown]
# ## Шаг 7: CATE по возрасту

# %%
age_lt_60_mask = cohort["age_ge_60"].values == 0
age_ge_60_mask = cohort["age_ge_60"].values == 1

cate_age_lt_60 = cate_scores[age_lt_60_mask].mean()
cate_age_ge_60 = cate_scores[age_ge_60_mask].mean()

print("\n=== DETAIL: Age CATE ===")
print(f"Age < 60 (n={age_lt_60_mask.sum()}): CATE = {cate_age_lt_60:.4f}")
print(f"Age >= 60 (n={age_ge_60_mask.sum()}): CATE = {cate_age_ge_60:.4f}")

# %%
plt.figure(figsize=(8, 5))

subgroups = ["Age < 60", "Age >= 60"]
cate_vals = [cate_age_lt_60, cate_age_ge_60]
ns = [age_lt_60_mask.sum(), age_ge_60_mask.sum()]
colors = ["orange" if v > 0 else "blue" for v in cate_vals]

plt.bar(subgroups, cate_vals, color=colors, alpha=0.7, edgecolor="black")
plt.axhline(y=0, color="black", linestyle="-", linewidth=2)
plt.ylabel("CATE (Treatment Effect)")
plt.title(f"CATE: Age Subgroups\n(Threshold = {age_threshold} years)")
plt.grid(alpha=0.3, axis="y")

for i, (v, n) in enumerate(zip(cate_vals, ns)):
    plt.text(i, v + np.sign(v) * 0.01, f"{v:.3f}\n(n={n})", ha="center", va="bottom" if v > 0 else "top")

plt.tight_layout()
plt.savefig(DATA_DIR / "cate_age.png", dpi=150)


# %% [markdown]
# ## Шаг 8: Bootstrap CI для CATE по подгруппам

# %%
def bootstrap_cate_subgroup(X, t, y, subgroup, n_bootstrap=500, random_state=42):
    """Bootstrap CI для CATE в подгруппе"""
    np.random.seed(random_state)
    n = len(t)
    cate_samples = []
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        t_boot = t[idx]
        y_boot = y[idx]
        subgroup_boot = subgroup[idx]
        
        # T-learner
        cate_boot, _, _ = t_learner_cate(X_boot, t_boot, y_boot, random_state=i)
        
        # CATE для subgroup=1
        mask = subgroup_boot == 1
        if mask.sum() > 0:
            cate_subgroup = cate_boot[mask].mean()
            cate_samples.append(cate_subgroup)
    
    if len(cate_samples) == 0:
        return None, None, None
    
    cate_samples = np.array(cate_samples)
    ci_lower = np.percentile(cate_samples, 2.5)
    ci_upper = np.percentile(cate_samples, 97.5)
    
    return cate_samples.mean(), ci_lower, ci_upper

# %%
print("=== Bootstrap CI для CATE (Septic Shock) ===")
cate_ss_mean, cate_ss_ci_lower, cate_ss_ci_upper = bootstrap_cate_subgroup(
    X, t, y, cohort["septic_shock"].values, n_bootstrap=500
)

if cate_ss_mean is not None:
    print(f"Septic shock: CATE = {cate_ss_mean:.4f}, 95% CI = [{cate_ss_ci_lower:.4f}, {cate_ss_ci_upper:.4f}]")
    print(f"Значимый benefit: {'Да' if cate_ss_ci_upper < 0 else 'Нет'}")

# %% [markdown]
# ## Шаг 9: Сохранение результатов

# %%
import pickle

cate_results_full = {
    "cate_mean": cate_scores.mean(),
    "cate_std": cate_scores.std(),
    "septic_shock": {
        "cate": cate_septic_shock,
        "n": septic_shock_mask.sum(),
        "ci": (cate_ss_ci_lower, cate_ss_ci_upper) if cate_ss_mean else None,
    },
    "no_septic_shock": {
        "cate": cate_no_septic_shock,
        "n": no_septic_shock_mask.sum(),
    },
    "age_lt_60": {
        "cate": cate_age_lt_60,
        "n": age_lt_60_mask.sum(),
    },
    "age_ge_60": {
        "cate": cate_age_ge_60,
        "n": age_ge_60_mask.sum(),
    },
}

with open(DATA_DIR / "cate_results.pkl", "wb") as f:
    pickle.dump(cate_results_full, f)

print(f"\nРезультаты CATE сохранены: {DATA_DIR / 'cate_results.pkl'}")

# %% [markdown]
# ## Шаг 10: Сводная таблица всех результатов

# %%
print("\n" + "="*70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*70)

print("\n1. ATE (Average Treatment Effect):")
print(f"   AIPW: {cate_scores.mean():.4f} ({100*cate_scores.mean():.2f}%)")

print("\n2. CATE (Conditional ATE) по подгруппам:")
print(f"   Septic Shock: {cate_septic_shock:.4f} (n={septic_shock_mask.sum()})")
print(f"   No Septic Shock: {cate_no_septic_shock:.4f} (n={no_septic_shock_mask.sum()})")
print(f"   Age < 60: {cate_age_lt_60:.4f} (n={age_lt_60_mask.sum()})")
print(f"   Age >= 60: {cate_age_ge_60:.4f} (n={age_ge_60_mask.sum()})")

print("\n3. Ожидаемые результаты (из статьи):")
print("   Septic Shock: CATE < 0 (benefit от альбумина)")
print("   Другие подгруппы: CATE ≈ 0 (нет значимого эффекта)")

print("\n4. Интерпретация:")
if cate_septic_shock < 0:
    print(f"   ✓ Альбумин показывает benefit для пациентов с септическим шоком")
else:
    print(f"   ✗ Альбумин не показывает benefit для пациентов с септическим шоком")

# %%
