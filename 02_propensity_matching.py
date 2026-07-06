"""
Propensity Score Matching и IPW анализ
Адаптировано из causal_ehr_mimic авторов под MIMIC-IV v3.1
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Пути
DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")

# Загружаем когорту
cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")
print(f"Когорта: {cohort.shape}")
print(f"Лечение: {cohort['treatment'].value_counts().to_dict()}")

# %% [markdown]
# ## Шаг 1: Определяем конфаундеры и признаки

# %%
# Конфаундеры из методологии авторов
confounder_vars = [
    # Демография
    "admission_age", "Female", "White", "Black", "Hispanic",
    "emergency_admission", "insurance_Medicare", "insurance_Medicaid",
    
    # Severity scores
    "lactate_final", "charlson_comorbidity_index",
    
    # Vitals
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    
    # Drugs
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    "has_vasopressors",
    
    # Procedures
    "rrt_flag", "ventilation_flag",
]

# Проверяем что все колонки есть
available_confounders = [col for col in confounder_vars if col in cohort.columns]
print(f"Используемые конфаундеры ({len(available_confounders)}): {available_confounders}")

X = cohort[available_confounders].values
treatment = cohort["treatment"].values
outcome = cohort["mortality_28days"].values

# %% [markdown]
# ## Шаг 2: Propensity score модель с подбором гиперпараметров

# %%
# Разделяем на train/test для валидации модели
X_train, X_test, t_train, t_test = train_test_split(
    X, treatment, test_size=0.3, random_state=42, stratify=treatment
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %%
# Propensity score модель с RandomizedSearchCV
propensity_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

propensity_base = GradientBoostingClassifier(random_state=42)

propensity_search = RandomizedSearchCV(
    estimator=propensity_base,
    param_distributions=propensity_param_dist,
    n_iter=20,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

propensity_search.fit(X_train, t_train)

print(f"Лучшие параметры: {propensity_search.best_params_}")
print(f"Лучший AUC-ROC: {propensity_search.best_score_:.4f}")

# %%
# Предсказываем propensity scores для всей когорты
best_propensity_model = propensity_search.best_estimator_
propensity_scores = best_propensity_model.predict_proba(X)[:, 1]

cohort["propensity_score"] = propensity_scores

# Проверяем качество модели
t_pred = best_propensity_model.predict(X_test)
test_auc = roc_auc_score(t_test, best_propensity_model.predict_proba(X_test)[:, 1])
print(f"Test AUC-ROC: {test_auc:.4f}")

# %% [markdown]
# ## Шаг 3: Проверка positivity assumption

# %%
plt.figure(figsize=(10, 6))
plt.hist(
    propensity_scores[treatment == 1],
    bins=50,
    alpha=0.5,
    label="Treated (Albumin)",
    color="blue",
    density=True,
)
plt.hist(
    propensity_scores[treatment == 0],
    bins=50,
    alpha=0.5,
    label="Control",
    color="orange",
    density=True,
)
plt.xlabel("Propensity Score")
plt.ylabel("Density")
plt.title("Propensity Score Distribution by Treatment Group")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "propensity_distribution.png", dpi=150)
# plt.show()

# %%
# Статистики propensity scores
print("=== Propensity Score Statistics ===")
print(f"Treated - Mean: {propensity_scores[treatment == 1].mean():.3f}, "
      f"Std: {propensity_scores[treatment == 1].std():.3f}")
print(f"Control - Mean: {propensity_scores[treatment == 0].mean():.3f}, "
      f"Std: {propensity_scores[treatment == 0].std():.3f}")

# Проверка overlap
ps_min = propensity_scores[treatment == 1].min()
ps_max = propensity_scores[treatment == 1].max()
print(f"\nTreated PS range: [{ps_min:.3f}, {ps_max:.3f}]")
print(f"Control PS range: [{propensity_scores[treatment == 0].min():.3f}, "
      f"{propensity_scores[treatment == 0].max():.3f}]")

# %% [markdown]
# ## Шаг 4: Trimming (обрезка экстремальных propensity scores)

# %%
# Обрезаем по правилу 1-99 перцентиля
ps_lower = np.percentile(propensity_scores, 1)
ps_upper = np.percentile(propensity_scores, 99)

print(f"Trimming пороги: [{ps_lower:.3f}, {ps_upper:.3f}]")

trimmed_mask = (propensity_scores >= ps_lower) & (propensity_scores <= ps_upper)
trimmed_cohort = cohort[trimmed_mask].copy()

print(f"После trimming: {trimmed_cohort.shape[0]} пациентов (удалено {(~trimmed_mask).sum()})")

# %% [markdown]
# ## Шаг 5: Propensity Score Matching с калипером

# %%
# Выделяем treated и control группы
treated_mask = trimmed_cohort["treatment"] == 1
control_mask = trimmed_cohort["treatment"] == 0

treated_ps = trimmed_cohort.loc[treated_mask, "propensity_score"].values.reshape(-1, 1)
control_ps = trimmed_cohort.loc[control_mask, "propensity_score"].values.reshape(-1, 1)
control_indices = np.where(control_mask)[0]

# Вычисляем калипер (0.2 * std propensity score)
ps_std = trimmed_cohort["propensity_score"].std()
caliper = 0.2 * ps_std
print(f"Калипер: {caliper:.4f} (0.2 * {ps_std:.4f})")

# %%
# Matching с калипером
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(control_ps)

distances, indices = nn.kneighbors(treated_ps)

# Фильтруем matches вне калипера
valid_matches = distances[:, 0] <= caliper
print(f"Valid matches (в пределах калипера): {valid_matches.sum()} из {len(treated_ps)}")

# Создаем matched когорту
matched_treated_indices = np.where(treated_mask)[0][valid_matches]
matched_control_indices = control_indices[indices[valid_matches, 0]]

matched_indices = np.concatenate([matched_treated_indices, matched_control_indices])
matched_cohort = cohort.loc[matched_indices].copy()

print(f"\nMatched когорта: {matched_cohort.shape[0]} пациентов")
print(f"Treated: {(matched_cohort['treatment'] == 1).sum()}")
print(f"Control: {(matched_cohort['treatment'] == 0).sum()}")

# %% [markdown]
# ## Шаг 6: Проверка баланса после matching

# %%
def compute_smd(group1, group2):
    """Вычисляет Standardized Mean Difference"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    smd = (mean1 - mean2) / pooled_std
    
    return smd

# Вычисляем SMD до и после matching
balance_results = []

for var in available_confounders:
    # До matching (на trimmed когорте)
    treated_vals = trimmed_cohort.loc[treated_mask, var].values
    control_vals = trimmed_cohort.loc[control_mask, var].values
    smd_before = compute_smd(treated_vals, control_vals)
    
    # После matching
    matched_treated = matched_cohort[matched_cohort["treatment"] == 1][var].values
    matched_control = matched_cohort[matched_cohort["treatment"] == 0][var].values
    smd_after = compute_smd(matched_treated, matched_control)
    
    balance_results.append({
        "variable": var,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "abs_smd_before": abs(smd_before),
        "abs_smd_after": abs(smd_after),
    })

balance_df = pd.DataFrame(balance_results)
balance_df = balance_df.sort_values("abs_smd_after", ascending=False)

print("=== Баланс ковариат после matching ===")
print(balance_df[["variable", "smd_before", "smd_after", "abs_smd_after"]].head(15))

# Сколько ковариат с |SMD| > 0.1
n_imbalanced = (balance_df["abs_smd_after"] > 0.1).sum()
n_total = len(balance_df)
print(f"\nКовариат с |SMD| > 0.1: {n_imbalanced} из {n_total} ({100*n_imbalanced/n_total:.1f}%)")

# %%
# Визуализация баланса
plt.figure(figsize=(10, 8))

# Love plot
y_pos = np.arange(len(balance_df))
plt.scatter(balance_df["smd_before"], y_pos, alpha=0.6, s=50, label="Before matching", color="red")
plt.scatter(balance_df["smd_after"], y_pos, alpha=0.6, s=50, label="After matching", color="blue")

plt.axvline(x=-0.1, color="green", linestyle="--", alpha=0.7, label="|SMD| = 0.1")
plt.axvline(x=0.1, color="green", linestyle="--", alpha=0.7)
plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

plt.yticks(y_pos, balance_df["variable"], fontsize=9)
plt.xlabel("Standardized Mean Difference")
plt.title("Covariate Balance: Before vs After Matching")
plt.legend()
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(DATA_DIR / "love_plot.png", dpi=150, bbox_inches="tight")
# plt.show()

# %% [markdown]
# ## Шаг 7: IPW (Inverse Propensity Weighting)

# %%
# Вычисляем IPW веса
# Stabilized weights для лучшей стабильности
ps = trimmed_cohort["propensity_score"].values
t = trimmed_cohort["treatment"].values

# Stabilized weights
sw = np.where(t == 1, 
              ps.mean() / ps,           # Для treated
              (1 - ps.mean()) / (1 - ps))  # Для control

trimmed_cohort["ipw_weight"] = sw

print(f"IPW веса - Mean: {sw.mean():.3f}, Std: {sw.std():.3f}")
print(f"IPW веса - Range: [{sw.min():.3f}, {sw.max():.3f}]")

# %% [markdown]
# ## Шаг 8: Оценка ATE через IPW

# %%
from statsmodels.stats.weightstats import DescrStatsW

# Взвешенные средние для outcome
treated_mask_trimmed = trimmed_cohort["treatment"] == 1
control_mask_trimmed = trimmed_cohort["treatment"] == 0

# Средний исход в treated группе (взвешенный)
weighted_treated = DescrStatsW(
    trimmed_cohort.loc[treated_mask_trimmed, "mortality_28days"],
    weights=trimmed_cohort.loc[treated_mask_trimmed, "ipw_weight"],
    ddof=0,
)
outcome_treated = weighted_treated.mean

# Средний исход в control группе (взвешенный)
weighted_control = DescrStatsW(
    trimmed_cohort.loc[control_mask_trimmed, "mortality_28days"],
    weights=trimmed_cohort.loc[control_mask_trimmed, "ipw_weight"],
    ddof=0,
)
outcome_control = weighted_control.mean

# ATE
ate_ipw = outcome_treated - outcome_control
print(f"\n=== IPW ATE ===")
print(f"Outcome (Treated): {outcome_treated:.4f}")
print(f"Outcome (Control): {outcome_control:.4f}")
print(f"ATE (IPW): {ate_ipw:.4f} ({100*ate_ipw:.2f}%)")

# %% [markdown]
# ## Шаг 9: Оценка ATE через Matching

# %%
# Средний исход в matched группах
outcome_treated_matched = matched_cohort[matched_cohort["treatment"] == 1]["mortality_28days"].mean()
outcome_control_matched = matched_cohort[matched_cohort["treatment"] == 0]["mortality_28days"].mean()

ate_matching = outcome_treated_matched - outcome_control_matched

print(f"\n=== Matching ATE ===")
print(f"Outcome (Treated): {outcome_treated_matched:.4f}")
print(f"Outcome (Control): {outcome_control_matched:.4f}")
print(f"ATE (Matching): {ate_matching:.4f} ({100*ate_matching:.2f}%)")

# %% [markdown]
# ## Шаг 10: Bootstrap доверительные интервалы

# %%
def bootstrap_ate(data, treatment_col, outcome_col, weights_col=None, n_bootstrap=1000, random_state=42):
    """Bootstrap для оценки ATE и CI"""
    np.random.seed(random_state)
    n = len(data)
    ate_samples = []
    
    for i in range(n_bootstrap):
        # Ресемплинг с заменой
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[sample_idx]
        
        if weights_col is not None:
            # IPW
            treated = sample[sample[treatment_col] == 1]
            control = sample[sample[treatment_col] == 0]
            
            weighted_treated = DescrStatsW(treated[outcome_col], weights=treated[weights_col], ddof=0)
            weighted_control = DescrStatsW(control[outcome_col], weights=control[weights_col], ddof=0)
            
            ate = weighted_treated.mean - weighted_control.mean
        else:
            # Matching / простой
            ate = sample[sample[treatment_col] == 1][outcome_col].mean() - \
                  sample[sample[treatment_col] == 0][outcome_col].mean()
        
        ate_samples.append(ate)
    
    ate_samples = np.array(ate_samples)
    ate_mean = ate_samples.mean()
    ci_lower = np.percentile(ate_samples, 2.5)
    ci_upper = np.percentile(ate_samples, 97.5)
    
    return ate_mean, ci_lower, ci_upper, ate_samples

# %%
# Bootstrap для IPW
print("Вычисляем bootstrap CI для IPW...")
ate_ipw_boot, ci_ipw_lower, ci_ipw_upper, _ = bootstrap_ate(
    trimmed_cohort, "treatment", "mortality_28days", "ipw_weight", n_bootstrap=500
)

print(f"\nIPW ATE: {ate_ipw_boot:.4f}")
print(f"95% CI: [{ci_ipw_lower:.4f}, {ci_ipw_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_ipw_lower > 0 or ci_ipw_upper < 0) else 'Нет'}")

# %%
# Bootstrap для Matching
print("\nВычисляем bootstrap CI для Matching...")
ate_match_boot, ci_match_lower, ci_match_upper, _ = bootstrap_ate(
    matched_cohort, "treatment", "mortality_28days", None, n_bootstrap=500
)

print(f"\nMatching ATE: {ate_match_boot:.4f}")
print(f"95% CI: [{ci_match_lower:.4f}, {ci_match_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_match_lower > 0 or ci_match_upper < 0) else 'Нет'}")

# %% [markdown]
# ## Шаг 11: Сохранение результатов

# %%
import pickle

results = {
    "ate_ipw": ate_ipw_boot,
    "ci_ipw": (ci_ipw_lower, ci_ipw_upper),
    "ate_matching": ate_match_boot,
    "ci_matching": (ci_match_lower, ci_match_upper),
    "propensity_auc": test_auc,
    "n_covariates": len(available_confounders),
    "n_imbalanced": n_imbalanced,
}

with open(DATA_DIR / "ps_matching_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\nРезультаты сохранены: {DATA_DIR / 'ps_matching_results.pkl'}")

# Визуализация распределения bootstrap
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(_, bins=50, alpha=0.7, color="blue", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
plt.axvline(x=ci_ipw_lower, color="green", linestyle="--", label="95% CI")
plt.axvline(x=ci_ipw_upper, color="green", linestyle="--")
plt.xlabel("ATE (Bootstrap samples)")
plt.ylabel("Frequency")
plt.title(f"IPW ATE Distribution\nATE={ate_ipw_boot:.4f}, 95% CI=[{ci_ipw_lower:.4f}, {ci_ipw_upper:.4f}]")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / "bootstrap_ipw.png", dpi=150)
# plt.show()

# %%
