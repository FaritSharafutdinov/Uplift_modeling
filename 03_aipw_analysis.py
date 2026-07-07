
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")


cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")
print(f"Когорта: {cohort.shape}")


# ## Шаг 1: Подготовка данных


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
print(f"Конфаундеры: {len(available_confounders)}")

X = cohort[available_confounders].values
treatment = cohort["treatment"].values
outcome = cohort["mortality_28days"].values


# ## Шаг 2: Propensity model с гиперпараметрами



propensity_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

propensity_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=propensity_param_dist,
    n_iter=50,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
)

propensity_search.fit(X, treatment)
propensity_model = propensity_search.best_estimator_

print(f"Propensity AUC-ROC: {roc_auc_score(treatment, propensity_model.predict_proba(X)[:, 1]):.4f}")


propensity_scores = propensity_model.predict_proba(X)[:, 1]


ps_lower, ps_upper = np.percentile(propensity_scores, [1, 99])
trim_mask = (propensity_scores >= ps_lower) & (propensity_scores <= ps_upper)

X_trimmed = X[trim_mask]
t_trimmed = treatment[trim_mask]
y_trimmed = outcome[trim_mask]
ps_trimmed = propensity_scores[trim_mask]

print(f"После trimming: {X_trimmed.shape[0]} пациентов")


# ## Шаг 3: Outcome models с гиперпараметрами



outcome_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10, 15],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


treated_mask = t_trimmed == 1
X_treated = X_trimmed[treated_mask]
y_treated = y_trimmed[treated_mask]

outcome_model_treated_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=outcome_param_dist,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
)

outcome_model_treated_search.fit(X_treated, y_treated)
outcome_model_treated = outcome_model_treated_search.best_estimator_

print(f"Treated outcome model - Best params: {outcome_model_treated_search.best_params_}")


control_mask = t_trimmed == 0
X_control = X_trimmed[control_mask]
y_control = y_trimmed[control_mask]

outcome_model_control_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=outcome_param_dist,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
)

outcome_model_control_search.fit(X_control, y_control)
outcome_model_control = outcome_model_control_search.best_estimator_

print(f"Control outcome model - Best params: {outcome_model_control_search.best_params_}")


# ## Шаг 4: AIPW оценка


def compute_aipw(X, t, y, propensity_model, outcome_model_treated, outcome_model_control):
    """
    Вычисляет AIPW оценку
    
    AIPW = mean[
        (T * Y / e(X)) - ((1-T) * Y / (1-e(X))) +
        (mu_1(X) - mu_0(X))
    ]
    """
    # Propensity scores
    e_x = propensity_model.predict_proba(X)[:, 1]
    e_x = np.clip(e_x, 0.01, 0.99)  # Защита от деления на 0
    
    # Outcome predictions
    mu_1_x = outcome_model_treated.predict(X)
    mu_0_x = outcome_model_control.predict(X)
    
    # AIPW weights
    #第一部分: IPW component
    ipw_treated = (t * y) / e_x
    ipw_control = ((1 - t) * y) / (1 - e_x)
    
    #第二部分: Outcome model augmentation
    augmentation = mu_1_x - mu_0_x
    
    # AIPW estimator для каждого наблюдения
    aipw_individual = ipw_treated - ipw_control + augmentation
    
    # ATE = mean
    ate = aipw_individual.mean()
    
    return ate, aipw_individual


ate_aipw, aipw_individual = compute_aipw(
    X_trimmed, t_trimmed, y_trimmed,
    propensity_model,
    outcome_model_treated,
    outcome_model_control
)

print(f"\n=== AIPW ATE ===")
print(f"ATE: {ate_aipw:.4f} ({100*ate_aipw:.2f}%)")


n = len(aipw_individual)
se_aipw = aipw_individual.std(ddof=1) / np.sqrt(n)
ci_lower = ate_aipw - 1.96 * se_aipw
ci_upper = ate_aipw + 1.96 * se_aipw

print(f"SE: {se_aipw:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_lower > 0 or ci_upper < 0) else 'Нет'}")


# ## Шаг 5: Bootstrap доверительные интервалы


def bootstrap_aipw(X, t, y, propensity_model, outcome_t, outcome_c, n_bootstrap=1000, random_state=42):
    """Bootstrap для AIPW"""
    np.random.seed(random_state)
    n = len(t)
    ate_samples = []
    
    for i in range(n_bootstrap):
        
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        t_boot = t[idx]
        y_boot = y[idx]
        
        ate, _ = compute_aipw(
            X_boot, t_boot, y_boot,
            propensity_model, outcome_t, outcome_c
        )
        ate_samples.append(ate)
    
    ate_samples = np.array(ate_samples)
    ci_lower = np.percentile(ate_samples, 2.5)
    ci_upper = np.percentile(ate_samples, 97.5)
    
    return ate_samples.mean(), ci_lower, ci_upper, ate_samples


print("Вычисляем bootstrap CI для AIPW...")
ate_aipw_boot, ci_aipw_lower, ci_aipw_upper, aipw_samples = bootstrap_aipw(
    X_trimmed, t_trimmed, y_trimmed,
    propensity_model, outcome_model_treated, outcome_model_control,
    n_bootstrap=1000
)

print(f"\nAIPW ATE (bootstrap): {ate_aipw_boot:.4f}")
print(f"95% CI: [{ci_aipw_lower:.4f}, {ci_aipw_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_aipw_lower > 0 or ci_aipw_upper < 0) else 'Нет'}")



plt.figure(figsize=(10, 5))
plt.hist(aipw_samples, bins=50, alpha=0.7, color="purple", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Null effect")
plt.axvline(x=ci_aipw_lower, color="green", linestyle="--", label="95% CI")
plt.axvline(x=ci_aipw_upper, color="green", linestyle="--")
plt.xlabel("ATE (Bootstrap samples)")
plt.ylabel("Frequency")
plt.title(f"AIPW ATE Distribution\nATE={ate_aipw_boot:.4f}, 95% CI=[{ci_aipw_lower:.4f}, {ci_aipw_upper:.4f}]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "bootstrap_aipw.png", dpi=150)
# plt.show()


# ## Шаг 6: Сравнение всех методов


import pickle


try:
    with open(DATA_DIR / "ps_matching_results.pkl", "rb") as f:
        matching_results = pickle.load(f)
except:
    matching_results = {"ate_ipw": None, "ci_ipw": (None, None), "ate_matching": None, "ci_matching": (None, None)}


print("\n=== СРАВНЕНИЕ МЕТОДОВ ===")
print(f"{'Метод':<20} {'ATE':>10} {'95% CI':>25}")
print("-" * 60)
print(f"{'Matching':<20} {matching_results['ate_matching']:>10.4f} [{matching_results['ci_matching'][0]:>8.4f}, {matching_results['ci_matching'][1]:>8.4f}]")
print(f"{'IPW':<20} {matching_results['ate_ipw']:>10.4f} [{matching_results['ci_ipw'][0]:>8.4f}, {matching_results['ci_ipw'][1]:>8.4f}]")
print(f"{'AIPW':<20} {ate_aipw_boot:>10.4f} [{ci_aipw_lower:>8.4f}, {ci_aipw_upper:>8.4f}]")
print(f"{'RCT Gold Standard':<20} {0:>10.4f} [  -  ,   -  ]")


# ## Шаг 7: Сохранение результатов


results = {
    "ate_aipw": ate_aipw_boot,
    "ci_aipw": (ci_aipw_lower, ci_aipw_upper),
    "ate_ipw": matching_results.get("ate_ipw"),
    "ci_ipw": matching_results.get("ci_ipw"),
    "ate_matching": matching_results.get("ate_matching"),
    "ci_matching": matching_results.get("ci_matching"),
    "propensity_auc": roc_auc_score(treatment, propensity_model.predict_proba(X)[:, 1]),
    "n_covariates": len(available_confounders),
}

with open(DATA_DIR / "aipw_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\nРезультаты сохранены: {DATA_DIR / 'aipw_results.pkl'}")


# ## Шаг 8: Проверка робастности (Doubly Robust свойство)


"""
AIPW является doubly robust: он даст состоятельную оценку если
1) Propensity model ИЛИ 2) Outcome model правильно специфицированы

Проверим робастность, сравнив с чистыми оценками:
- Только IPW (без outcome models)
- Только Outcome model (T-learner)
"""

# T-learner (только outcome models)
mu_1 = outcome_model_treated.predict(X_trimmed)
mu_0 = outcome_model_control.predict(X_trimmed)
ate_tlearner = mu_1.mean() - mu_0.mean()

print(f"\nT-Learner ATE: {ate_tlearner:.4f}")


e_x = np.clip(ps_trimmed, 0.01, 0.99)
ipw_ate = (t_trimmed * y_trimmed / e_x).mean() - ((1 - t_trimmed) * y_trimmed / (1 - e_x)).mean()
print(f"Pure IPW ATE: {ipw_ate:.4f}")

print(f"\nAIPW ATE: {ate_aipw_boot:.4f}")
print("\nВсе три оценки должны быть близки если модели хорошо специфицированы")


