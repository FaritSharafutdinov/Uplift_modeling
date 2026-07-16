
import polars as pl
import pandas as pd
import numpy as np
import pickle
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
    "lactate_final", "charlson_comorbidity_index",
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    "rrt_flag", "ventilation_flag",
    "lactate_missing",
]

available_confounders = [col for col in confounder_vars if col in cohort.columns]
print(f"Конфаундеры: {len(available_confounders)}")

X = cohort[available_confounders].values
treatment = cohort["treatment"].values
outcome = cohort["mortality_28days"].values


# ## Шаг 2: Propensity model с 5-fold cross-fitting
# N nuisance models (propensity AND outcome) должны train на train-fold only и predict held-out fold

from sklearn.model_selection import KFold

print("\n" + "="*70)
print("ШАГ 2: Propensity score с 5-fold cross-fitting")
print("="*70)

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize array for OOF propensity scores
e_x_oof = np.zeros(len(X))

propensity_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
}

print(f"\n5-fold cross-fitting для propensity model...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    t_train, t_val = treatment[train_idx], treatment[val_idx]
    
    # Train propensity model on fold
    propensity_fold = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions=propensity_param_dist,
        n_iter=15,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )
    propensity_fold.fit(X_train, t_train)
    
    # Predict on held-out fold
    e_x_oof[val_idx] = propensity_fold.predict_proba(X_val)[:, 1]
    
    print(f"  Fold {fold + 1}/{n_folds} completed")

# Handle any zeros (folds with all treated or all control)
e_x_oof = np.clip(e_x_oof, 0.01, 0.99)

# Trimming unified [0.1, 0.9]
ps_lower, ps_upper = 0.1, 0.9
trim_mask = (e_x_oof >= ps_lower) & (e_x_oof <= ps_upper)

X_trimmed = X[trim_mask]
t_trimmed = treatment[trim_mask]
y_trimmed = outcome[trim_mask]
ps_trimmed = e_x_oof[trim_mask]  # Use OOF predictions

print(f"\nПосле trimming [0.1, 0.9]: {X_trimmed.shape[0]} пациентов")

# Save OOF predictions for audit
np.save(DATA_DIR / "oof_propensity_scores.npy", e_x_oof)
print(f"OOF propensity scores сохранены: {DATA_DIR / 'oof_propensity_scores.npy'}")


# ## Шаг 3: Outcome models с 5-fold cross-fitting
# Cross-fitting предотвращает overfitting predictions

from sklearn.model_selection import KFold

outcome_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

mu_1_cv = np.zeros(len(X_trimmed))
mu_0_cv = np.zeros(len(X_trimmed))

treated_mask_full = t_trimmed == 1
control_mask_full = t_trimmed == 0

print(f"\n5-fold cross-fitting для outcome models...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_trimmed)):
    X_train, X_val = X_trimmed[train_idx], X_trimmed[val_idx]
    t_train, t_val = t_trimmed[train_idx], t_trimmed[val_idx]
    y_train, y_val = y_trimmed[train_idx], y_trimmed[val_idx]
    
    treated_train = t_train == 1
    control_train = t_train == 0
    
    if treated_train.sum() > 10 and control_train.sum() > 10:
        outcome_t_fold = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_distributions=outcome_param_dist,
            n_iter=20,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
        )
        outcome_c_fold = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_distributions=outcome_param_dist,
            n_iter=20,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
        )
        
        outcome_t_fold.fit(X_train[treated_train], y_train[treated_train])
        outcome_c_fold.fit(X_train[control_train], y_train[control_train])
        
        mu_1_cv[val_idx] = outcome_t_fold.predict(X_val)
        mu_0_cv[val_idx] = outcome_c_fold.predict(X_val)
    
    print(f"Fold {fold + 1}/{n_folds} completed")

print(f"Cross-fitting predictions: mu_1 mean={mu_1_cv.mean():.4f}, mu_0 mean={mu_0_cv.mean():.4f}")


# ## Шаг 4: AIPW оценка с cross-fitted predictions


def compute_aipw_cf(t, y, e_x, mu_1_x, mu_0_x):
    """
    AIPW estimator с готовыми cross-fitted predictions.
    Doubly robust: consistent if either propensity OR outcome model is correctly specified.
    
    Формула (causal_ehr_mimic/caumim/inference/utils.py:191-208):
    aipw_individual = mu_1_x - mu_0_x + t * (y - mu_1_x) / e_x - (1 - t) * (y - mu_0_x) / (1 - e_x)
    """
    e_x = np.clip(e_x, 0.01, 0.99)
    
    aipw_individual = (
        mu_1_x - mu_0_x
        + t * (y - mu_1_x) / e_x
        - (1 - t) * (y - mu_0_x) / (1 - e_x)
    )
    
    ate = aipw_individual.mean()
    return ate, aipw_individual


e_x_trimmed = ps_trimmed
ate_aipw, aipw_individual = compute_aipw_cf(t_trimmed, y_trimmed, e_x_trimmed, mu_1_cv, mu_0_cv)

print(f"\n=== AIPW ATE (cross-fitted) ===")
print(f"ATE: {ate_aipw:.4f} ({100*ate_aipw:.2f}%)")


n = len(aipw_individual)
se_aipw = aipw_individual.std(ddof=1) / np.sqrt(n)
ci_lower = ate_aipw - 1.96 * se_aipw
ci_upper = ate_aipw + 1.96 * se_aipw

print(f"SE: {se_aipw:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_lower > 0 or ci_upper < 0) else 'Нет'}")


# ## Шаг 5: Bootstrap с переобучением моделей
# Bootstrap должен переобучать модели на каждой итерации (не использовать готовые predictions)


def bootstrap_aipw_full_pipeline(X, t, y, outcome_param_dist, n_bootstrap=500, random_state=42):
    """
    Full pipeline bootstrap for AIPW:
    1. Resample patients
    2. Retrain propensity model (with cross-fitting)
    3. Re-trim
    4. Retrain outcome models (with cross-fitting)
    5. Estimate ATE
    """
    from sklearn.model_selection import KFold
    
    np.random.seed(random_state)
    n = len(t)
    ate_samples = []
    n_folds = 5
    
    for i in range(n_bootstrap):
        # Step 1: Resample
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        t_boot = t[idx]
        y_boot = y[idx]
        
        # Step 2: Retrain propensity model with cross-fitting
        kf_boot = KFold(n_splits=n_folds, shuffle=True, random_state=i)
        e_x_oof_boot = np.zeros(len(X_boot))
        
        propensity_param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
        }
        
        for train_idx, val_idx in kf_boot.split(X_boot):
            X_train, X_val = X_boot[train_idx], X_boot[val_idx]
            t_train, t_val = t_boot[train_idx], t_boot[val_idx]
            
            propensity_fold = RandomizedSearchCV(
                estimator=GradientBoostingClassifier(random_state=42),
                param_distributions=propensity_param_dist,
                n_iter=10,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=42,
            )
            propensity_fold.fit(X_train, t_train)
            e_x_oof_boot[val_idx] = propensity_fold.predict_proba(X_val)[:, 1]
        
        e_x_oof_boot = np.clip(e_x_oof_boot, 0.01, 0.99)
        
        # Step 3: Re-trim
        trim_mask_boot = (e_x_oof_boot >= 0.1) & (e_x_oof_boot <= 0.9)
        if trim_mask_boot.sum() < 100:
            continue
        
        X_trimmed_boot = X_boot[trim_mask_boot]
        t_trimmed_boot = t_boot[trim_mask_boot]
        y_trimmed_boot = y_boot[trim_mask_boot]
        ps_trimmed_boot = e_x_oof_boot[trim_mask_boot]
        
        # Step 4: Retrain outcome models with cross-fitting
        kf_outcome = KFold(n_splits=3, shuffle=True, random_state=i)
        mu_1_boot = np.zeros(len(X_trimmed_boot))
        mu_0_boot = np.zeros(len(X_trimmed_boot))
        
        for train_idx, val_idx in kf_outcome.split(X_trimmed_boot):
            X_train, X_val = X_trimmed_boot[train_idx], X_trimmed_boot[val_idx]
            t_train, t_val = t_trimmed_boot[train_idx], t_trimmed_boot[val_idx]
            y_train, y_val = y_trimmed_boot[train_idx], y_trimmed_boot[val_idx]
            
            treated_train = t_train == 1
            control_train = t_train == 0
            
            if treated_train.sum() > 10 and control_train.sum() > 10:
                outcome_t_fold = RandomizedSearchCV(
                    estimator=GradientBoostingRegressor(random_state=42),
                    param_distributions=outcome_param_dist,
                    n_iter=10,
                    cv=3,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    random_state=42,
                )
                outcome_c_fold = RandomizedSearchCV(
                    estimator=GradientBoostingRegressor(random_state=42),
                    param_distributions=outcome_param_dist,
                    n_iter=10,
                    cv=3,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    random_state=42,
                )
                
                outcome_t_fold.fit(X_train[treated_train], y_train[treated_train])
                outcome_c_fold.fit(X_train[control_train], y_train[control_train])
                
                mu_1_boot[val_idx] = outcome_t_fold.predict(X_val)
                mu_0_boot[val_idx] = outcome_c_fold.predict(X_val)
        
        # Step 5: Estimate ATE
        ate, _ = compute_aipw_cf(t_trimmed_boot, y_trimmed_boot, ps_trimmed_boot, mu_1_boot, mu_0_boot)
        ate_samples.append(ate)
        
        if (i + 1) % 100 == 0:
            print(f"  AIPW Bootstrap {i + 1}/{n_bootstrap}")
    
    ate_samples = np.array(ate_samples)
    ci_lower = np.percentile(ate_samples, 2.5)
    ci_upper = np.percentile(ate_samples, 97.5)
    
    return ate_samples.mean(), ci_lower, ci_upper, ate_samples


print("Вычисляем bootstrap CI для AIPW (n=100, с полным переобучением propensity + outcome)...")
ate_aipw_boot, ci_aipw_lower, ci_aipw_upper, aipw_samples = bootstrap_aipw_full_pipeline(
    X, treatment, outcome,
    outcome_param_dist,
    n_bootstrap=100  # Reduced for faster runtime
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

# Save OOF predictions for audit
np.save(DATA_DIR / "oof_mu1_predictions.npy", mu_1_cv)
np.save(DATA_DIR / "oof_mu0_predictions.npy", mu_0_cv)
print(f"\nOOF predictions сохранены: {DATA_DIR / 'oof_mu*_predictions.npy'}")

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
if matching_results['ate_matching'] is not None:
    print(f"{'Matching':<20} {matching_results['ate_matching']:>10.4f} [{matching_results['ci_matching'][0]:>8.4f}, {matching_results['ci_matching'][1]:>8.4f}]")
if matching_results['ate_ipw'] is not None:
    print(f"{'IPW':<20} {matching_results['ate_ipw']:>10.4f} [{matching_results['ci_ipw'][0]:>8.4f}, {matching_results['ci_ipw'][1]:>8.4f}]")
print(f"{'AIPW':<20} {ate_aipw_boot:>10.4f} [{ci_aipw_lower:>8.4f}, {ci_aipw_upper:>8.4f}]")


# ## Шаг 6: Сохранение результатов


results = {
    "ate_aipw": ate_aipw_boot,
    "ci_aipw": (ci_aipw_lower, ci_aipw_upper),
    "ate_ipw": matching_results.get("ate_ipw"),
    "ci_ipw": matching_results.get("ci_ipw"),
    "ate_matching": matching_results.get("ate_matching"),
    "ci_matching": matching_results.get("ci_matching"),
    "propensity_auc": propensity_info["auc"],
    "n_covariates": len(available_confounders),
    "n_trimmed": len(X_trimmed),
}

with open(DATA_DIR / "aipw_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"\nРезультаты сохранены: {DATA_DIR / 'aipw_results.pkl'}")


# ## Шаг 7: Проверка робастности (Doubly Robust свойство)

"""
AIPW является doubly robust: он даст состоятельную оценку если
1) Propensity model ИЛИ 2) Outcome model правильно специфицированы
"""

print(f"\n=== AIPW ATE (final): {ate_aipw_boot:.4f} ({100*ate_aipw_boot:.2f}%) ===")
print(f"95% CI: [{ci_aipw_lower:.4f}, {ci_aipw_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_aipw_lower > 0 or ci_aipw_upper < 0) else 'Нет'}")


