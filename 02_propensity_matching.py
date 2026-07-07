
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import pickle

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM не установлен. Пропускаем LightGBM модель.")


DATA_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")


cohort = pd.read_csv(DATA_DIR / "cohort_sepsis.csv")
print(f"Когорта: {cohort.shape}")
print(f"Лечение: {cohort['treatment'].value_counts().to_dict()}")




confounder_vars = [
    # Демография
    "admission_age", "Female", "White", "Black", "Hispanic",
    "emergency_admission", "insurance_Medicare", "insurance_Medicaid",
    
    # Severity scores
    "lactate_final", "charlson_comorbidity_index",
    
    # Vitals
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    
    # Drugs (убрали has_vasopressors!)
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    
    # Procedures
    "rrt_flag", "ventilation_flag",
    
    # Missing indicators
    "lactate_missing",
]

available_confounders = [col for col in confounder_vars if col in cohort.columns]
print(f"\nИспользуемые конфаундеры ({len(available_confounders)}): {available_confounders}")


if "lactate_missing" not in cohort.columns:
    print("\nWARNING: lactate_missing не найден в когорте!")
    print("Запустите обновленный 01_cohort_creation.py для создания когорты с missing indicators")
    # Временно создаем dummy колонку
    cohort["lactate_missing"] = 0

X = cohort[available_confounders].values
treatment = cohort["treatment"].values
outcome = cohort["mortality_28days"].values


# Шаг 2: Propensity score - ТРИ модели: LogReg + GBM + LightGBM

print("\n" + "="*70)
print("ШАГ 2: Propensity score модели")
print("="*70)


X_train, X_test, t_train, t_test = train_test_split(
    X, treatment, test_size=0.3, random_state=42, stratify=treatment
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models_to_compare = {}


print("\n--- 1. Logistic Regression (baseline) ---")
logreg_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
logreg_model.fit(X_train, t_train)
propensity_scores_logreg = logreg_model.predict_proba(X)[:, 1]
logreg_auc = roc_auc_score(t_test, logreg_model.predict_proba(X_test)[:, 1])
print(f"LogReg AUC-ROC: {logreg_auc:.4f}")
models_to_compare["LogReg"] = {
    "model": logreg_model,
    "scores": propensity_scores_logreg,
    "auc": logreg_auc
}


print("\n--- 2. GradientBoostingClassifier ---")
propensity_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
}

propensity_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=propensity_param_dist,
    n_iter=15,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

propensity_search.fit(X_train, t_train)
gb_model = propensity_search.best_estimator_

propensity_scores_gb = gb_model.predict_proba(X)[:, 1]
gb_auc = roc_auc_score(t_test, gb_model.predict_proba(X_test)[:, 1])
print(f"GradientBoosting AUC-ROC: {gb_auc:.4f}")
print(f"Лучшие параметры: {propensity_search.best_params_}")
models_to_compare["GradientBoosting"] = {
    "model": gb_model,
    "scores": propensity_scores_gb,
    "auc": gb_auc
}


if LIGHTGBM_AVAILABLE:
    print("\n--- 3. LightGBM ---")
    lgb_param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
    }
    
    lgb_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42, verbose=-1),
        param_distributions=lgb_param_dist,
        n_iter=15,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    
    lgb_search.fit(X_train, t_train)
    lgb_model = lgb_search.best_estimator_
    
    propensity_scores_lgb = lgb_model.predict_proba(X)[:, 1]
    lgb_auc = roc_auc_score(t_test, lgb_model.predict_proba(X_test)[:, 1])
    print(f"LightGBM AUC-ROC: {lgb_auc:.4f}")
    print(f"Лучшие параметры: {lgb_search.best_params_}")
    models_to_compare["LightGBM"] = {
        "model": lgb_model,
        "scores": propensity_scores_lgb,
        "auc": lgb_auc
    }


# Сравнение моделей и выбор

print("\n" + "="*70)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*70)

for name, info in models_to_compare.items():
    print(f"{name:.<20} AUC-ROC: {info['auc']:.4f}")





if logreg_auc >= 0.70:
    print(f"\n=== ИСПОЛЬЗУЕМ LogReg (AUC={logreg_auc:.4f}, достаточно хорошо) ===")
    propensity_model = logreg_model
    propensity_scores = propensity_scores_logreg
elif LIGHTGBM_AVAILABLE and lgb_auc > gb_auc:
    print(f"\n=== ИСПОЛЬЗУЕМ LightGBM (AUC={lgb_auc:.4f}) ===")
    propensity_model = lgb_model
    propensity_scores = propensity_scores_lgb
else:
    print(f"\n=== ИСПОЛЬЗУЕМ GradientBoosting (AUC={gb_auc:.4f}) ===")
    propensity_model = gb_model
    propensity_scores = propensity_scores_gb

cohort["propensity_score"] = propensity_scores


# Шаг 3: Overlap / Positivity check

print("\n" + "="*70)
print("ШАГ 3: Проверка overlap / positivity")
print("="*70)


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
print("Сохранено: propensity_distribution.png")


print("\n=== Propensity Score Statistics ===")
print(f"Treated - Mean: {propensity_scores[treatment == 1].mean():.3f}, "
      f"Std: {propensity_scores[treatment == 1].std():.3f}, "
      f"Range: [{propensity_scores[treatment == 1].min():.3f}, {propensity_scores[treatment == 1].max():.3f}]")
print(f"Control - Mean: {propensity_scores[treatment == 0].mean():.3f}, "
      f"Std: {propensity_scores[treatment == 0].std():.3f}, "
      f"Range: [{propensity_scores[treatment == 0].min():.3f}, {propensity_scores[treatment == 0].max():.3f}]")

# Проверка positivity
n_outside = ((propensity_scores < 0.1) | (propensity_scores > 0.9)).sum()
n_inside = ((propensity_scores >= 0.1) & (propensity_scores <= 0.9)).sum()
pct_outside = 100 * n_outside / len(propensity_scores)

print(f"\n=== Positivity Check [0.1, 0.9] ===")
print(f"Пациентов вне [0.1, 0.9]: {n_outside} ({pct_outside:.1f}%)")
print(f"Пациентов в [0.1, 0.9]: {n_inside} ({100-pct_outside:.1f}%)")

# Trim по [0.1, 0.9]
trim_mask = (propensity_scores >= 0.1) & (propensity_scores <= 0.9)
trimmed_cohort = cohort[trim_mask].copy()
print(f"\nПосле trimming по [0.1, 0.9]: {trimmed_cohort.shape[0]} пациентов (удалено {(~trim_mask).sum()})")


# Шаг 4: Calibration plot (дополнительно)

print("\n" + "="*70)
print("ШАГ 4: Calibration plot")
print("="*70)

prob_true, prob_pred = calibration_curve(treatment, propensity_scores, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot(prob_pred, prob_true, "s-", label="LogReg calibration", markersize=10)
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Propensity Score Calibration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "calibration_plot.png", dpi=150)
print("Сохранено: calibration_plot.png")


# Шаг 5: Propensity Score Matching

print("\n" + "="*70)
print("ШАГ 5: Matching 1:1 nearest neighbor")
print("="*70)

treated_mask_trimmed = trimmed_cohort["treatment"] == 1
control_mask_trimmed = trimmed_cohort["treatment"] == 0

treated_ps = trimmed_cohort.loc[treated_mask_trimmed, "propensity_score"].values.reshape(-1, 1)
control_ps = trimmed_cohort.loc[control_mask_trimmed, "propensity_score"].values.reshape(-1, 1)
control_indices = np.where(control_mask_trimmed)[0]


ps_std = trimmed_cohort["propensity_score"].std()
caliper = 0.2 * ps_std
print(f"Калипер: {caliper:.4f} (0.2 * {ps_std:.4f})")


nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(control_ps)
distances, indices = nn.kneighbors(treated_ps)


valid_matches = distances[:, 0] <= caliper
print(f"Valid matches (в пределах калипера): {valid_matches.sum()} из {len(treated_ps)}")

matched_treated_indices = np.where(treated_mask_trimmed)[0][valid_matches]
matched_control_indices = control_indices[indices[valid_matches, 0]]

matched_indices = np.concatenate([matched_treated_indices, matched_control_indices])
matched_cohort = cohort.loc[matched_indices].copy()

print(f"\n=== Matching Statistics ===")
print(f"Matched когорта: {matched_cohort.shape[0]} пациентов")
print(f"Treated: {(matched_cohort['treatment'] == 1).sum()}")
print(f"Control: {(matched_cohort['treatment'] == 0).sum()}")
print(f"Matched pairs: {valid_matches.sum()}")
print(f"Уникальных control: {len(np.unique(matched_control_indices))}")
print(f"Control с повторами: {len(matched_control_indices) - len(np.unique(matched_control_indices))}")
print(f"Treated выкинуто (вне калипера): {(~valid_matches).sum()}")
print(f"Matching: WITHOUT replacement (по умолчанию)")


# Шаг 6: Баланс ковариат (SMD до/после matching)

print("\n" + "="*70)
print("ШАГ 6: Баланс ковариат (SMD)")
print("="*70)

def compute_smd(group1, group2):
    """Standardized Mean Difference"""
    # Если переданы уже скалярные mean (для взвешенных данных)
    if np.isscalar(group1) or np.isscalar(group2):
        return 0  # Для взвешенных SMD используем упрощенную версию
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (mean1 - mean2) / pooled_std

def compute_smd_weighted(mean1, mean2, var1, var2, n1, n2):
    """SMD для взвешенных данных (использует средние и дисперсии)"""
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (mean1 - mean2) / pooled_std

balance_results = []

for var in available_confounders:
    
    treated_vals = trimmed_cohort.loc[treated_mask_trimmed, var].values
    control_vals = trimmed_cohort.loc[control_mask_trimmed, var].values
    smd_before = compute_smd(treated_vals, control_vals)
    
    
    matched_treated = matched_cohort[matched_cohort["treatment"] == 1][var].values
    matched_control = matched_cohort[matched_cohort["treatment"] == 0][var].values
    smd_after = compute_smd(matched_treated, matched_control)
    
    balance_results.append({
        "variable": var,
        "smd_before": smd_before,
        "smd_after": smd_after,
    })

balance_df = pd.DataFrame(balance_results)
balance_df["abs_smd_before"] = balance_df["smd_before"].abs()
balance_df["abs_smd_after"] = balance_df["smd_after"].abs()
balance_df = balance_df.sort_values("abs_smd_after", ascending=False)

print("\nТоп-10 ковариат по |SMD| после matching:")
print(balance_df[["variable", "smd_before", "smd_after", "abs_smd_after"]].head(10))

n_imbalanced = (balance_df["abs_smd_after"] > 0.1).sum()
n_total = len(balance_df)
print(f"\nКовариат с |SMD| > 0.1: {n_imbalanced} из {n_total} ({100*n_imbalanced/n_total:.1f}%)")
print(f"Ковариат с |SMD| < 0.1: {n_total - n_imbalanced} из {n_total} ({100*(n_total-n_imbalanced)/n_total:.1f}%)")

# Love plot
plt.figure(figsize=(10, 8))
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
print("Сохранено: love_plot.png")


# Шаг 7: IPW с extreme weights clipping

print("\n" + "="*70)
print("ШАГ 7: IPW с extreme weights clipping")
print("="*70)

ps = trimmed_cohort["propensity_score"].values
t = trimmed_cohort["treatment"].values


sw = np.where(
    t == 1, 
    ps.mean() / ps,
    (1 - ps.mean()) / (1 - ps)
)


sw_clipped = np.clip(sw, 0.1, 10.0)
n_extreme = (sw > 10).sum() + (sw < 0.1).sum()

print(f"IPW веса (до clipping) - Mean: {sw.mean():.3f}, Std: {sw.std():.3f}")
print(f"IPW веса (до clipping) - Range: [{sw.min():.3f}, {sw.max():.3f}]")
print(f"Extreme weights (>10 или <0.1): {n_extreme} ({100*n_extreme/len(sw):.1f}%)")

trimmed_cohort["ipw_weight"] = sw_clipped
print(f"IPW веса (после clipping) - Range: [{sw_clipped.min():.3f}, {sw_clipped.max():.3f}]")

# Effective sample size
ess = (sw_clipped.sum())**2 / (sw_clipped**2).sum()
print(f"\nEffective Sample Size (ESS): {ess:.0f} (из {len(trimmed_cohort)})")
print(f"ESS ratio: {100*ess/len(trimmed_cohort):.1f}%")


# Шаг 8: SMD до/после IPW

print("\n" + "="*70)
print("ШАГ 8: SMD до/после IPW")
print("="*70)

from statsmodels.stats.weightstats import DescrStatsW

balance_ipw_results = []

for var in available_confounders:
    # До IPW
    treated_vals = trimmed_cohort.loc[treated_mask_trimmed, var].values
    control_vals = trimmed_cohort.loc[control_mask_trimmed, var].values
    smd_before = compute_smd(treated_vals, control_vals)
    
    # После IPW (взвешенные)
    treated_weights = trimmed_cohort.loc[treated_mask_trimmed, "ipw_weight"].values
    control_weights = trimmed_cohort.loc[control_mask_trimmed, "ipw_weight"].values
    
    treated_weighted = DescrStatsW(treated_vals, weights=treated_weights, ddof=0)
    control_weighted = DescrStatsW(control_vals, weights=control_weights, ddof=0)
    
    # Используем упрощенный SMD для взвешенных данных
    n_treated = treated_mask_trimmed.sum()
    n_control = control_mask_trimmed.sum()
    smd_after = compute_smd_weighted(
        treated_weighted.mean, control_weighted.mean,
        treated_weighted.var, control_weighted.var,
        n_treated, n_control
    )
    
    balance_ipw_results.append({
        "variable": var,
        "smd_before": smd_before,
        "smd_after_ipw": smd_after,
    })

balance_ipw_df = pd.DataFrame(balance_ipw_results)
balance_ipw_df["abs_smd_before"] = balance_ipw_df["smd_before"].abs()
balance_ipw_df["abs_smd_after"] = balance_ipw_df["smd_after_ipw"].abs()

n_imbalanced_ipw = (balance_ipw_df["abs_smd_after"] > 0.1).sum()
print(f"\nКовариат с |SMD| > 0.1 после IPW: {n_imbalanced_ipw} из {n_total} ({100*n_imbalanced_ipw/n_total:.1f}%)")

print("\nТоп-5 ковариат по |SMD| после IPW:")
print(balance_ipw_df.sort_values("abs_smd_after")[["variable", "smd_before", "smd_after_ipw"]].head(5))


# Шаг 9: ATE оценки (IPW и Matching)

print("\n" + "="*70)
print("ШАГ 9: ATE оценки")
print("="*70)

# IPW ATE
weighted_treated = DescrStatsW(
    trimmed_cohort.loc[treated_mask_trimmed, "mortality_28days"],
    weights=trimmed_cohort.loc[treated_mask_trimmed, "ipw_weight"],
    ddof=0,
)
outcome_treated = weighted_treated.mean

weighted_control = DescrStatsW(
    trimmed_cohort.loc[control_mask_trimmed, "mortality_28days"],
    weights=trimmed_cohort.loc[control_mask_trimmed, "ipw_weight"],
    ddof=0,
)
outcome_control = weighted_control.mean

ate_ipw = outcome_treated - outcome_control
print(f"\n=== IPW ATE ===")
print(f"Outcome (Treated): {outcome_treated:.4f}")
print(f"Outcome (Control): {outcome_control:.4f}")
print(f"ATE (IPW): {ate_ipw:.4f} ({100*ate_ipw:.2f}%)")

# Matching ATE
outcome_treated_matched = matched_cohort[matched_cohort["treatment"] == 1]["mortality_28days"].mean()
outcome_control_matched = matched_cohort[matched_cohort["treatment"] == 0]["mortality_28days"].mean()
ate_matching = outcome_treated_matched - outcome_control_matched

print(f"\n=== Matching ATE ===")
print(f"Outcome (Treated): {outcome_treated_matched:.4f}")
print(f"Outcome (Control): {outcome_control_matched:.4f}")
print(f"ATE (Matching): {ate_matching:.4f} ({100*ate_matching:.2f}%)")


# Шаг 10: Bootstrap CI

print("\n" + "="*70)
print("ШАГ 10: Bootstrap 95% CI")
print("="*70)

def bootstrap_ate(data, treatment_col, outcome_col, weights_col=None, n_bootstrap=500, random_state=42):
    """Bootstrap для ATE и CI"""
    np.random.seed(random_state)
    n = len(data)
    ate_samples = []
    
    for i in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[sample_idx]
        
        if weights_col is not None:
            treated = sample[sample[treatment_col] == 1]
            control = sample[sample[treatment_col] == 0]
            
            weighted_treated = DescrStatsW(treated[outcome_col], weights=treated[weights_col], ddof=0)
            weighted_control = DescrStatsW(control[outcome_col], weights=control[weights_col], ddof=0)
            
            ate = weighted_treated.mean - weighted_control.mean
        else:
            ate = sample[sample[treatment_col] == 1][outcome_col].mean() - \
                  sample[sample[treatment_col] == 0][outcome_col].mean()
        
        ate_samples.append(ate)
    
    ate_samples = np.array(ate_samples)
    ci_lower = np.percentile(ate_samples, 2.5)
    ci_upper = np.percentile(ate_samples, 97.5)
    
    return ate_samples.mean(), ci_lower, ci_upper, ate_samples


print("Вычисляем bootstrap CI для IPW (n=500)...")
ate_ipw_boot, ci_ipw_lower, ci_ipw_upper, ipw_samples = bootstrap_ate(
    trimmed_cohort, "treatment", "mortality_28days", "ipw_weight", n_bootstrap=500
)

print(f"\nIPW ATE: {ate_ipw_boot:.4f}")
print(f"95% CI: [{ci_ipw_lower:.4f}, {ci_ipw_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_ipw_lower > 0 or ci_ipw_upper < 0) else 'Нет'}")


print("\nВычисляем bootstrap CI для Matching (n=500)...")
ate_match_boot, ci_match_lower, ci_match_upper, match_samples = bootstrap_ate(
    matched_cohort, "treatment", "mortality_28days", None, n_bootstrap=500
)

print(f"\nMatching ATE: {ate_match_boot:.4f}")
print(f"95% CI: [{ci_match_lower:.4f}, {ci_match_upper:.4f}]")
print(f"Значимо: {'Да' if (ci_match_lower > 0 or ci_match_upper < 0) else 'Нет'}")


plt.figure(figsize=(10, 5))
plt.hist(ipw_samples, bins=50, alpha=0.7, color="blue", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Null effect")
plt.axvline(x=ci_ipw_lower, color="green", linestyle="--", label="95% CI")
plt.axvline(x=ci_ipw_upper, color="green", linestyle="--")
plt.xlabel("ATE (Bootstrap samples)")
plt.ylabel("Frequency")
plt.title(f"IPW ATE Distribution\nATE={ate_ipw_boot:.4f}, 95% CI=[{ci_ipw_lower:.4f}, {ci_ipw_upper:.4f}]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "bootstrap_ipw.png", dpi=150)
print("Сохранено: bootstrap_ipw.png")


# Шаг 11: Сохранение результатов

print("\n" + "="*70)
print("ШАГ 11: Сохранение результатов")
print("="*70)

results = {
    "ate_ipw": ate_ipw_boot,
    "ci_ipw": (ci_ipw_lower, ci_ipw_upper),
    "ate_matching": ate_match_boot,
    "ci_matching": (ci_match_lower, ci_match_upper),
    "propensity_auc": logreg_auc,
    "n_covariates": len(available_confounders),
    "n_imbalanced": n_imbalanced,
    "ess": ess,
    "n_trimmed": (trim_mask).sum(),
    "n_matched": matched_cohort.shape[0],
}

with open(DATA_DIR / "ps_matching_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"Результаты сохранены: {DATA_DIR / 'ps_matching_results.pkl'}")


# ИТОГОВАЯ ТАБЛИЦА

print("\n" + "="*70)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*70)
print(f"{'Метод':<20} {'ATE':>10} {'95% CI':>25} {'Значимо?':>10}")
print("-" * 70)
print(f"{'Matching':<20} {ate_match_boot:>10.4f} [{ci_match_lower:>8.4f}, {ci_match_upper:>8.4f}] {'Да' if (ci_match_lower > 0 or ci_match_upper < 0) else 'Нет':>10}")
print(f"{'IPW':<20} {ate_ipw_boot:>10.4f} [{ci_ipw_lower:>8.4f}, {ci_ipw_upper:>8.4f}] {'Да' if (ci_ipw_lower > 0 or ci_ipw_upper < 0) else 'Нет':>10}")
print("="*70)
