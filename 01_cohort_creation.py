"""
Создание когорты для causal inference анализа альбумина при сепсисе
Адаптировано из causal_ehr_mimic авторов (Doutreligne et al.) под MIMIC-IV v3.1

Ключевые изменения по сравнению с v2.2:
- Используем напрямую parquet файлы из v3.1
- Добавляем все конфаундеры из методологии авторов
- Правильная импутация пропусков
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from datetime import timedelta

# Пути к данным
MIMIC_DIR = Path("/Users/faritsharafutdinov/untitled folder/mimic-iv-3.1/mimiciv_as_parquet")
MIMIC_ICU = MIMIC_DIR / "mimiciv_icu"
MIMIC_HOSP = MIMIC_DIR / "mimiciv_hosp"
MIMIC_DERIVED = MIMIC_DIR / "mimiciv_derived"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)

print("="*70)
print("ШАГ 1: Базовая популяция (ICU stays)")
print("="*70)

# Загружаем icustays - это наша начальная популяция
icustays = pl.read_parquet(MIMIC_ICU / "icustays")
print(f"Всего ICU stays: {icustays.shape[0]}")

# Присоединяем демографию из patients
patients = pl.read_parquet(MIMIC_HOSP / "patients")

base_population = icustays.join(
    patients.select(["subject_id", "gender", "anchor_age", "anchor_year", "dod"]),
    on="subject_id",
    how="left"
)

# Для v3.1: возраст = anchor_age + (intime.year - anchor_year)
base_population = base_population.with_columns(
    (
        pl.col("anchor_age") + 
        (pl.col("intime").dt.year() - pl.col("anchor_year"))
    ).alias("admission_age")
)

# Фильтры
base_population = base_population.filter(
    (pl.col("admission_age") >= 18) &
    (pl.col("los") >= 1.0)  # LOS >= 1 дня (в днях, float)
)

# Берем только первый ICU stay для каждого пациента
base_population = base_population.sort(["subject_id", "intime"]).group_by("subject_id").first()

print(f"После фильтрации (возраст >= 18, LOS >= 1 день, первый stay): {base_population.shape[0]}")

# Создаем флаги пола и расы
base_population = base_population.with_columns(
    (pl.col("gender") == "F").cast(pl.Int32).alias("Female"),
    (pl.col("gender") == "M").cast(pl.Int32).alias("Male")
)

print("\n" + "="*70)
print("ШАГ 2: Admissions (emergency, insurance, race)")
print("="*70)

admissions = pl.read_parquet(MIMIC_HOSP / "admissions")

base_population = base_population.join(
    admissions.select([
        "subject_id", "hadm_id", "admission_type", "insurance", "race",
        "admittime", "dischtime", "deathtime"
    ]),
    on=["subject_id", "hadm_id"],
    how="left"
)

# Emergency admission
base_population = base_population.with_columns(
    (pl.col("admission_type") == "EMERGENCY").cast(pl.Int32).alias("emergency_admission")
)

# Insurance (one-hot encoding)
base_population = base_population.with_columns(
    (pl.col("insurance") == "Medicare").cast(pl.Int32).alias("insurance_Medicare"),
    (pl.col("insurance") == "Medicaid").cast(pl.Int32).alias("insurance_Medicaid"),
    (pl.col("insurance") == "Private").cast(pl.Int32).alias("insurance_Private"),
)

# Race (one-hot encoding)
base_population = base_population.with_columns(
    (pl.col("race").str.to_lowercase().str.contains("white")).cast(pl.Int32).alias("White"),
    (pl.col("race").str.to_lowercase().str.contains("black")).cast(pl.Int32).alias("Black"),
    (pl.col("race").str.to_lowercase().str.contains("hispanic")).cast(pl.Int32).alias("Hispanic"),
)

print(f"После присоединения admissions: {base_population.shape[0]}")

print("\n" + "="*70)
print("ШАГ 3: Витальные признаки (за 24ч до ICU)")
print("="*70)

vitalsign = pl.read_parquet(MIMIC_DERIVED / "vitalsign")

# Фильтруем витальные за 24ч до ICU поступления
vitalsign_window = vitalsign.join(
    base_population.select(["stay_id", "intime"]),
    on="stay_id",
    how="inner"
).filter(
    (pl.col("charttime") >= (pl.col("intime") - pl.duration(hours=24))) &
    (pl.col("charttime") <= pl.col("intime"))
)

# Агрегируем по mean
vitalsign_agg = vitalsign_window.group_by("stay_id").agg(
    pl.col("heart_rate").mean().alias("hr_mean"),
    pl.col("spo2").mean().alias("spo2_mean"),
    pl.col("mbp").mean().alias("mbp_mean"),
    pl.col("temperature").mean().alias("temp_mean"),
    pl.col("resp_rate").mean().alias("resp_mean"),
)

base_population = base_population.join(
    vitalsign_agg,
    on="stay_id",
    how="left"
)

print(f"Витальные признаки добавлены")

print("\n" + "="*70)
print("ШАГ 4: RRT (renal replacement therapy)")
print("="*70)

rrt = pl.read_parquet(MIMIC_DERIVED / "rrt")

# RRT до time zero
rrt_window = rrt.join(
    base_population.select(["stay_id", "intime"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("charttime") <= pl.col("intime")
).group_by("stay_id").agg(
    pl.count("dialysis_active").alias("rrt_count")
)

base_population = base_population.join(
    rrt_window.select(["stay_id", "rrt_count"]),
    on="stay_id",
    how="left"
)

base_population = base_population.with_columns(
    (pl.col("rrt_count") > 0).cast(pl.Int32).alias("rrt_flag")
)

print(f"Пациентов с RRT: {base_population.filter(pl.col('rrt_flag') == 1).shape[0]}")

print("\n" + "="*70)
print("ШАГ 5: Ventilation")
print("="*70)

ventilation = pl.read_parquet(MIMIC_DERIVED / "ventilation")

# Ventilation до time zero
vent_window = ventilation.join(
    base_population.select(["stay_id", "intime"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("charttime") <= pl.col("intime")
).group_by("stay_id").agg(
    pl.count("ventilation_active").alias("vent_count")
)

base_population = base_population.join(
    vent_window.select(["stay_id", "vent_count"]),
    on="stay_id",
    how="left"
)

base_population = base_population.with_columns(
    (pl.col("vent_count") > 0).cast(pl.Int32).alias("ventilation_flag")
)

print(f"Пациентов на вентиляции: {base_population.filter(pl.col('ventilation_flag') == 1).shape[0]}")

print("\n" + "="*70)
print("ШАГ 6: Вазопрессоры")
print("="*70)

vasoactive = pl.read_parquet(MIMIC_DERIVED / "vasoactive_agent")

# Вазопрессоры до time zero
vaso_window = vasoactive.join(
    base_population.select(["stay_id", "intime"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("starttime") <= pl.col("intime")
).group_by("stay_id").agg(
    pl.count("vasoactive_agent").alias("vasopressor_count")
)

base_population = base_population.join(
    vaso_window.select(["stay_id", "vasopressor_count"]),
    on="stay_id",
    how="left"
)

base_population = base_population.with_columns(
    (pl.col("vasopressor_count") > 0).cast(pl.Int32).alias("has_vasopressors")
)

print(f"Пациентов с вазопрессорами: {base_population.filter(pl.col('has_vasopressors') == 1).shape[0]}")

print("\n" + "="*70)
print("ШАГ 7: Лактат (из first_day_sofa и labevents)")
print("="*70)

# Из first_day_sofa
sofa = pl.read_parquet(MIMIC_DERIVED / "first_day_sofa")
lactate_sofa = sofa.select(["stay_id", "lactate"]).rename({"lactate": "lactate_sofa"})

base_population = base_population.join(
    lactate_sofa,
    on="stay_id",
    how="left"
)

# Из labevents
labevents = pl.read_parquet(MIMIC_HOSP / "labevents")
lactate_itemids = [50813, 50815, 52442, 53154]  # Lactate itemIDs

lactate_lab = labevents.filter(
    pl.col("itemid").is_in(lactate_itemids)
).join(
    base_population.select(["hadm_id", "intime"]),
    on="hadm_id",
    how="inner"
).filter(
    (pl.col("charttime") >= (pl.col("intime") - pl.duration(hours=24))) &
    (pl.col("charttime") <= pl.col("intime"))
).group_by("hadm_id").agg(
    pl.col("valuenum").mean().alias("lactate_lab")
)

base_population = base_population.join(
    lactate_lab.select(["hadm_id", "lactate_lab"]),
    on="hadm_id",
    how="left"
)

# Объединяем lactate из sofa и labevents (предпочтение labevents)
base_population = base_population.with_columns(
    pl.coalesce("lactate_lab", "lactate_sofa").alias("lactate_final")
)

print(f"Лактат добавлен, пропусков: {base_population.filter(pl.col('lactate_final').is_null()).shape[0]}")

print("\n" + "="*70)
print("ШАГ 8: Антибиотики (по классам)")
print("="*70)

prescriptions = pl.read_parquet(MIMIC_HOSP / "prescriptions")

# Классы антибиотиков
antibiotic_classes = {
    "carbapenems": ["imipenem", "meropenem", "doripenem", "ertapenem"],
    "aminoglycosides": ["gentamicin", "tobramycin", "amikacin", "streptomycin"],
    "beta_lactams": ["cefazolin", "cefuroxime", "ceftriaxone", "ceftazidime", 
                     "cefepime", "piperacillin", "ampicillin", "amoxicillin"],
    "glycopeptides": ["vancomycin", "teicoplanin"],
}

# Фильтруем prescriptions до time zero
prescriptions_window = prescriptions.join(
    base_population.select(["hadm_id", "intime"]),
    on="hadm_id",
    how="inner"
).filter(
    (pl.col("starttime") >= (pl.col("intime") - pl.duration(hours=24))) &
    (pl.col("starttime") <= pl.col("intime"))
)

# Для каждого класса создаем флаг
for cls, patterns in antibiotic_classes.items():
    cls_mask = pl.lit(False)
    for pattern in patterns:
        cls_mask = cls_mask | pl.col("drug").str.to_lowercase().str.contains(pattern)
    
    cls_abx = prescriptions_window.filter(cls_mask).select(["hadm_id"]).unique()
    cls_abx = cls_abx.with_columns(pl.lit(1).alias(f"has_{cls}"))
    
    base_population = base_population.join(
        cls_abx.select(["hadm_id", f"has_{cls}"]),
        on="hadm_id",
        how="left"
    )
    base_population = base_population.with_columns(
        pl.col(f"has_{cls}").fill_null(0).cast(pl.Int32)
    )

print("Флаги антибиотиков добавлены")

print("\n" + "="*70)
print("ШАГ 9: Charlson Comorbidity Index (из ICD кодов)")
print("="*70)

diagnoses_icd = pl.read_parquet(MIMIC_HOSP / "diagnoses_icd")

charlson_conditions = {
    "myocardial_infarct": ["I21", "I22", "I25.2"],
    "congestive_heart_failure": ["I50"],
    "peripheral_vascular_disease": ["I70", "I71", "I73"],
    "cerebrovascular_disease": ["I60", "I61", "I62", "I63", "I64"],
    "dementia": ["F00", "F01", "F02", "F03", "G30", "G31"],
    "chronic_pulmonary_disease": ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],
    "rheumatic_disease": ["M05", "M06", "M08", "M09", "M30", "M31", "M32", "M33", "M34", "M35"],
    "peptic_ulcer_disease": ["K25", "K26", "K27"],
    "mild_liver_disease": ["K70", "K71", "K73", "K74", "K76"],
    "diabetes": ["E10", "E11", "E12", "E13", "E14"],
    "paraplegia": ["G82", "G83", "S14", "S24", "S34"],
    "renal_disease": ["N18", "N19", "I12", "I13"],
    "malignant_cancer": ["C"],
    "metastatic_solid_tumor": ["C77", "C78", "C79", "C80"],
    "aids": ["B20", "B21", "B22", "B23", "B24"],
}

charlson_weights = {
    "myocardial_infarct": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes": 1,
    "paraplegia": 2,
    "renal_disease": 2,
    "malignant_cancer": 2,
    "metastatic_solid_tumor": 6,
    "aids": 6,
}

# Вычисляем Charlson
charlson_flags = {}
for condition, patterns in charlson_conditions.items():
    mask = pl.lit(False)
    for pattern in patterns:
        mask = mask | pl.col("icd_code").str.starts_with(pattern)
    
    condition_df = diagnoses_icd.filter(mask).select(["hadm_id"]).unique()
    condition_df = condition_df.with_columns(pl.lit(1).alias(condition))
    charlson_flags[condition] = condition_df

# Присоединяем все флаги
for condition, flag_df in charlson_flags.items():
    base_population = base_population.join(
        flag_df,
        on="hadm_id",
        how="left"
    )
    base_population = base_population.with_columns(
        pl.col(condition).fill_null(0).cast(pl.Int32)
    )

# Вычисляем Charlson Comorbidity Index
charlson_expr = sum(
    pl.col(condition) * weight 
    for condition, weight in charlson_weights.items()
)

base_population = base_population.with_columns(
    charlson_expr.alias("charlson_comorbidity_index")
)

print(f"Charlson index вычислен")
print(f"Распределение: mean={base_population['charlson_comorbidity_index'].mean():.2f}, max={base_population['charlson_comorbidity_index'].max()}")

print("\n" + "="*70)
print("ШАГ 10: Сепсис и септический шок")
print("="*70)

# Suspicion of infection
suspicion = pl.read_parquet(MIMIC_DERIVED / "suspicion_of_infection")

base_population = base_population.join(
    suspicion.select(["hadm_id", "suspected_infection_time"]),
    on="hadm_id",
    how="left"
)

# Проверяем окно времени
base_population = base_population.with_columns(
    (
        (pl.col("suspected_infection_time").is_not_null()) &
        (
            (pl.col("suspected_infection_time") >= (pl.col("intime") - pl.duration(hours=24))) &
            (pl.col("suspected_infection_time") <= (pl.col("intime") + pl.duration(hours=24)))
        )
    ).alias("has_suspicion_in_window")
)

# Сепсис = подозрение + органная дисфункция
# Органная дисфункция: лактат > 2 ИЛИ вазопрессоры ИЛИ вентиляция ИЛИ RRT
base_population = base_population.with_columns(
    (
        (pl.col("has_suspicion_in_window")) &
        (
            (pl.col("lactate_final") > 2.0) |
            (pl.col("has_vasopressors") == 1) |
            (pl.col("ventilation_flag") == 1) |
            (pl.col("rrt_flag") == 1)
        )
    ).alias("sepsis")
)

print(f"Пациентов с сепсисом: {base_population.filter(pl.col('sepsis') == True).shape[0]}")

# Септический шок = сепсис + вазопрессоры + лактат > 2
base_population = base_population.with_columns(
    (
        (pl.col("sepsis") == True) &
        (pl.col("has_vasopressors") == 1) &
        (pl.col("lactate_final") > 2.0)
    ).alias("septic_shock")
)

print(f"Пациентов с септическим шоком: {base_population.filter(pl.col('septic_shock') == True).shape[0]}")

print("\n" + "="*70)
print("ШАГ 11: Лечение альбумином")
print("="*70)

inputevents = pl.read_parquet(MIMIC_ICU / "inputevents")
d_items = pl.read_parquet(MIMIC_ICU / "d_items")

# Ищем albumin в d_items
albumin_items = d_items.filter(
    pl.col("label").str.to_lowercase().str.contains("albumin")
)
print(f"Найдено itemids для albumin: {albumin_items.shape[0]}")

if albumin_items.shape[0] > 0:
    albumin_itemids = albumin_items["itemid"].to_list()
    
    # Находим пациентов получавших альбумин ПОСЛЕ поступления в ICU
    albumin_input = inputevents.filter(
        pl.col("itemid").is_in(albumin_itemids)
    ).join(
        base_population.select(["stay_id", "intime"]),
        on="stay_id",
        how="inner"
    ).filter(
        pl.col("starttime") >= pl.col("intime")
    ).group_by("stay_id").agg(
        pl.lit(1).alias("treatment")
    )
    
    base_population = base_population.join(
        albumin_input.select(["stay_id", "treatment"]),
        on="stay_id",
        how="left"
    )
    
    base_population = base_population.with_columns(
        pl.col("treatment").fill_null(0).cast(pl.Int32)
    )
else:
    print("WARNING: Albumin не найден в inputevents!")
    base_population = base_population.with_columns(
        pl.lit(0).alias("treatment")
    )

print(f"Пациентов получавших альбумин: {base_population.filter(pl.col('treatment') == 1).shape[0]}")
print(f"Пациентов в контрольной группе: {base_population.filter(pl.col('treatment') == 0).shape[0]}")

print("\n" + "="*70)
print("ШАГ 12: Исход (28-day mortality)")
print("="*70)

base_population = base_population.with_columns(
    (
        (pl.col("deathtime").is_not_null()) &
        (pl.col("deathtime") <= (pl.col("intime") + pl.duration(days=28)))
    ).cast(pl.Int32).alias("mortality_28days")
)

print(f"28-day mortality: {base_population.filter(pl.col('mortality_28days') == 1).shape[0]}")

print("\n" + "="*70)
print("ШАГ 13: Финальная когорта (сепсис)")
print("="*70)

# Фильтруем только пациентов с сепсисом
cohort_sepsis = base_population.filter(pl.col("sepsis") == True)

print(f"Финальная когорта (сепсис): {cohort_sepsis.shape[0]}")
print(f"Лечение: {cohort_sepsis.filter(pl.col('treatment') == 1).shape[0]}")
print(f"Контроль: {cohort_sepsis.filter(pl.col('treatment') == 0).shape[0]}")
print(f"Смертность: {cohort_sepsis.filter(pl.col('mortality_28days') == 1).shape[0]}")

print("\n" + "="*70)
print("ШАГ 14: Отбор конфаундеров и импутация")
print("="*70)

# Выбираем нужные колонки
confounder_cols = [
    "stay_id", "subject_id", "hadm_id",
    "treatment", "mortality_28days", "sepsis", "septic_shock",
    "admission_age", "Female", "Male", "White", "Black", "Hispanic",
    "emergency_admission", "insurance_Medicare", "insurance_Medicaid",
    "lactate_final",
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    "has_vasopressors", "rrt_flag", "ventilation_flag",
    "charlson_comorbidity_index",
]

# Проверяем какие колонки есть
available_cols = [col for col in confounder_cols if col in cohort_sepsis.columns]
print(f"Доступные колонки: {len(available_cols)}")

cohort_final = cohort_sepsis.select(available_cols)

# Конвертируем в pandas для импутации
cohort_pd = cohort_final.to_pandas()

# Разделяем на идентификаторы, лечение, исход и конфаундеры
id_cols = ["stay_id", "subject_id", "hadm_id"]
target_cols = ["treatment", "mortality_28days", "sepsis", "septic_shock"]
confounder_vars = [col for col in available_cols if col not in id_cols + target_cols]

print(f"Конфаундеры ({len(confounder_vars)}): {confounder_vars}")

# Импутация конфаундеров
X = cohort_pd[confounder_vars].copy()

# KNN импутация
imputer = KNNImputer(n_neighbors=10)
X_imputed = imputer.fit_transform(X)

cohort_pd[confounder_vars] = X_imputed

# Проверяем что нет пропусков
print(f"Пропусков после импутации: {cohort_pd[confounder_vars].isnull().sum().sum()}")

print("\n" + "="*70)
print("ШАГ 15: Сохранение когорты")
print("="*70)

# Сохраняем
OUTPUT_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cohort_pd.to_csv(OUTPUT_DIR / "cohort_sepsis.csv", index=False)
print(f"Когорта сохранена: {OUTPUT_DIR / 'cohort_sepsis.csv'}")
print(f"Размер: {cohort_pd.shape}")

# Также сохраняем в parquet
cohort_final.write_parquet(OUTPUT_DIR / "cohort_sepsis.parquet")
print(f"Когорта сохранена: {OUTPUT_DIR / 'cohort_sepsis.parquet'}")

print("\n" + "="*70)
print("ШАГ 16: Проверка баланса до matching")
print("="*70)

# Сравниваем базовые характеристики между группами
treated = cohort_pd[cohort_pd["treatment"] == 1]
control = cohort_pd[cohort_pd["treatment"] == 0]

print(f"Treated: {treated.shape[0]}, Control: {control.shape[0]}")

# Вычисляем SMD
def compute_smd(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    smd = (mean1 - mean2) / pooled_std
    
    return smd

smd_results = []
for var in confounder_vars:
    smd = compute_smd(treated[var], control[var])
    smd_results.append({"variable": var, "smd": smd})

smd_df = pd.DataFrame(smd_results)
smd_df["abs_smd"] = smd_df["smd"].abs()
smd_df = smd_df.sort_values("abs_smd", ascending=False)

print("\nТоп-10 ковариат с наибольшим дисбалансом:")
print(smd_df.head(10))

print(f"\nКовариат с |SMD| > 0.1: {(smd_df['abs_smd'] > 0.1).sum()}")
print(f"Ковариат с |SMD| > 0.2: {(smd_df['abs_smd'] > 0.2).sum()}")

print("\n" + "="*70)
print("ГОТОВО!")
print("="*70)
