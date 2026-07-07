"""
Создание когорты для causal inference анализа альбумина при сепсисе
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

MIMIC_DIR = Path("/Users/faritsharafutdinov/untitled folder/mimic-iv-3.1/mimiciv_as_parquet")
MIMIC_ICU = MIMIC_DIR / "mimiciv_icu"
MIMIC_HOSP = MIMIC_DIR / "mimiciv_hosp"
MIMIC_DERIVED = MIMIC_DIR / "mimiciv_derived"

pd.set_option("display.max_columns", None)

cohort_counts = {}

print("="*70)
print("="*70)

# Загружаем icustays
icustays = pl.read_parquet(MIMIC_ICU / "icustays")
cohort_counts["all_icu_stays"] = icustays.shape[0]
print(f"Всего ICU stays: {cohort_counts['all_icu_stays']}")

patients = pl.read_parquet(MIMIC_HOSP / "patients")

base_population = icustays.join(
    patients.select(["subject_id", "gender", "anchor_age", "anchor_year", "dod"]),
    on="subject_id",
    how="left"
)

base_population = base_population.with_columns(
    (
        pl.col("anchor_age") + 
        (pl.col("intime").dt.year() - pl.col("anchor_year"))
    ).alias("admission_age")
)

base_population = base_population.filter(pl.col("admission_age") >= 18)
cohort_counts["adults_age_ge_18"] = base_population.shape[0]
print(f"После фильтра возраст >= 18: {cohort_counts['adults_age_ge_18']}")

base_population = base_population.with_columns(
    (pl.col("los") * 24).alias("los_icu_hours")
)
base_population = base_population.filter(pl.col("los_icu_hours") >= 24)
cohort_counts["los_ge_24h"] = base_population.shape[0]
print(f"После фильтра LOS >= 24ч: {cohort_counts['los_ge_24h']}")

base_population = base_population.sort(["subject_id", "intime"]).groupby("subject_id").first()
cohort_counts["first_stay"] = base_population.shape[0]
print(f"Первый ICU stay: {cohort_counts['first_stay']}")

base_population = base_population.with_columns(
    (pl.col("gender") == "F").cast(pl.Int32).alias("Female"),
    (pl.col("gender") == "M").cast(pl.Int32).alias("Male")
)

print("\n" + "="*70)
print("="*70)

input_events = pl.read_parquet(MIMIC_ICU / "inputevents")

crystalloids_itemids = [
    226364, 226375, 225158, 225159, 225161,
    220967, 220968, 220964, 220965,
]

crystalloids_first = (
    input_events.filter(pl.col("itemid").is_in(crystalloids_itemids))
    .join(base_population.select(["stay_id", "intime"]), on="stay_id", how="inner")
    .sort(["stay_id", "starttime"])
    .groupby("stay_id")
    .first()
    .select(["stay_id", "starttime", "intime"])
    .rename({"starttime": "time_zero"})
)

crystalloids_first = crystalloids_first.with_columns(
    ((pl.col("time_zero") - pl.col("intime")).dt.hours()).alias("hours_from_icu")
)

crystalloids_first = crystalloids_first.filter(
    (pl.col("hours_from_icu") >= 0) & (pl.col("hours_from_icu") <= 24)
)
cohort_counts["crystalloids_in_24h"] = crystalloids_first.shape[0]
print(f"Пациенты с кристаллоидами в первые 24ч: {cohort_counts['crystalloids_in_24h']}")

base_population = base_population.join(
    crystalloids_first.select(["stay_id", "time_zero"]),
    on="stay_id",
    how="inner"
)
cohort_counts["with_time_zero"] = base_population.shape[0]
print(f"После добавления time_zero: {cohort_counts['with_time_zero']}")

print("\n" + "="*70)
print("="*70)

admissions = pl.read_parquet(MIMIC_HOSP / "admissions")

base_population = base_population.join(
    admissions.select(["subject_id", "hadm_id", "admission_type", "insurance", "race", "admittime", "dischtime", "deathtime"]),
    on=["subject_id", "hadm_id"],
    how="left"
)

base_population = base_population.with_columns(
    (pl.col("admission_type") == "EMERGENCY").cast(pl.Int32).alias("emergency_admission"),
    (pl.col("insurance") == "Medicare").cast(pl.Int32).alias("insurance_Medicare"),
    (pl.col("insurance") == "Medicaid").cast(pl.Int32).alias("insurance_Medicaid"),
    (pl.col("race").str.to_lowercase().str.contains("white")).cast(pl.Int32).alias("White"),
    (pl.col("race").str.to_lowercase().str.contains("black")).cast(pl.Int32).alias("Black"),
    (pl.col("race").str.to_lowercase().str.contains("hispanic")).cast(pl.Int32).alias("Hispanic"),
)

cohort_counts["with_demographics"] = base_population.shape[0]
print(f"После admissions: {cohort_counts['with_demographics']}")

print("\n" + "="*70)
print("="*70)

vitalsign = pl.read_parquet(MIMIC_DERIVED / "vitalsign")

# Витальные за 24ч ДО time_zero
vitalsign_window = vitalsign.join(
    base_population.select(["stay_id", "time_zero"]),
    on="stay_id",
    how="inner"
).filter(
    (pl.col("charttime") >= (pl.col("time_zero") - pl.duration(hours=24))) &
    (pl.col("charttime") <= pl.col("time_zero"))
)

vitalsign_agg = vitalsign_window.groupby("stay_id").agg(
    pl.col("heart_rate").mean().alias("hr_mean"),
    pl.col("spo2").mean().alias("spo2_mean"),
    pl.col("mbp").mean().alias("mbp_mean"),
    pl.col("temperature").mean().alias("temp_mean"),
    pl.col("resp_rate").mean().alias("resp_mean"),
)

base_population = base_population.join(vitalsign_agg, on="stay_id", how="left")
print("Витальные признаки добавлены (за 24ч до time_zero)")

print("\n" + "="*70)
print("="*70)


rrt = pl.read_parquet(MIMIC_DERIVED / "rrt")
rrt_window = rrt.join(
    base_population.select(["stay_id", "time_zero"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("charttime") <= pl.col("time_zero")
).groupby("stay_id").agg(
    pl.count("dialysis_active").alias("rrt_count")
)

base_population = base_population.join(rrt_window.select(["stay_id", "rrt_count"]), on="stay_id", how="left")
base_population = base_population.with_columns(
    (pl.col("rrt_count") > 0).cast(pl.Int32).alias("rrt_flag")
)
print(f"RRT: {base_population.filter(pl.col('rrt_flag') == 1).shape[0]}")


ventilation = pl.read_parquet(MIMIC_DERIVED / "ventilation")
vent_window = ventilation.join(
    base_population.select(["stay_id", "time_zero"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("charttime") <= pl.col("time_zero")
).groupby("stay_id").agg(
    pl.count("ventilation_active").alias("vent_count")
)

base_population = base_population.join(vent_window.select(["stay_id", "vent_count"]), on="stay_id", how="left")
base_population = base_population.with_columns(
    (pl.col("vent_count") > 0).cast(pl.Int32).alias("ventilation_flag")
)
print(f"Ventilation: {base_population.filter(pl.col('ventilation_flag') == 1).shape[0]}")


vasoactive = pl.read_parquet(MIMIC_DERIVED / "vasoactive_agent")
vaso_window = vasoactive.join(
    base_population.select(["stay_id", "time_zero"]),
    on="stay_id",
    how="inner"
).filter(
    pl.col("starttime") <= pl.col("time_zero")
).groupby("stay_id").agg(
    pl.count("vasoactive_agent").alias("vaso_count")
)

base_population = base_population.join(vaso_window.select(["stay_id", "vaso_count"]), on="stay_id", how="left")
base_population = base_population.with_columns(
    (pl.col("vaso_count") > 0).cast(pl.Int32).alias("has_vasopressors")
)
print(f"Vasopressors: {base_population.filter(pl.col('has_vasopressors') == 1).shape[0]}")

print("\n" + "="*70)
print("="*70)


sofa = pl.read_parquet(MIMIC_DERIVED / "first_day_sofa")
lactate_sofa = sofa.select(["stay_id", "lactate"]).rename({"lactate": "lactate_sofa"})

base_population = base_population.join(lactate_sofa, on="stay_id", how="left")


labevents = pl.read_parquet(MIMIC_HOSP / "labevents")
lactate_itemids = [50813, 50815, 52442, 53154]

lactate_lab = labevents.filter(
    pl.col("itemid").is_in(lactate_itemids)
).join(
    base_population.select(["hadm_id", "time_zero"]),
    on="hadm_id",
    how="inner"
).filter(
    (pl.col("charttime") >= (pl.col("time_zero") - pl.duration(hours=24))) &
    (pl.col("charttime") <= pl.col("time_zero"))
).groupby("hadm_id").agg(
    pl.col("valuenum").mean().alias("lactate_lab")
)

base_population = base_population.join(
    lactate_lab.select(["hadm_id", "lactate_lab"]),
    on="hadm_id",
    how="left"
)


base_population = base_population.with_columns(
    pl.coalesce("lactate_lab", "lactate_sofa").alias("lactate_final")
)

# Missing indicator для лактата
base_population = base_population.with_columns(
    pl.col("lactate_final").is_null().cast(pl.Int32).alias("lactate_missing")
)

n_lactate_missing = base_population.filter(pl.col('lactate_missing') == 1).shape[0]
print(f"Лактат добавлен, пропусков: {n_lactate_missing} ({100*n_lactate_missing/base_population.shape[0]:.1f}%)")

print("\n" + "="*70)
print("="*70)

prescriptions = pl.read_parquet(MIMIC_HOSP / "prescriptions")

antibiotic_classes = {
    "carbapenems": ["imipenem", "meropenem", "doripenem", "ertapenem"],
    "aminoglycosides": ["gentamicin", "tobramycin", "amikacin", "streptomycin"],
    "beta_lactams": ["cefazolin", "cefuroxime", "ceftriaxone", "ceftazidime", 
                     "cefepime", "piperacillin", "ampicillin", "amoxicillin"],
    "glycopeptides": ["vancomycin", "teicoplanin"],
}

prescriptions_window = prescriptions.join(
    base_population.select(["hadm_id", "time_zero"]),
    on="hadm_id",
    how="inner"
).filter(
    (pl.col("starttime") >= (pl.col("time_zero") - pl.duration(hours=24))) &
    (pl.col("starttime") <= pl.col("time_zero"))
)

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

print("Антибиотики добавлены (за 24ч до time_zero)")

print("\n" + "="*70)
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
    "myocardial_infarct": 1, "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1, "cerebrovascular_disease": 1,
    "dementia": 1, "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1, "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1, "diabetes": 1,
    "paraplegia": 2, "renal_disease": 2,
    "malignant_cancer": 2, "metastatic_solid_tumor": 6, "aids": 6,
}

for condition, patterns in charlson_conditions.items():
    mask = pl.lit(False)
    for pattern in patterns:
        mask = mask | pl.col("icd_code").str.starts_with(pattern)
    
    condition_df = diagnoses_icd.filter(mask).select(["hadm_id"]).unique()
    condition_df = condition_df.with_columns(pl.lit(1).alias(condition))
    
    base_population = base_population.join(condition_df, on="hadm_id", how="left")
    base_population = base_population.with_columns(
        pl.col(condition).fill_null(0).cast(pl.Int32)
    )

charlson_expr = sum(pl.col(cond) * w for cond, w in charlson_weights.items())
base_population = base_population.with_columns(
    charlson_expr.alias("charlson_comorbidity_index")
)

print(f"Charlson index: mean={base_population['charlson_comorbidity_index'].mean():.2f}")

print("\n" + "="*70)
print("="*70)

suspicion = pl.read_parquet(MIMIC_DERIVED / "suspicion_of_infection")


base_population = base_population.join(
    suspicion.select(["hadm_id", "suspected_infection_time"]),
    on="hadm_id",
    how="left"
)



base_population = base_population.with_columns(
    (
        (pl.col("suspected_infection_time").is_not_null()) &
        (
            (pl.col("suspected_infection_time") >= (pl.col("time_zero") - pl.duration(hours=24))) &
            (pl.col("suspected_infection_time") <= (pl.col("time_zero") + pl.duration(hours=24)))
        ) &
        (
            (pl.col("lactate_final") > 2.0) |
            (pl.col("has_vasopressors") == 1) |
            (pl.col("ventilation_flag") == 1) |
            (pl.col("rrt_flag") == 1)
        )
    ).alias("sepsis")
)

sepsis_pop = base_population.filter(pl.col("sepsis") == True)
cohort_counts["sepsis_proxy"] = sepsis_pop.shape[0]
print(f"Пациентов с сепсисом: {cohort_counts['sepsis_proxy']}")


sepsis_pop = sepsis_pop.with_columns(
    (
        (pl.col("sepsis") == True) &
        (pl.col("has_vasopressors") == 1) &
        (pl.col("lactate_final") > 2.0)
    ).alias("septic_shock")
)

print(f"С септическим шоком: {sepsis_pop.filter(pl.col('septic_shock') == True).shape[0]}")

print("\n" + "="*70)
print("="*70)
print("КРИТИЧНО: Albumin должен быть в первые 24ч ПОСЛЕ кристаллоидов")

d_items = pl.read_parquet(MIMIC_ICU / "d_items")

albumin_items = d_items.filter(
    pl.col("label").str.to_lowercase().str.contains("albumin")
)
print(f"Найдено itemids для albumin: {albumin_items.shape[0]}")

if albumin_items.shape[0] > 0:
    albumin_itemids = albumin_items["itemid"].to_list()
    
    
    albumin_window = input_events.filter(
        pl.col("itemid").is_in(albumin_itemids)
    ).join(
        sepsis_pop.select(["stay_id", "time_zero"]),
        on="stay_id",
        how="inner"
    ).filter(
        (pl.col("starttime") >= pl.col("time_zero")) &
        (pl.col("starttime") <= (pl.col("time_zero") + pl.duration(hours=24)))
    ).groupby("stay_id").agg(
        pl.first("starttime").alias("albumin_time")
    )
    
    sepsis_pop = sepsis_pop.join(
        albumin_window.select(["stay_id", "albumin_time"]),
        on="stay_id",
        how="left"
    )
    
    sepsis_pop = sepsis_pop.with_columns(
        (pl.col("albumin_time").is_not_null()).cast(pl.Int32).alias("treatment")
    )
    
    
    late_albumin = input_events.filter(
        pl.col("itemid").is_in(albumin_itemids)
    ).join(
        sepsis_pop.select(["stay_id", "time_zero"]),
        on="stay_id",
        how="inner"
    ).filter(
        pl.col("starttime") > (pl.col("time_zero") + pl.duration(hours=24))
    ).groupby("stay_id").agg(
        pl.lit(1).alias("late_albumin")
    )
    
    sepsis_pop = sepsis_pop.join(
        late_albumin.select(["stay_id", "late_albumin"]),
        on="stay_id",
        how="left"
    )
    sepsis_pop = sepsis_pop.with_columns(
        pl.col("late_albumin").fill_null(0).cast(pl.Int32)
    )
    
    n_late = sepsis_pop.filter(pl.col('late_albumin') == 1).shape[0]
    print(f"Пациентов с late albumin (>24ч): {n_late} (исключаем из анализа)")
    
    
    sepsis_pop = sepsis_pop.filter(pl.col("late_albumin") == 0)
    cohort_counts["exclude_late_albumin"] = n_late
    
else:
    sepsis_pop = sepsis_pop.with_columns(pl.lit(0).alias("treatment"))
    sepsis_pop = sepsis_pop.with_columns(pl.lit(0).alias("late_albumin"))

cohort_counts["with_treatment"] = sepsis_pop.shape[0]
print(f"Сепсис когорта (после исключения late albumin): {cohort_counts['with_treatment']}")
print(f"Лечение (альбумин в первые 24ч): {sepsis_pop.filter(pl.col('treatment') == 1).shape[0]}")
print(f"Контроль: {sepsis_pop.filter(pl.col('treatment') == 0).shape[0]}")

print("\n" + "="*70)
print("="*70)

# 28-day mortality от time_zero
sepsis_pop = sepsis_pop.with_columns(
    (
        (pl.col("deathtime").is_not_null()) &
        (pl.col("deathtime") <= (pl.col("time_zero") + pl.duration(days=28)))
    ).cast(pl.Int32).alias("mortality_28days")
)

n_mortality = sepsis_pop.filter(pl.col('mortality_28days') == 1).shape[0]
cohort_counts["known_outcome"] = sepsis_pop.shape[0]
print(f"28-day mortality: {n_mortality} ({100*n_mortality/sepsis_pop.shape[0]:.1f}%)")

print("\n" + "="*70)
print("="*70)

# Отбор конфаундеров (УБРАЛИ has_vasopressors из конфаундеров - это часть определения сепсиса!)
confounder_cols = [
    "stay_id", "subject_id", "hadm_id",
    "treatment", "mortality_28days", "sepsis", "septic_shock",
    "admission_age", "Female", "Male", "White", "Black", "Hispanic",
    "emergency_admission", "insurance_Medicare", "insurance_Medicaid",
    "lactate_final", "lactate_missing",  # добавили missing indicator
    "hr_mean", "spo2_mean", "mbp_mean", "temp_mean", "resp_mean",
    "has_carbapenems", "has_aminoglycosides", "has_beta_lactams", "has_glycopeptides",
    # "has_vasopressors",  # УБРАЛИ - это part of sepsis definition
    "rrt_flag", "ventilation_flag",
    "charlson_comorbidity_index",
]

available_cols = [col for col in confounder_cols if col in sepsis_pop.columns]
cohort_final = sepsis_pop.select(available_cols)


cohort_pd = cohort_final.to_pandas()
id_cols = ["stay_id", "subject_id", "hadm_id"]
target_cols = ["treatment", "mortality_28days", "sepsis", "septic_shock"]
confounder_vars = [col for col in available_cols if col not in id_cols + target_cols]

print(f"Конфаундеры ({len(confounder_vars)}): {confounder_vars}")

X = cohort_pd[confounder_vars].copy()


imputer = KNNImputer(n_neighbors=10)
X_imputed = imputer.fit_transform(X)
cohort_pd[confounder_vars] = X_imputed

n_missing_after = cohort_pd[confounder_vars].isnull().sum().sum()
print(f"Пропусков после импутации: {n_missing_after}")


OUTPUT_DIR = Path("/Users/faritsharafutdinov/untitled folder/notebook_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cohort_pd.to_csv(OUTPUT_DIR / "cohort_sepsis.csv", index=False)
cohort_final.write_parquet(OUTPUT_DIR / "cohort_sepsis.parquet")

cohort_counts["final_cohort"] = cohort_pd.shape[0]
print(f"\nКогорта сохранена: {OUTPUT_DIR / 'cohort_sepsis.csv'}")
print(f"Финальный размер: {cohort_counts['final_cohort']}")

print("\n" + "="*70)
print("COHORT COUNTS - полная цепочка")
print("="*70)
for key, value in cohort_counts.items():
    print(f"{key:.<40} {value:>10,}")

print("\n" + "="*70)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*70)
print(f"Всего пациентов: {cohort_pd.shape[0]}")
print(f"Лечение (альбумин): {cohort_pd['treatment'].sum()} ({100*cohort_pd['treatment'].mean():.1f}%)")
print(f"Контроль: {(cohort_pd['treatment'] == 0).sum()}")
print(f"28-day mortality: {cohort_pd['mortality_28days'].sum()} ({100*cohort_pd['mortality_28days'].mean():.1f}%)")

print("\n" + "="*70)
print("КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ВНЕСЕНЫ:")
print("="*70)
print("Time zero = first crystalloid (НЕ ICU intime)")
print("Albumin в окне [time_zero, time_zero + 24h]")
print("LOS >= 24 часов (явный фильтр)")
print("Late albumin исключен из когорты")
print("Missing indicator для lactate добавлен")
print("has_vasopressors убран из конфаундеров")
print("28-day mortality от time_zero (НЕ от ICU)")
print("="*70)
