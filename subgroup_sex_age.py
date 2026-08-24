import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==========================================
# 설정
# ==========================================

META_PATH = "data/df_chexpert_plus_240401.csv"

B_PATH = "data/biomedclip_image_test_predictions.csv"
D_PATH = "data/biomedclip_multimodal_test_predictions.csv"

OUTPUT_PATH = "data/subgroup_sex_age_results.csv"

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# ==========================================
# 1. B / D prediction
# ==========================================

b = pd.read_csv(B_PATH)
d = pd.read_csv(D_PATH)

b = b.rename(
    columns={
        "label": "label_b",
        "probability": "prob_b"
    }
)

d = d.rename(
    columns={
        "label": "label_d",
        "probability": "prob_d"
    }
)


df = pd.merge(
    b,
    d,
    on=["patient_id", "path_to_image"],
    how="inner",
    validate="one_to_one"
)

assert len(df) == len(b) == len(d)

assert np.array_equal(
    df["label_b"].to_numpy(),
    df["label_d"].to_numpy()
)

df["label"] = df["label_b"].astype(int)


# ==========================================
# 2. 원본 metadata에서 age / sex만 가져오기
# ==========================================

meta = pd.read_csv(
    META_PATH,
    usecols=[
        "path_to_image",
        "age",
        "sex"
    ]
)

meta = meta.drop_duplicates(
    subset="path_to_image"
)


df = pd.merge(
    df,
    meta,
    on="path_to_image",
    how="left",
    validate="many_to_one"
)


# ==========================================
# 3. Sex 정리
# ==========================================

def normalize_sex(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["male", "m"]:
        return "Male"

    if value in ["female", "f"]:
        return "Female"

    return value


df["sex_group"] = df["sex"].apply(
    normalize_sex
)


# ==========================================
# 4. Age 정리
# ==========================================

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

# 탐색적 subgroup:
# <50 / 50-69 / 70+
df["age_group"] = pd.cut(
    df["age"],
    bins=[
        -np.inf,
        50,
        70,
        np.inf
    ],
    labels=[
        "<50",
        "50-69",
        "70+"
    ],
    right=False
)


print("===== Sex 분포 =====")
print(
    df["sex_group"]
    .value_counts(dropna=False)
)

print("\n===== Age 분포 =====")
print(
    df["age_group"]
    .value_counts(dropna=False)
)

print("\nAge 통계:")
print(
    df["age"].describe()
)


# ==========================================
# 5. Subgroup 분석 함수
# ==========================================

def analyze_subgroup(
    subgroup_df,
    variable,
    subgroup_name
):

    subgroup_df = subgroup_df.reset_index(
        drop=True
    )

    n_images = len(subgroup_df)

    n_patients = subgroup_df[
        "patient_id"
    ].nunique()

    positives = int(
        subgroup_df["label"].sum()
    )

    negatives = (
        n_images - positives
    )


    print(
        f"\n===== {variable}: {subgroup_name} ====="
    )

    print("Images   :", n_images)
    print("Patients :", n_patients)
    print("Positive :", positives)
    print("Negative :", negatives)


    if subgroup_df["label"].nunique() < 2:

        print(
            "양성/음성 class가 모두 존재하지 않아 분석 제외"
        )

        return None


    # ======================================
    # Observed AUROC
    # ======================================

    auc_b = roc_auc_score(
        subgroup_df["label"],
        subgroup_df["prob_b"]
    )

    auc_d = roc_auc_score(
        subgroup_df["label"],
        subgroup_df["prob_d"]
    )

    delta_auc = auc_d - auc_b


    print(
        f"B Image-only AUROC : {auc_b:.4f}"
    )

    print(
        f"D Multimodal AUROC : {auc_d:.4f}"
    )

    print(
        f"Delta AUROC (D-B)  : {delta_auc:+.4f}"
    )


    # ======================================
    # Patient-level paired bootstrap
    # ======================================

    patient_groups = {
        patient_id:
            group.index.to_numpy()

        for patient_id, group
        in subgroup_df.groupby(
            "patient_id",
            sort=False
        )
    }


    patients = np.array(
        list(
            patient_groups.keys()
        )
    )


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    bootstrap_delta = []


    for i in range(
        N_BOOTSTRAP
    ):

        sampled_patients = rng.choice(
            patients,
            size=len(patients),
            replace=True
        )


        sampled_indices = np.concatenate([
            patient_groups[p]
            for p in sampled_patients
        ])


        boot = subgroup_df.loc[
            sampled_indices
        ]


        y = boot[
            "label"
        ].to_numpy()


        if len(
            np.unique(y)
        ) < 2:
            continue


        boot_auc_b = roc_auc_score(
            y,
            boot["prob_b"]
        )

        boot_auc_d = roc_auc_score(
            y,
            boot["prob_d"]
        )


        bootstrap_delta.append(
            boot_auc_d
            - boot_auc_b
        )


    delta_ci = np.percentile(
        bootstrap_delta,
        [2.5, 97.5]
    )


    print(
        "Delta AUROC 95% CI : "
        f"{delta_ci[0]:+.4f} ~ "
        f"{delta_ci[1]:+.4f}"
    )


    return {
        "variable": variable,
        "subgroup": subgroup_name,
        "images": n_images,
        "patients": n_patients,
        "positive": positives,
        "negative": negatives,
        "auc_b": auc_b,
        "auc_d": auc_d,
        "delta_auc": delta_auc,
        "delta_ci_lower": delta_ci[0],
        "delta_ci_upper": delta_ci[1]
    }


# ==========================================
# 6. Sex subgroup
# ==========================================

results = []


for sex in [
    "Male",
    "Female"
]:

    subgroup = df[
        df["sex_group"] == sex
    ].copy()

    if len(subgroup) == 0:
        continue

    result = analyze_subgroup(
        subgroup,
        "Sex",
        sex
    )

    if result is not None:
        results.append(
            result
        )


# ==========================================
# 7. Age subgroup
# ==========================================

for age_group in [
    "<50",
    "50-69",
    "70+"
]:

    subgroup = df[
        df["age_group"]
        == age_group
    ].copy()

    if len(subgroup) == 0:
        continue

    result = analyze_subgroup(
        subgroup,
        "Age",
        age_group
    )

    if result is not None:
        results.append(
            result
        )


# ==========================================
# 8. 결과 저장
# ==========================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n===== 최종 결과 ====="
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\n저장 완료:",
    OUTPUT_PATH
)