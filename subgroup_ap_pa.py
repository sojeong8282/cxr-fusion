import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==========================================
# 설정
# ==========================================

MANIFEST_PATH = "data/cxr_fusion_manifest.csv"

B_PATH = "data/biomedclip_image_test_predictions.csv"
D_PATH = "data/biomedclip_multimodal_test_predictions.csv"

OUTPUT_PATH = "data/subgroup_ap_pa_results.csv"

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# ==========================================
# 1. 데이터 불러오기
# ==========================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

b = pd.read_csv(
    B_PATH
)

d = pd.read_csv(
    D_PATH
)


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


# ==========================================
# 2. B / D prediction 합치기
# ==========================================

df = pd.merge(
    b,
    d,
    on=[
        "patient_id",
        "path_to_image"
    ],
    how="inner",
    validate="one_to_one"
)

assert np.array_equal(
    df["label_b"].to_numpy(),
    df["label_d"].to_numpy()
)

df["label"] = df["label_b"].astype(int)


# ==========================================
# 3. AP / PA 정보 붙이기
# ==========================================

projection = manifest[
    [
        "path_to_image",
        "ap_pa"
    ]
].copy()

df = pd.merge(
    df,
    projection,
    on="path_to_image",
    how="left",
    validate="one_to_one"
)

df["ap_pa"] = (
    df["ap_pa"]
    .astype(str)
    .str.strip()
    .str.upper()
)


print("===== Test AP / PA 분포 =====")
print(df["ap_pa"].value_counts(dropna=False))


# ==========================================
# 4. subgroup별 bootstrap 함수
# ==========================================

def analyze_subgroup(
    subgroup_df,
    subgroup_name,
    n_bootstrap=5000,
    random_seed=42
):

    subgroup_df = subgroup_df.reset_index(
        drop=True
    )

    n_images = len(subgroup_df)
    n_patients = subgroup_df[
        "patient_id"
    ].nunique()

    positives = subgroup_df[
        "label"
    ].sum()

    negatives = (
        n_images - positives
    )

    print(
        f"\n===== {subgroup_name} ====="
    )

    print("Images   :", n_images)
    print("Patients :", n_patients)
    print("Positive :", positives)
    print("Negative :", negatives)


    # --------------------------------------
    # Observed AUROC
    # --------------------------------------

    auc_b = roc_auc_score(
        subgroup_df["label"],
        subgroup_df["prob_b"]
    )

    auc_d = roc_auc_score(
        subgroup_df["label"],
        subgroup_df["prob_d"]
    )

    delta = auc_d - auc_b


    print(
        f"B Image-only AUROC : {auc_b:.4f}"
    )

    print(
        f"D Multimodal AUROC : {auc_d:.4f}"
    )

    print(
        f"Delta AUROC (D-B)  : {delta:+.4f}"
    )


    # --------------------------------------
    # Patient-level paired bootstrap
    # --------------------------------------

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
        random_seed
    )

    bootstrap_delta = []


    for i in range(
        n_bootstrap
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
        f"Delta AUROC 95% CI : "
        f"{delta_ci[0]:+.4f} ~ "
        f"{delta_ci[1]:+.4f}"
    )


    return {
        "subgroup": subgroup_name,
        "images": n_images,
        "patients": n_patients,
        "positive": positives,
        "negative": negatives,
        "auc_b": auc_b,
        "auc_d": auc_d,
        "delta_auc": delta,
        "delta_ci_lower": delta_ci[0],
        "delta_ci_upper": delta_ci[1]
    }


# ==========================================
# 5. AP / PA 각각 분석
# ==========================================

results = []

for projection_name in [
    "AP",
    "PA"
]:

    subgroup = df[
        df["ap_pa"]
        == projection_name
    ].copy()

    if len(subgroup) == 0:
        print(
            f"\n{projection_name} 데이터 없음"
        )
        continue

    result = analyze_subgroup(
        subgroup,
        projection_name,
        N_BOOTSTRAP,
        RANDOM_SEED
    )

    results.append(
        result
    )


# ==========================================
# 6. 저장
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