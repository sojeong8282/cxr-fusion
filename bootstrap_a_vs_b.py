import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==========================================
# 설정
# ==========================================

A_PATH = "data/efficientnet_image_test_predictions.csv"
B_PATH = "data/biomedclip_image_test_predictions.csv"

OUTPUT_PATH = "data/bootstrap_a_vs_b_results.csv"

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# ==========================================
# 1. Prediction 불러오기
# ==========================================

a = pd.read_csv(A_PATH)
b = pd.read_csv(B_PATH)

a = a.rename(
    columns={
        "label": "label_a",
        "probability": "prob_a"
    }
)

b = b.rename(
    columns={
        "label": "label_b",
        "probability": "prob_b"
    }
)


# ==========================================
# 2. 동일한 test sample인지 확인
# ==========================================

df = pd.merge(
    a,
    b,
    on=[
        "patient_id",
        "path_to_image"
    ],
    how="inner",
    validate="one_to_one"
)

assert len(df) == len(a) == len(b), \
    "A와 B prediction sample 수가 다릅니다."

assert np.array_equal(
    df["label_a"].to_numpy(),
    df["label_b"].to_numpy()
), "A와 B label이 다릅니다."

df["label"] = df["label_a"].astype(int)


print("===== 데이터 확인 =====")
print("Test images :", len(df))
print("Test patients:", df["patient_id"].nunique())


# ==========================================
# 3. 실제 Test AUROC
# ==========================================

observed_a = roc_auc_score(
    df["label"],
    df["prob_a"]
)

observed_b = roc_auc_score(
    df["label"],
    df["prob_b"]
)

observed_delta = observed_b - observed_a


print("\n===== Observed =====")

print(
    f"A EfficientNet AUROC : {observed_a:.6f}"
)

print(
    f"B BiomedCLIP AUROC   : {observed_b:.6f}"
)

print(
    f"Delta AUROC (B-A)    : {observed_delta:+.6f}"
)


# ==========================================
# 4. 환자별 row index
# ==========================================

patient_groups = {
    patient_id: group.index.to_numpy()
    for patient_id, group
    in df.groupby("patient_id", sort=False)
}

patients = np.array(
    list(patient_groups.keys())
)

n_patients = len(patients)


print("\nBootstrap patients:", n_patients)
print("Bootstrap 반복:", N_BOOTSTRAP)


# ==========================================
# 5. Patient-level paired bootstrap
# ==========================================

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_results = []


print("\n===== Bootstrap 시작 =====")


for i in range(N_BOOTSTRAP):

    # 환자를 복원추출
    sampled_patients = rng.choice(
        patients,
        size=n_patients,
        replace=True
    )

    # 선택된 환자의 모든 이미지를 포함
    sampled_indices = np.concatenate([
        patient_groups[p]
        for p in sampled_patients
    ])

    boot = df.loc[
        sampled_indices
    ]

    y = boot["label"].to_numpy()

    # 혹시 한 class만 뽑힌 경우 제외
    if len(np.unique(y)) < 2:
        continue

    auc_a = roc_auc_score(
        y,
        boot["prob_a"]
    )

    auc_b = roc_auc_score(
        y,
        boot["prob_b"]
    )

    delta = auc_b - auc_a

    bootstrap_results.append({
        "iteration": i,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": delta
    })


results = pd.DataFrame(
    bootstrap_results
)


# ==========================================
# 6. 95% CI
# ==========================================

a_ci = np.percentile(
    results["auc_a"],
    [2.5, 97.5]
)

b_ci = np.percentile(
    results["auc_b"],
    [2.5, 97.5]
)

delta_ci = np.percentile(
    results["delta_auc"],
    [2.5, 97.5]
)


print(
    "\n===== Patient-level Paired Bootstrap ====="
)

print(
    f"A AUROC: "
    f"{observed_a:.4f} "
    f"(95% CI {a_ci[0]:.4f} ~ {a_ci[1]:.4f})"
)

print(
    f"B AUROC: "
    f"{observed_b:.4f} "
    f"(95% CI {b_ci[0]:.4f} ~ {b_ci[1]:.4f})"
)

print(
    f"Delta AUROC (B-A): "
    f"{observed_delta:+.4f} "
    f"(95% CI {delta_ci[0]:+.4f} ~ {delta_ci[1]:+.4f})"
)


# ==========================================
# 7. 저장
# ==========================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nBootstrap 결과 저장 완료")
print("파일:", OUTPUT_PATH)
print("유효 Bootstrap:", len(results))