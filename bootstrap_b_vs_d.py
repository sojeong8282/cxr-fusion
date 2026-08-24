import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==========================================
# 설정
# ==========================================

B_PATH = "data/biomedclip_image_test_predictions.csv"
D_PATH = "data/biomedclip_multimodal_test_predictions.csv"

OUTPUT_PATH = "data/bootstrap_b_vs_d_results.csv"

N_BOOTSTRAP = 5000
RANDOM_SEED = 42


# ==========================================
# 1. Prediction 불러오기
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


# ==========================================
# 2. 같은 Test sample인지 확인 후 결합
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

assert len(df) == len(b) == len(d), \
    "B와 D prediction sample 수가 다릅니다."

assert np.array_equal(
    df["label_b"].to_numpy(),
    df["label_d"].to_numpy()
), "B와 D의 label이 다릅니다."

df["label"] = df["label_b"].astype(int)

print("===== 데이터 확인 =====")
print("Test images :", len(df))
print("Test patients:", df["patient_id"].nunique())


# ==========================================
# 3. 실제 Test AUROC
# ==========================================

observed_b = roc_auc_score(
    df["label"],
    df["prob_b"]
)

observed_d = roc_auc_score(
    df["label"],
    df["prob_d"]
)

observed_delta = observed_d - observed_b

print("\n===== Observed =====")
print(f"B Image-only AUROC : {observed_b:.6f}")
print(f"D Multimodal AUROC : {observed_d:.6f}")
print(f"Delta AUROC (D-B)  : {observed_delta:+.6f}")


# ==========================================
# 4. 환자별 row index 저장
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

    # 뽑힌 환자들의 모든 이미지 포함
    sampled_indices = np.concatenate([
        patient_groups[p]
        for p in sampled_patients
    ])

    boot = df.loc[sampled_indices]

    y = boot["label"].to_numpy()

    # 혹시 bootstrap sample에 한 class만 있는 경우 제외
    if len(np.unique(y)) < 2:
        continue

    auc_b = roc_auc_score(
        y,
        boot["prob_b"]
    )

    auc_d = roc_auc_score(
        y,
        boot["prob_d"]
    )

    delta = auc_d - auc_b

    bootstrap_results.append({
        "iteration": i,
        "auc_b": auc_b,
        "auc_d": auc_d,
        "delta_auc": delta
    })


results = pd.DataFrame(
    bootstrap_results
)


# ==========================================
# 6. 95% CI
# ==========================================

b_ci = np.percentile(
    results["auc_b"],
    [2.5, 97.5]
)

d_ci = np.percentile(
    results["auc_d"],
    [2.5, 97.5]
)

delta_ci = np.percentile(
    results["delta_auc"],
    [2.5, 97.5]
)


print("\n===== Patient-level Paired Bootstrap =====")

print(
    f"B AUROC: "
    f"{observed_b:.4f} "
    f"(95% CI {b_ci[0]:.4f} ~ {b_ci[1]:.4f})"
)

print(
    f"D AUROC: "
    f"{observed_d:.4f} "
    f"(95% CI {d_ci[0]:.4f} ~ {d_ci[1]:.4f})"
)

print(
    f"Delta AUROC (D-B): "
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