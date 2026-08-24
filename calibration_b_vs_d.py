import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ==========================================
# 설정
# ==========================================

B_PATH = "data/biomedclip_image_test_predictions.csv"
D_PATH = "data/biomedclip_multimodal_test_predictions.csv"

CALIBRATION_OUTPUT = "data/calibration_b_vs_d.png"
BOOTSTRAP_OUTPUT = "data/bootstrap_brier_b_vs_d.csv"

N_BINS = 10
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
# 2. 같은 Test sample인지 확인
# ==========================================

df = pd.merge(
    b,
    d,
    on=["patient_id", "path_to_image"],
    how="inner",
    validate="one_to_one"
)

assert len(df) == len(b) == len(d), \
    "B와 D prediction sample 수가 다릅니다."

assert np.array_equal(
    df["label_b"].to_numpy(),
    df["label_d"].to_numpy()
), "B와 D label이 다릅니다."

df["label"] = df["label_b"].astype(int)

print("===== 데이터 확인 =====")
print("Test images :", len(df))
print("Test patients:", df["patient_id"].nunique())


# ==========================================
# 3. Brier score
# ==========================================

brier_b = brier_score_loss(
    df["label"],
    df["prob_b"]
)

brier_d = brier_score_loss(
    df["label"],
    df["prob_d"]
)

delta_brier = brier_d - brier_b

print("\n===== Brier Score =====")

print(
    f"B Image-only : {brier_b:.6f}"
)

print(
    f"D Multimodal : {brier_d:.6f}"
)

print(
    f"Delta Brier (D-B): {delta_brier:+.6f}"
)

if delta_brier < 0:
    print(
        "→ D가 B보다 Brier score가 낮아 "
        "확률 예측이 소폭 더 좋습니다."
    )
else:
    print(
        "→ D가 B보다 Brier score가 높아 "
        "확률 예측이 소폭 더 나쁩니다."
    )


# ==========================================
# 4. Calibration Curve
# ==========================================

# quantile:
# 각 bin에 비슷한 개수의 sample이 들어가도록 분할
prob_true_b, prob_pred_b = calibration_curve(
    df["label"],
    df["prob_b"],
    n_bins=N_BINS,
    strategy="quantile"
)

prob_true_d, prob_pred_d = calibration_curve(
    df["label"],
    df["prob_d"],
    n_bins=N_BINS,
    strategy="quantile"
)


plt.figure(figsize=(7, 7))

# 완벽한 calibration 기준선
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    prob_pred_b,
    prob_true_b,
    marker="o",
    label=f"B: Image-only (Brier={brier_b:.4f})"
)

plt.plot(
    prob_pred_d,
    prob_true_d,
    marker="o",
    label=f"D: Image + Text (Brier={brier_d:.4f})"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive proportion")

plt.title(
    "Calibration Curve: Cardiomegaly Prediction"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    CALIBRATION_OUTPUT,
    dpi=300
)

plt.close()

print("\nCalibration curve 저장 완료")
print("파일:", CALIBRATION_OUTPUT)


# ==========================================
# 5. Patient-level paired bootstrap
#    Brier score 차이
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

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_results = []

print("\n===== Brier Bootstrap 시작 =====")
print("Patients:", n_patients)
print("반복:", N_BOOTSTRAP)


for i in range(N_BOOTSTRAP):

    # 환자 단위 복원추출
    sampled_patients = rng.choice(
        patients,
        size=n_patients,
        replace=True
    )

    # 선택된 환자의 모든 이미지 가져오기
    sampled_indices = np.concatenate([
        patient_groups[p]
        for p in sampled_patients
    ])

    boot = df.loc[sampled_indices]

    boot_b = brier_score_loss(
        boot["label"],
        boot["prob_b"]
    )

    boot_d = brier_score_loss(
        boot["label"],
        boot["prob_d"]
    )

    bootstrap_results.append({
        "iteration": i,
        "brier_b": boot_b,
        "brier_d": boot_d,
        "delta_brier": boot_d - boot_b
    })


results = pd.DataFrame(
    bootstrap_results
)


# ==========================================
# 6. 95% CI
# ==========================================

b_ci = np.percentile(
    results["brier_b"],
    [2.5, 97.5]
)

d_ci = np.percentile(
    results["brier_d"],
    [2.5, 97.5]
)

delta_ci = np.percentile(
    results["delta_brier"],
    [2.5, 97.5]
)


print(
    "\n===== Patient-level Paired Bootstrap: Brier ====="
)

print(
    f"B Brier: "
    f"{brier_b:.4f} "
    f"(95% CI {b_ci[0]:.4f} ~ {b_ci[1]:.4f})"
)

print(
    f"D Brier: "
    f"{brier_d:.4f} "
    f"(95% CI {d_ci[0]:.4f} ~ {d_ci[1]:.4f})"
)

print(
    f"Delta Brier (D-B): "
    f"{delta_brier:+.4f} "
    f"(95% CI {delta_ci[0]:+.4f} ~ {delta_ci[1]:+.4f})"
)


# ==========================================
# 7. Bootstrap 결과 저장
# ==========================================

results.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False
)

print("\nBootstrap 결과 저장 완료")
print("파일:", BOOTSTRAP_OUTPUT)