import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ==========================================
# 1. 파일 설정
# ==========================================

MODEL_FILES = {
    "A: EfficientNet Image-only":
        "data/efficientnet_image_test_predictions.csv",

    "B: BiomedCLIP Image-only":
        "data/biomedclip_image_test_predictions.csv",

    "C: Clinical Text-only":
        "data/biomedclip_text_test_predictions.csv",

    "D: Image + Clinical Text":
        "data/biomedclip_multimodal_test_predictions.csv",
}

ROC_OUTPUT = "data/roc_abcd.png"
PR_OUTPUT = "data/pr_abcd.png"


# ==========================================
# 2. 데이터 불러오기
# ==========================================

models = {}

reference = None

for model_name, path in MODEL_FILES.items():

    df = pd.read_csv(path)

    # 정렬해서 같은 test sample인지 확인
    df = df.sort_values(
        ["patient_id", "path_to_image"]
    ).reset_index(drop=True)

    if reference is None:
        reference = df[
            ["patient_id", "path_to_image", "label"]
        ].copy()

    else:
        assert (
            df["patient_id"].equals(reference["patient_id"])
        ), f"{model_name}: patient_id 순서 불일치"

        assert (
            df["path_to_image"].equals(reference["path_to_image"])
        ), f"{model_name}: path 순서 불일치"

        assert (
            df["label"].equals(reference["label"])
        ), f"{model_name}: label 불일치"

    models[model_name] = df


y_true = reference["label"].to_numpy()

print("===== 데이터 확인 =====")
print("Test images :", len(y_true))
print("Positive    :", y_true.sum())
print("Negative    :", len(y_true) - y_true.sum())
print("Prevalence  :", y_true.mean())


# ==========================================
# 3. ROC Curve
# ==========================================

plt.figure(figsize=(8, 7))

for model_name, df in models.items():

    y_prob = df["probability"].to_numpy()

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUROC={auc:.4f})"
    )


# Random classifier 기준선
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves: Cardiomegaly Prediction"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(
    loc="lower right",
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    ROC_OUTPUT,
    dpi=300
)

plt.close()

print("\nROC curve 저장 완료")
print("파일:", ROC_OUTPUT)


# ==========================================
# 4. Precision-Recall Curve
# ==========================================

plt.figure(figsize=(8, 7))

for model_name, df in models.items():

    y_prob = df["probability"].to_numpy()

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    auprc = average_precision_score(
        y_true,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AUPRC={auprc:.4f})"
    )


# 무작위 모델 기준선 = 양성 비율
prevalence = y_true.mean()

plt.axhline(
    prevalence,
    linestyle="--",
    label=f"Prevalence ({prevalence:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curves: Cardiomegaly Prediction"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(
    loc="lower left",
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PR_OUTPUT,
    dpi=300
)

plt.close()

print("\nPR curve 저장 완료")
print("파일:", PR_OUTPUT)


# ==========================================
# 5. 성능표 출력
# ==========================================

print("\n===== Model Performance =====")

for model_name, df in models.items():

    y_prob = df["probability"].to_numpy()

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    auprc = average_precision_score(
        y_true,
        y_prob
    )

    print(
        f"{model_name}\n"
        f"  AUROC : {auc:.4f}\n"
        f"  AUPRC : {auprc:.4f}"
    )