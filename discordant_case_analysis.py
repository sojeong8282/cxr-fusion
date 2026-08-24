import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_curve


# ==========================================
# 경로
# ==========================================

IMAGE_FEATURE_PATH = "data/biomedclip_image_features.npz"
TEXT_FEATURE_PATH = "data/biomedclip_text_features.npz"

B_MODEL_PATH = "data/biomedclip_image_logistic.pkl"
D_MODEL_PATH = "data/biomedclip_multimodal_logistic.pkl"

MANIFEST_PATH = "data/cxr_fusion_manifest.csv"
META_PATH = "data/df_chexpert_plus_240401.csv"

OUTPUT_ALL = "data/discordant_cases_all.csv"
OUTPUT_HELP = "data/discordant_text_helped.csv"
OUTPUT_HURT = "data/discordant_text_hurt.csv"


# ==========================================
# 1. Feature / Model
# ==========================================

image_data = np.load(
    IMAGE_FEATURE_PATH,
    allow_pickle=True
)

text_data = np.load(
    TEXT_FEATURE_PATH,
    allow_pickle=True
)

model_b = joblib.load(
    B_MODEL_PATH
)

model_d = joblib.load(
    D_MODEL_PATH
)

assert np.array_equal(
    image_data["path_to_image"],
    text_data["path_to_image"]
)

assert np.array_equal(
    image_data["labels"],
    text_data["labels"]
)


X_image = image_data["features"]
X_text = text_data["features"]

X_fusion = np.concatenate(
    [X_image, X_text],
    axis=1
)

y = image_data["labels"].astype(int)
split = image_data["split"]


# ==========================================
# 2. Validation probability
# ==========================================

val_mask = split == "val"

y_val = y[val_mask]

b_val_prob = model_b.predict_proba(
    X_image[val_mask]
)[:, 1]

d_val_prob = model_d.predict_proba(
    X_fusion[val_mask]
)[:, 1]


# ==========================================
# 3. Validation에서 Youden threshold 선택
# ==========================================

def get_youden_threshold(
    y_true,
    probability
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probability
    )

    youden = (
        tpr - fpr
    )

    best_idx = np.argmax(
        youden
    )

    return thresholds[best_idx]


threshold_b = get_youden_threshold(
    y_val,
    b_val_prob
)

threshold_d = get_youden_threshold(
    y_val,
    d_val_prob
)


print("===== Validation Threshold =====")
print(
    f"B Image-only threshold : {threshold_b:.4f}"
)
print(
    f"D Multimodal threshold : {threshold_d:.4f}"
)


# ==========================================
# 4. Test probability
# ==========================================

test_mask = split == "test"

y_test = y[test_mask]

path_test = image_data[
    "path_to_image"
][test_mask]

patient_test = image_data[
    "patient_id"
][test_mask]


b_prob = model_b.predict_proba(
    X_image[test_mask]
)[:, 1]

d_prob = model_d.predict_proba(
    X_fusion[test_mask]
)[:, 1]


b_pred = (
    b_prob >= threshold_b
).astype(int)

d_pred = (
    d_prob >= threshold_d
).astype(int)


# ==========================================
# 5. 기본 DataFrame
# ==========================================

df = pd.DataFrame({
    "patient_id": patient_test,
    "path_to_image": path_test,
    "label": y_test,

    "b_probability": b_prob,
    "d_probability": d_prob,

    "b_prediction": b_pred,
    "d_prediction": d_pred
})


df["b_correct"] = (
    df["b_prediction"]
    == df["label"]
)

df["d_correct"] = (
    df["d_prediction"]
    == df["label"]
)


# D가 probability를 얼마나 정답 방향으로 움직였는지
df["probability_change"] = (
    df["d_probability"]
    - df["b_probability"]
)

df["improvement_score"] = (
    (2 * df["label"] - 1)
    * df["probability_change"]
)


# ==========================================
# 6. Clinical text / APPA 붙이기
# ==========================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

manifest = manifest[
    [
        "path_to_image",
        "clinical_text",
        "ap_pa"
    ]
].drop_duplicates(
    subset="path_to_image"
)


df = pd.merge(
    df,
    manifest,
    on="path_to_image",
    how="left",
    validate="one_to_one"
)


# ==========================================
# 7. Age / Sex 붙이기
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
    validate="one_to_one"
)


# ==========================================
# 8. Discordant 분류
# ==========================================

def classify_case(row):

    if (
        not row["b_correct"]
        and row["d_correct"]
    ):
        return "Text helped"

    if (
        row["b_correct"]
        and not row["d_correct"]
    ):
        return "Text hurt"

    if (
        row["b_correct"]
        and row["d_correct"]
    ):
        return "Both correct"

    return "Both wrong"


df["case_type"] = df.apply(
    classify_case,
    axis=1
)


print("\n===== Case Type =====")
print(
    df["case_type"]
    .value_counts()
)


# ==========================================
# 9. Text helped
# ==========================================

helped = df[
    df["case_type"]
    == "Text helped"
].copy()

helped = helped.sort_values(
    "improvement_score",
    ascending=False
)

# 대표 case에서는 같은 환자가 반복되지 않도록
helped_unique = helped.drop_duplicates(
    subset="patient_id"
)


# ==========================================
# 10. Text hurt
# ==========================================

hurt = df[
    df["case_type"]
    == "Text hurt"
].copy()

hurt = hurt.sort_values(
    "improvement_score",
    ascending=True
)

hurt_unique = hurt.drop_duplicates(
    subset="patient_id"
)


# ==========================================
# 11. 저장
# ==========================================

df.to_csv(
    OUTPUT_ALL,
    index=False
)

helped_unique.to_csv(
    OUTPUT_HELP,
    index=False
)

hurt_unique.to_csv(
    OUTPUT_HURT,
    index=False
)


# ==========================================
# 12. 대표 사례 출력
# ==========================================

columns = [
    "patient_id",
    "path_to_image",
    "label",
    "b_probability",
    "d_probability",
    "improvement_score",
    "age",
    "sex",
    "ap_pa",
    "clinical_text"
]


print(
    "\n===== Text Helped: TOP 10 ====="
)

print(
    helped_unique[
        columns
    ].head(10).to_string(
        index=False
    )
)


print(
    "\n===== Text Hurt: TOP 10 ====="
)

print(
    hurt_unique[
        columns
    ].head(10).to_string(
        index=False
    )
)


print(
    "\n===== 저장 완료 ====="
)

print(OUTPUT_ALL)
print(OUTPUT_HELP)
print(OUTPUT_HURT)