import numpy as np
import pandas as pd
import joblib


# ==========================================
# 경로
# ==========================================

IMAGE_FEATURE_PATH = "data/biomedclip_image_features.npz"
TEXT_FEATURE_PATH = "data/biomedclip_text_features.npz"

D_MODEL_PATH = "data/biomedclip_multimodal_logistic.pkl"

DISCORDANT_PATH = "data/discordant_cases_all.csv"

OUTPUT_PATH = "data/discordant_with_contributions.csv"


# ==========================================
# 1. Feature
# ==========================================

image_data = np.load(
    IMAGE_FEATURE_PATH,
    allow_pickle=True
)

text_data = np.load(
    TEXT_FEATURE_PATH,
    allow_pickle=True
)

assert np.array_equal(
    image_data["path_to_image"],
    text_data["path_to_image"]
)


X_image = image_data["features"]
X_text = text_data["features"]

paths = image_data["path_to_image"]


# ==========================================
# 2. D Logistic Regression
# ==========================================

model_d = joblib.load(
    D_MODEL_PATH
)

coef = model_d.coef_[0]

intercept = model_d.intercept_[0]


print("전체 coefficient:", len(coef))

assert len(coef) == 1024


# 앞 512 = image
# 뒤 512 = text

w_image = coef[:512]
w_text = coef[512:]


# ==========================================
# 3. 각 sample의 contribution
# ==========================================

image_contribution = (
    X_image @ w_image
)

text_contribution = (
    X_text @ w_text
)

total_logit = (
    intercept
    + image_contribution
    + text_contribution
)

probability = (
    1
    / (1 + np.exp(-total_logit))
)


# 모델 predict_proba와 일치하는지 검증
X_fusion = np.concatenate(
    [X_image, X_text],
    axis=1
)

model_probability = model_d.predict_proba(
    X_fusion
)[:, 1]

assert np.allclose(
    probability,
    model_probability,
    atol=1e-6
)

print("D probability decomposition 검증 완료")


# ==========================================
# 4. path 기준 contribution DataFrame
# ==========================================

contribution_df = pd.DataFrame({
    "path_to_image": paths,
    "intercept": intercept,
    "image_contribution": image_contribution,
    "text_contribution": text_contribution,
    "d_logit": total_logit,
    "d_probability_check": probability
})


# ==========================================
# 5. Discordant case와 결합
# ==========================================

discordant = pd.read_csv(
    DISCORDANT_PATH
)

df = pd.merge(
    discordant,
    contribution_df,
    on="path_to_image",
    how="left",
    validate="one_to_one"
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 6. 대표 case
# ==========================================

selected_patients = [
    "patient54238",
    "patient00211",
    "patient20565",
    "patient32357",
    "patient00618",
    "patient29497",
]


selected = df[
    df["patient_id"].isin(
        selected_patients
    )
].copy()


# 우리가 원하는 순서로
selected["order"] = selected[
    "patient_id"
].map({
    "patient54238": 1,
    "patient00211": 2,
    "patient20565": 3,
    "patient32357": 4,
    "patient00618": 5,
    "patient29497": 6
})

selected = selected.sort_values(
    "order"
)


columns = [
    "patient_id",
    "case_type",
    "label",
    "b_probability",
    "d_probability",
    "image_contribution",
    "text_contribution",
    "d_logit",
    "clinical_text"
]


print("\n===== Selected Case Contributions =====")

print(
    selected[
        columns
    ].to_string(
        index=False
    )
)


print("\n저장 완료:")
print(OUTPUT_PATH)