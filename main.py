import pandas as pd

# 1. 메타데이터
meta = pd.read_csv("data/df_chexpert_plus_240401.csv")

# 2. CheXpert 질환 라벨
labels = pd.read_json(
    "data/findings_fixed.json",
    lines=True
)

print("meta:", meta.shape)
print("labels:", labels.shape)

print(meta.columns.tolist())
print(labels.columns.tolist())
# ==========================================
# 2. metadata + Cardiomegaly label 병합
# ==========================================

cardio_labels = labels[
    ["path_to_image", "Cardiomegaly"]
]

df = meta.merge(
    cardio_labels,
    on="path_to_image",
    how="inner",
    validate="one_to_one"
)

print("\n===== 병합 확인 =====")
print("병합 후:", df.shape)


# ==========================================
# 3. 영상 방향 확인
# ==========================================

print("\n===== frontal_lateral 분포 =====")
print(df["frontal_lateral"].value_counts(dropna=False))


# ==========================================
# 4. Clinical text 만들기
# ==========================================

df["clinical_text"] = (
    df["section_clinical_history"].fillna("").astype(str).str.strip()
    + " "
    + df["section_history"].fillna("").astype(str).str.strip()
).str.strip()


print("\n===== Clinical text =====")
print("전체:", len(df))
print("Clinical text 있음:", (df["clinical_text"].str.len() > 0).sum())
print("Clinical text 없음:", (df["clinical_text"].str.len() == 0).sum())


# ==========================================
# 5. Cardiomegaly label 분포 확인
# ==========================================

print("\n===== Cardiomegaly 전체 분포 =====")
print(df["Cardiomegaly"].value_counts(dropna=False))


# ==========================================
# 6. 우리가 원하는 기본 조건
#    Frontal + Clinical text 존재
# ==========================================

candidate = df[
    (df["frontal_lateral"].str.lower() == "frontal")
    &
    (df["clinical_text"].str.len() > 0)
].copy()

print("\n===== Frontal + Clinical text 후보 =====")
print("후보 수:", len(candidate))

print("\n후보의 Cardiomegaly 분포:")
print(candidate["Cardiomegaly"].value_counts(dropna=False))


print("\n환자 수:")
print(candidate["deid_patient_id"].nunique())


# 실제 clinical text 예시
print("\n===== 임상텍스트 예시 =====")
print(
    candidate[
        [
            "deid_patient_id",
            "path_to_image",
            "ap_pa",
            "clinical_text",
            "Cardiomegaly"
        ]
    ].head(10).to_string(index=False)
)
# ==========================================
# 7. 최종 라벨이 명확한 데이터만 선택
# ==========================================

clean_df = candidate[
    candidate["Cardiomegaly"].isin([0.0, 1.0])
].copy()

clean_df["Cardiomegaly"] = clean_df["Cardiomegaly"].astype(int)

print("\n===== 명확한 0/1 데이터 =====")
print("전체:", len(clean_df))
print(clean_df["Cardiomegaly"].value_counts())

print("\n환자 수:")
print(clean_df["deid_patient_id"].nunique())

import re

# ==========================================
# 8. Target term 검사
# ==========================================

target_pattern = (
    r"\bcardiomegaly\b|"
    r"\bcardiomegalic\b|"
    r"\bcardiac enlargement\b|"
    r"\bcardiac enlarged\b|"
    r"\benlarged heart\b|"
    r"\bheart enlargement\b|"
    r"\benlarged cardiac silhouette\b|"
    r"\bcardiac silhouette enlargement\b|"
    r"\benlarged cardiomediastinal silhouette\b"
)

clean_df["has_target_term"] = clean_df["clinical_text"].str.contains(
    target_pattern,
    case=False,
    regex=True,
    na=False
)

print("\n===== Target term 검사 =====")
print(clean_df["has_target_term"].value_counts())

print("\nTarget term 포함 개수:")
print(clean_df["has_target_term"].sum())

print("\nTarget term 포함 비율:")
print(f"{clean_df['has_target_term'].mean() * 100:.2f}%")

print("\n===== Target term 포함 실제 문장 =====")
print(
    clean_df.loc[
        clean_df["has_target_term"],
        ["clinical_text", "Cardiomegaly"]
    ]
    .head(30)
    .to_string(index=False)
)

# ==========================================
# 9. Target term 포함 케이스 제거
# ==========================================

final_df = clean_df[
    ~clean_df["has_target_term"]
].copy()

print("\n===== Target term 제거 후 =====")
print("이미지 수:", len(final_df))
print("환자 수:", final_df["deid_patient_id"].nunique())
print(final_df["Cardiomegaly"].value_counts())


# ==========================================
# 10. Patient-level Train / Val / Test split
# ==========================================

from sklearn.model_selection import GroupShuffleSplit

# Train 70% / 나머지 30%
split1 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(
        final_df,
        y=final_df["Cardiomegaly"],
        groups=final_df["deid_patient_id"]
    )
)

train_df = final_df.iloc[train_idx].copy()
temp_df = final_df.iloc[temp_idx].copy()


# 남은 30%를 Validation 15% / Test 15%로 절반 분할
split2 = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    split2.split(
        temp_df,
        y=temp_df["Cardiomegaly"],
        groups=temp_df["deid_patient_id"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()


# ==========================================
# 11. 결과 확인
# ==========================================

def show_split(name, data):
    print(f"\n===== {name} =====")
    print("이미지:", len(data))
    print("환자:", data["deid_patient_id"].nunique())
    print(data["Cardiomegaly"].value_counts())
    print(
        data["Cardiomegaly"]
        .value_counts(normalize=True)
        .sort_index()
    )


show_split("TRAIN", train_df)
show_split("VALIDATION", val_df)
show_split("TEST", test_df)


# ==========================================
# 12. 환자 중복 확인
# ==========================================

train_patients = set(train_df["deid_patient_id"])
val_patients = set(val_df["deid_patient_id"])
test_patients = set(test_df["deid_patient_id"])

print("\n===== 환자 중복 검사 =====")
print("Train ∩ Val:", len(train_patients & val_patients))
print("Train ∩ Test:", len(train_patients & test_patients))
print("Val ∩ Test:", len(val_patients & test_patients))
# ==========================================
# 13. 모델에 사용할 안전한 컬럼만 저장
# ==========================================

use_columns = [
    "deid_patient_id",
    "path_to_image",
    "ap_pa",
    "clinical_text",
    "Cardiomegaly"
]

train_save = train_df[use_columns].copy()
val_save = val_df[use_columns].copy()
test_save = test_df[use_columns].copy()

train_save["split"] = "train"
val_save["split"] = "val"
test_save["split"] = "test"

train_save.to_csv(
    "data/train.csv",
    index=False
)

val_save.to_csv(
    "data/val.csv",
    index=False
)

test_save.to_csv(
    "data/test.csv",
    index=False
)

final_manifest = pd.concat(
    [train_save, val_save, test_save],
    ignore_index=True
)

final_manifest.to_csv(
    "data/cxr_fusion_manifest.csv",
    index=False
)

print("\n===== 저장 완료 =====")
print("train.csv:", len(train_save))
print("val.csv:", len(val_save))
print("test.csv:", len(test_save))
print("cxr_fusion_manifest.csv:", len(final_manifest))
# ==========================================
# 14. Redivis PNG_train과 연결할 파일명 생성
# ==========================================

manifest = pd.read_csv("data/cxr_fusion_manifest.csv")

manifest["file_name"] = (
    manifest["path_to_image"]
    .str.replace(r"^train/", "", regex=True)
    .str.replace(r"\.jpg$", ".png", regex=True)
)

png_needed = manifest[["file_name"]].copy()

png_needed.to_csv(
    "data/png_needed.csv",
    index=False
)

print("\n===== PNG 다운로드 목록 생성 =====")
print("전체:", len(png_needed))
print("고유 파일명:", png_needed["file_name"].nunique())

print("\n예시:")
print(png_needed.head(5).to_string(index=False))

manifest = pd.read_csv("data/cxr_fusion_manifest.csv")

print("\n===== 원본 이미지 경로 prefix =====")
print(
    manifest["path_to_image"]
    .str.split("/")
    .str[0]
    .value_counts()
)

# ==========================================
# PNG_valid 전용 21개 목록 생성
# ==========================================

manifest = pd.read_csv("data/cxr_fusion_manifest.csv")

valid_needed = manifest[
    manifest["path_to_image"].str.startswith("valid/")
].copy()

valid_needed["file_name"] = (
    valid_needed["path_to_image"]
    .str.replace(r"^valid/", "", regex=True)
    .str.replace(r"\.jpg$", ".png", regex=True)
)

valid_needed[["file_name"]].to_csv(
    "data/png_valid_needed.csv",
    index=False
)

print("\n===== PNG valid 목록 =====")
print("개수:", len(valid_needed))
print(valid_needed["file_name"].head().to_string(index=False))