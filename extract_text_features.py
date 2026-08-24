import numpy as np
import pandas as pd
import torch
import open_clip
from tqdm import tqdm


# ==========================================
# 설정
# ==========================================

MANIFEST_PATH = "data/cxr_fusion_manifest.csv"
OUTPUT_PATH = "data/biomedclip_text_features.npz"

BATCH_SIZE = 64

device = "xpu" if torch.xpu.is_available() else "cpu"

print("Device:", device)

if device == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))


# ==========================================
# BiomedCLIP 로딩
# ==========================================

model_name = (
    "hf-hub:"
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

print("\nBiomedCLIP 로딩 중...")

model, _, _ = open_clip.create_model_and_transforms(
    model_name
)

tokenizer = open_clip.get_tokenizer(
    model_name
)

model = model.to(device)
model.eval()

print("BiomedCLIP 로딩 완료")


# ==========================================
# Manifest
# ==========================================

df = pd.read_csv(MANIFEST_PATH)

texts = (
    df["clinical_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("\n전체 Clinical text:", len(texts))

print("\n텍스트 예시:")
for text in texts[:5]:
    print("-", text)


# ==========================================
# Text Feature Extraction
# ==========================================

all_features = []

print("\n===== Text Feature extraction 시작 =====")

with torch.no_grad():

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE)
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        tokens = tokenizer(
            batch_texts,
            context_length=256
        )

        tokens = tokens.to(device)

        features = model.encode_text(tokens)

        # Image feature와 동일하게 L2 normalization
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        features = features.cpu().numpy()

        all_features.append(features)


features = np.concatenate(
    all_features,
    axis=0
)


# ==========================================
# 결과 확인
# ==========================================

print("\nFeature shape:", features.shape)

assert features.shape[0] == len(df)
assert features.shape[1] == 512


# ==========================================
# 저장
# ==========================================

np.savez_compressed(
    OUTPUT_PATH,

    features=features,

    labels=df["Cardiomegaly"].to_numpy(),

    split=df["split"].to_numpy(),

    patient_id=df["deid_patient_id"].to_numpy(),

    path_to_image=df["path_to_image"].to_numpy()
)

print("\n===== 저장 완료 =====")
print("파일:", OUTPUT_PATH)
print("Features:", features.shape)
print("Labels:", len(df))