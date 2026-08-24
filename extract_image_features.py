from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==========================================
# 설정
# ==========================================

MANIFEST_PATH = "data/cxr_fusion_manifest.csv"
IMAGE_DIR = Path("images")
OUTPUT_PATH = "data/biomedclip_image_features.npz"

BATCH_SIZE = 32

device = "xpu" if torch.xpu.is_available() else "cpu"

print("Device:", device)

if device == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))


# ==========================================
# BiomedCLIP
# ==========================================

model_name = (
    "hf-hub:"
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

print("\nBiomedCLIP 로딩 중...")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name
)

model = model.to(device)
model.eval()

print("BiomedCLIP 로딩 완료")


# ==========================================
# Manifest
# ==========================================

df = pd.read_csv(MANIFEST_PATH)

df["png_path"] = (
    df["path_to_image"]
    .str.replace(r"^(train|valid)/", "", regex=True)
    .str.replace(r"\.jpg$", ".png", regex=True)
)

print("\n전체 이미지:", len(df))


# ==========================================
# Dataset
# ==========================================

class CXRDataset(Dataset):

    def __init__(self, dataframe, image_dir, preprocess):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_dir / row["png_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        return image


dataset = CXRDataset(
    df,
    IMAGE_DIR,
    preprocess
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# Feature extraction
# ==========================================

all_features = []

print("\n===== Feature extraction 시작 =====")

with torch.no_grad():

    for images in tqdm(loader):

        images = images.to(device)

        features = model.encode_image(images)

        # CLIP feature normalization
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
# 저장
# ==========================================

print("\nFeature shape:", features.shape)

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