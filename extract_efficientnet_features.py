from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)
from tqdm import tqdm


MANIFEST_PATH = "data/cxr_fusion_manifest.csv"
IMAGE_DIR = Path("images")
OUTPUT_PATH = "data/efficientnet_image_features.npz"

BATCH_SIZE = 32

device = "xpu" if torch.xpu.is_available() else "cpu"

print("Device:", device)

if device == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))


# ==========================================
# EfficientNet-B0
# ==========================================

print("\nEfficientNet-B0 로딩 중...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

# 마지막 ImageNet 1000-class 분류기 제거
# → 1280차원 feature 출력
model.classifier = torch.nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("EfficientNet-B0 로딩 완료")


# ==========================================
# Manifest
# ==========================================

df = pd.read_csv(MANIFEST_PATH)

df["png_path"] = (
    df["path_to_image"]
    .str.replace(r"^(train|valid)/", "", regex=True)
    .str.replace(r"\.jpg$", ".png", regex=True)
)

print("전체 이미지:", len(df))


# ==========================================
# Dataset
# ==========================================

class CXRDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_dir / row["png_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

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

print("\n===== EfficientNet Feature extraction 시작 =====")

with torch.no_grad():

    for images in tqdm(loader):

        images = images.to(device)

        features = model(images)

        # B와 마찬가지로 feature 정규화
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        all_features.append(
            features.cpu().numpy()
        )


features = np.concatenate(
    all_features,
    axis=0
)

print("\nFeature shape:", features.shape)


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