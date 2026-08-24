from pathlib import Path

import torch
import open_clip
from PIL import Image


# ==========================================
# 1. GPU 확인
# ==========================================

device = "xpu" if torch.xpu.is_available() else "cpu"

print("Device:", device)

if device == "xpu":
    print("GPU:", torch.xpu.get_device_name(0))


# ==========================================
# 2. BiomedCLIP 불러오기
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
# 3. X-ray 한 장 찾기
# ==========================================

image_path = next(Path("images").rglob("*.png"))

print("\n테스트 이미지:")
print(image_path)


# ==========================================
# 4. 이미지 전처리
# ==========================================

image = Image.open(image_path).convert("RGB")

image_tensor = (
    preprocess(image)
    .unsqueeze(0)
    .to(device)
)

print("입력 shape:", image_tensor.shape)


# ==========================================
# 5. 이미지 feature 추출
# ==========================================

with torch.no_grad():
    image_features = model.encode_image(image_tensor)

print("\n===== BiomedCLIP 테스트 성공 =====")
print("Feature shape:", image_features.shape)
print("Feature device:", image_features.device)
