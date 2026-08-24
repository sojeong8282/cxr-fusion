from pathlib import Path
import pandas as pd

IMAGE_DIR = Path("images")
MANIFEST_PATH = Path("data/cxr_fusion_manifest.csv")

manifest = pd.read_csv(MANIFEST_PATH)

# manifest의 원본 경로
# train/patient00034/...jpg
# valid/patient64580/...jpg
# ↓
# patient00034/...png
manifest["png_path"] = (
    manifest["path_to_image"]
    .str.replace(r"^(train|valid)/", "", regex=True)
    .str.replace(r"\.jpg$", ".png", regex=True)
)

expected = set(manifest["png_path"])

# 실제 다운로드된 PNG
actual_files = list(IMAGE_DIR.rglob("*.png"))

actual = {
    str(p.relative_to(IMAGE_DIR)).replace("\\", "/")
    for p in actual_files
}

missing = expected - actual
extra = actual - expected

print("===== 이미지 검증 =====")
print(f"Manifest 이미지: {len(expected)}")
print(f"실제 PNG:       {len(actual)}")
print(f"누락:           {len(missing)}")
print(f"추가 파일:      {len(extra)}")

if missing:
    print("\n누락 예시:")
    for x in list(missing)[:10]:
        print(x)

if extra:
    print("\n추가 파일 예시:")
    for x in list(extra)[:10]:
        print(x)
    