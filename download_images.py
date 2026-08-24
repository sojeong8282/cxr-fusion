from pathlib import Path
import redivis

EXPECTED_COUNT = 15174

# download_images.py가 있는 폴더 아래의 images 폴더에 저장
download_dir = Path(__file__).resolve().parent / "images"
download_dir.mkdir(parents=True, exist_ok=True)

# Redivis에서 만든 최종 15,174개 테이블
user = redivis.user("sojeongan652")
workflow = user.workflow("cxr_fusion:w3tj")
table = workflow.table("transform_3_output:36vs")

print("CXR 이미지 다운로드 시작...")

# 최신 방식
directory = table.to_directory(
    file_id_variable="file_id"
)

directory.download(
    path=download_dir,
    overwrite=True,          # 정상 파일은 건너뛰고 불완전한 파일은 다시 다운로드
    progress=True,
    max_parallelization=1,    # 기존 20보다 연결이 안정적
    max_concurrency=1       # 동시에 다운로드하는 연결도 1개
    )

# 실제 저장된 PNG 수 확인
png_count = sum(1 for _ in download_dir.rglob("*.png"))

print(f"\n다운로드 작업 종료")
print(f"실제 PNG 파일 수: {png_count}/{EXPECTED_COUNT}")
print(f"남은 파일 수: {EXPECTED_COUNT - png_count}")