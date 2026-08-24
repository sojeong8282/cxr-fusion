# CXR-Fusion 저장소 올리는 순서

## 0. 준비 — 이 폴더의 파일들을 CXR-Fusion 폴더에 복사

README.md, .gitignore, requirements.txt, LICENSE, results/, data/.gitkeep
를 기존 CXR-Fusion 폴더 최상단에 붙여넣습니다. (.py 스크립트는 그대로 둡니다)

## 1. 올라갈 파일 확인 — 반드시 먼저

```bash
cd CXR-Fusion
git init
git add .
git status --short          # 여기에 data/ 안 파일이 하나라도 보이면 중단
```

`df_chexpert_plus_240401.csv`, `findings_fixed.json`, `*.npz`, `*.pkl`,
`train.csv` / `val.csv` / `test.csv` 가 목록에 없어야 정상입니다.

용량도 확인:

```bash
du -sh .git
git count-objects -vH       # size-pack이 몇 MB 수준이어야 정상
```

## 2. 첫 커밋

```bash
git commit -m "CXR-Fusion: BiomedCLIP vs EfficientNet, image vs image+clinical text"
git branch -M main
```

## 3. GitHub 저장소 생성 후 연결

GitHub에서 `cxr-fusion` 이름으로 **빈 저장소**를 만든 뒤 (README 체크 해제):

```bash
git remote add origin https://github.com/sojeong8282/cxr-fusion.git
git push -u origin main
```

## 만약 실수로 큰 파일을 커밋했다면

push 전이라면:

```bash
git rm -r --cached data
git commit --amend -C HEAD
```

이미 push 했다면 히스토리에 남으므로, 저장소를 지우고 다시 만드는 편이 빠릅니다.
