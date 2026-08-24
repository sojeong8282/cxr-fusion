# CXR-Fusion

**Does Clinical Context Add Value to a Biomedical Foundation Model for Cardiomegaly Prediction?**

흉부 X-ray에 Clinical History를 추가하면 Cardiomegaly 예측 성능과 신뢰성이 실제로 향상되는가를 검증한 프로젝트입니다.

단순히 멀티모달 모델을 만드는 것이 목적이 아니라, **강력한 Biomedical Foundation Model이 이미 X-ray를 잘 읽는 상황에서 clinical text가 추가적인 정보를 제공하는가**를 통계적으로 확인했습니다.

---

## 핵심 결과

| Model | Test AUROC | Test AUPRC | Brier |
|---|---:|---:|---:|
| **A.** EfficientNet-B0 (ImageNet) image-only | 0.8407 | 0.8472 | 0.1626 |
| **B.** BiomedCLIP image-only | 0.8791 | 0.8900 | 0.1414 |
| **C.** Clinical text-only | 0.6762 | 0.7004 | 0.2270 |
| **D.** Image + Clinical text (late fusion) | 0.8809 | 0.8914 | 0.1401 |

Patient-level paired bootstrap (5,000 iterations):

| 비교 | 지표 | 추정값 | 95% CI | 해석 |
|---|---|---:|---|---|
| A vs B | ΔAUROC | **+0.0384** | +0.0232 ~ +0.0533 | CI가 0을 포함하지 않음 |
| B vs D | ΔAUROC | **+0.0018** | −0.0028 ~ +0.0064 | CI가 0을 포함 |
| B vs D | ΔBrier | −0.0013 | −0.0042 ~ +0.0016 | CI가 0을 포함 |

<table>
<tr>
<td width="50%"><img src="results/roc_abcd.png" alt="ROC curves"></td>
<td width="50%"><img src="results/pr_abcd.png" alt="Precision-Recall curves"></td>
</tr>
<tr>
<td><b>Figure 1.</b> ROC Curves — B(BiomedCLIP)가 A(EfficientNet)보다 뚜렷하게 위에 있고, <b>B와 D는 거의 완전히 겹칩니다.</b></td>
<td><b>Figure 2.</b> Precision–Recall Curves — 점선은 test prevalence 0.523에 해당하는 random baseline입니다.</td>
</tr>
</table>

두 지표가 동일한 순서(**B ≈ D > A > C**)를 보여, 지표 선택에 따라 결론이 뒤바뀌지 않음을 확인했습니다.

### 결론

1. **이미지 representation의 영향은 컸다.** EfficientNet → BiomedCLIP에서 ΔAUROC +0.0384, 신뢰구간 전체가 0보다 큼.
2. **Clinical History 자체에는 신호가 있었다.** Text-only AUROC 0.6762로 random을 분명히 상회.
3. **그러나 단순 late fusion의 incremental benefit은 확인되지 않았다.** ΔAUROC +0.0018, CI가 0을 포함.
4. **평균 효과는 작아도 개별 환자에서는 크게 작용했다.** Text 추가로 75건이 교정되고 62건이 악화되어, 상반된 방향의 변화가 상쇄되었을 가능성.

> 이 결과는 "Foundation Model이 무조건 좋다"거나 "clinical text가 무용하다"는 의미가 **아닙니다**.
> 정확히는 *CheXpert Plus Cardiomegaly classification의 frozen-feature + late fusion setting에서* 관찰된 결과입니다.

---

## 데이터

- **Dataset:** Stanford CheXpert Plus v1.0 (원본 metadata 223,462 X-ray images)
- **Text:** `section_clinical_history` + `section_history`
  - `Findings` / `Impression`은 **판독 후 작성된 정보**이므로 label leakage 방지를 위해 전면 제외
- **Label:** Cardiomegaly (1 / 0만 사용, Uncertain(−1) 및 NaN 제외)
- Clinical History에 정답이 직접 노출된 문장(`cardiomegaly`, `enlarged heart`, `cardiac enlargement`) **19건 검출 후 제외**

### 최종 코호트

| | Images | Patients |
|---|---:|---:|
| **전체** | 15,174 | 10,980 |
| Train | 10,669 | 7,685 |
| Validation | 2,227 | 1,647 |
| Test | 2,278 | 1,648 |

Cardiomegaly Positive 8,049 / Negative 7,125 (test prevalence 0.523).
**Patient-level split**이며 Train ∩ Val ∩ Test 환자 overlap은 모두 0입니다.

---

## 모델 구성

모든 pretrained encoder는 **frozen** 상태로 사용했습니다. End-to-end fine-tuning이 아니라 feature extraction + Logistic Regression 구조를 택해, 비교 대상 간 차이를 **representation의 차이로 한정**했습니다.

```
A  Chest X-ray → EfficientNet-B0 (ImageNet)      → 1280-d → LogisticRegression
B  Chest X-ray → BiomedCLIP image encoder        →  512-d → LogisticRegression
C  Clinical History → BiomedCLIP text encoder    →  512-d → LogisticRegression
D  [Image 512-d ⊕ Text 512-d]                    → 1024-d → LogisticRegression
```

- Encoder: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Logistic Regression의 `C`는 후보 `{0.01, 0.1, 1, 10}` 중 **Validation AUROC** 기준으로 선택 (Test는 최종 평가에만 사용)

---

## Calibration — 예측 확률은 믿을 수 있는가

임상에서 모델 출력은 "양성/음성"이 아니라 **확률**로 쓰입니다. 모델이 70%라고 예측한 환자 중
실제로 약 70%가 positive일 때 그 확률을 의사결정에 사용할 수 있습니다.

<img src="results/calibration_b_vs_d.png" width="480" alt="Calibration curve">

두 모델 모두 대각선에 상당히 가까웠고 두 곡선도 매우 유사했습니다.
Image+Text가 Brier score에서 수치상 근소하게 낮았으나(0.1401 vs 0.1414),
patient-level paired bootstrap 신뢰구간이 0을 포함해
**clinical text 추가가 probability calibration을 명확하게 개선하지는 않았습니다.**

---

## Discordant case 분석 — 평균이 감추는 것

ΔAUROC **+0.0018**은 "거의 아무 일도 일어나지 않았다"는 뜻일 수도, **좋은 방향과 나쁜 방향의 변화가
서로 상쇄된 결과**일 수도 있습니다. 두 해석은 임상적 의미가 완전히 다릅니다.

Validation set에서 Youden's J로 threshold를 먼저 정하고(Image-only 0.5197, Image+Text 0.5164)
Test set에 고정 적용해 개별 환자 수준의 변화를 확인했습니다.

| 구분 | n |
|---|---:|
| Both correct | 1,742 |
| Both wrong | 399 |
| **Prediction corrected after text** | **75** |
| **Prediction worsened after text** | **62** |

교정 75건과 악화 62건으로 두 방향의 건수가 거의 비슷했고, 개별 환자의 예측 확률은
`0.30 → 0.72`, `0.73 → 0.41`, `0.36 → 0.80` 처럼 크게 움직였습니다.

> Cohort-level improvement was minimal, but clinical context substantially altered predictions
> in individual cases — sometimes correcting image-only errors and sometimes introducing new errors.

대표 사례 중에는 `"History of heart failure"`(과거력)가 positive 방향으로 작용해 false-positive를 만든 경우와,
Cardiomegaly와 관계가 뚜렷하지 않은 `"Status epilepticus"`가 가장 큰 positive 기여(+2.146)를 보인 경우가 있었습니다.
후자는 text encoder가 **dataset-specific shortcut 또는 spurious association을 이용할 가능성**을 시사하지만,
본 분석은 이를 증명한 것이 아니라 가능성을 제시한 것입니다.

> 대표 사례 패널(`plot_case_analysis.py` 출력)은 CheXpert Plus의 실제 X-ray 이미지를 포함하므로
> data use agreement에 따라 저장소에 포함하지 않았습니다.

---

## Subgroup 분석 <sub>(exploratory)</sub>

Clinical text의 효과가 특정 환자군에서만 나타나는지 확인했습니다.
**사전에 검정력을 확보하도록 설계된 분석이 아니며 다중비교 보정도 적용하지 않았습니다.**

| Subgroup | n | B (Image) | D (Image+Text) | ΔAUROC | 95% CI |
|---|---:|---:|---:|---:|---|
| AP | 1,713 | 0.8642 | 0.8663 | +0.0021 | −0.0039 ~ +0.0081 |
| PA | 564 | 0.8954 | 0.8974 | +0.0020 | −0.0070 ~ +0.0111 |
| Male | 1,382 | 0.9009 | 0.8990 | −0.0019 | −0.0077 ~ +0.0037 |
| Female | 894 | 0.8427 | 0.8499 | +0.0072 | −0.0005 ~ +0.0152 |
| Age &lt;50 | 550 | 0.9350 | 0.9413 | +0.0062 | −0.0005 ~ +0.0141 |
| Age 50–69 | 861 | 0.8710 | 0.8669 | −0.0042 | −0.0117 ~ +0.0030 |
| Age 70+ | 865 | 0.8339 | 0.8395 | +0.0056 | −0.0039 ~ +0.0152 |

**모든 subgroup에서 신뢰구간이 0을 포함**했습니다. Female(+0.0072)과 Age &lt;50(+0.0062)에서 상대적으로
큰 증가 경향이 보였으나 exploratory signal일 뿐 명확한 subgroup effect로 주장할 수 없습니다.

AP는 portable·bedside 촬영에서 흔하고 심장이 확대되어 보이는 magnification 문제가 있어 별도로 확인했습니다.
Image-only 성능이 AP(0.8642)에서 PA(0.8954)보다 낮았지만, clinical text가 AP에서 특별히 더 도움이 되지는 않았습니다.

---

## 실행 순서

```bash
pip install -r requirements.txt

# 0) 데이터 준비 (CheXpert Plus 접근 권한 필요 — 아래 Data Access 참조)
python main.py                          # metadata + label 병합, 코호트 구성, patient-level split
python download_images.py               # 이미지 다운로드
python check_images.py                  # 유효 이미지 검증

# 1) Feature extraction (frozen encoders)
python extract_image_features.py        # BiomedCLIP image  → 512-d
python extract_text_features.py         # BiomedCLIP text   → 512-d
python extract_efficientnet_features.py # EfficientNet-B0   → 1280-d

# 2) 분류기 학습 (A / B / C / D)
python train_efficientnet_classifier.py
python train_biomedclip_classifier.py
python train_text_classifier.py
python train_multimodal_classifier.py

# 3) 통계 검증
python bootstrap_a_vs_b.py              # ΔAUROC CI (representation 효과)
python bootstrap_b_vs_d.py              # ΔAUROC CI (clinical text의 incremental value)
python calibration_b_vs_d.py            # Calibration curve + Brier + ΔBrier CI
python subgroup_ap_pa.py                # AP / PA subgroup (exploratory)
python subgroup_sex_age.py              # Sex / Age subgroup (exploratory)

# 4) Failure analysis & 시각화
python discordant_case_analysis.py      # Youden threshold 기반 discordant case 분해
python analyze_text_contribution.py     # logit을 image / text 기여로 분리
python plot_abcd_curves.py              # ROC / PR curve
python plot_case_analysis.py            # 대표 사례 패널
```

---

## 결과 파일

`results/` 폴더에는 **집계된 결과만** 포함되어 있습니다.

| 파일 | 내용 |
|---|---|
| `main_results.csv` | A/B/C/D 최종 성능 |
| `bootstrap_summary.csv` | ΔAUROC / ΔBrier 95% CI |
| `subgroup_ap_pa_results.csv` | AP / PA subgroup |
| `subgroup_sex_age_results.csv` | Sex / Age subgroup |
| `discordant_summary.csv` | Discordant case 집계 |
| `roc_abcd.png` / `pr_abcd.png` | ROC / Precision-Recall curve |
| `calibration_b_vs_d.png` | Calibration curve (B vs D) |

---

## Data Access

이 저장소에는 **CheXpert Plus 원본 데이터, 파생 메타데이터, 리포트 텍스트, X-ray 이미지가 포함되어 있지 않습니다.**

CheXpert Plus는 Stanford AIMI의 data use agreement 하에 배포되며 재배포가 허용되지 않습니다.
재현하려면 아래에서 직접 접근 권한을 받아 `data/` 폴더에 배치해야 합니다.

- Stanford AIMI Shared Datasets — CheXpert Plus
- 필요 파일: `df_chexpert_plus_240401.csv`, `findings_fixed.json`, 그리고 이미지

---

## 한계

1. Cardiomegaly 단일 target만 분석했습니다.
2. CheXpert Plus 단일 dataset 내부 평가이며 **external validation이 없습니다.**
3. Cardiomegaly label은 **report-derived reference label**로, 별도 재판독을 거치지 않았습니다.
4. Frozen encoder + Logistic Regression 설정입니다. Fine-tuning 시 결과가 달라질 수 있습니다.
5. Multimodal fusion이 단순 concatenation 기반 **late fusion**입니다. Cross-attention 등 학습형 fusion은 사용하지 않았습니다.
6. Sex / Age / AP-PA subgroup 분석은 모두 **exploratory**이며 다중비교 보정을 적용하지 않았습니다.
7. Text logit contribution은 **문장 전체 embedding의 기여**이며 개별 단어의 인과적 영향이 아닙니다.

## Future work

- External dataset validation (MIMIC-CXR 등)
- Learned multimodal fusion / cross-attention
- 다른 chest finding (Pneumonia, Edema 등)으로 확장
- 특정 clinical phrase ablation (예: CHF 제거 전후 probability 비교)
- Dataset shortcut / spurious correlation 정량 분석

---

## Disclaimer

연구·교육 목적의 프로젝트이며 임상 진단에 사용할 수 없습니다.
