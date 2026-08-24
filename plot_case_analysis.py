from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ==========================================
# 설정
# ==========================================

DATA_PATH = "data/discordant_with_contributions.csv"
IMAGE_DIR = Path("images")

OUTPUT_PATH = "data/case_analysis_6panel_refined.png"

# Validation에서 이미 결정한 threshold
B_THRESHOLD = 0.5197
D_THRESHOLD = 0.5164


# ==========================================
# 대표 Case 6개
# 환자 ID가 아니라 정확한 image path로 선택
# ==========================================

selected_cases = [

    # --------------------------------------
    # Prediction corrected after text
    # --------------------------------------

    {
        "case": "H1",
        "group": "corrected",
        "path": "train/patient54238/study1/view1_frontal.jpg",
        "display_context": "CHF and CAD; pre-op study."
    },

    {
        "case": "H2",
        "group": "corrected",
        "path": "train/patient00211/study2/view1_frontal.jpg",
        "display_context": "R/o CHF; rapid A-fib."
    },

    {
        "case": "H3",
        "group": "corrected",
        "path": "train/patient20565/study2/view1_frontal.jpg",
        "display_context": (
            "Status post assault; possible intracranial hemorrhage."
        )
    },


    # --------------------------------------
    # Prediction worsened after text
    # --------------------------------------

    {
        "case": "F1",
        "group": "worsened",
        "path": "train/patient32357/study1/view1_frontal.jpg",
        "display_context": "History of heart failure."
    },

    {
        "case": "F2",
        "group": "worsened",
        "path": "train/patient00618/study2/view1_frontal.jpg",
        "display_context": (
            "Post esophagectomy/pharyngectomy; "
            "evaluation for possible pneumothorax "
            "and tube positioning."
        )
    },

    {
        "case": "F3",
        "group": "worsened",
        "path": "train/patient29497/study1/view1_frontal.jpg",
        "display_context": "Status epilepticus."
    }
]


# ==========================================
# 데이터
# ==========================================

df = pd.read_csv(
    DATA_PATH
)


# ==========================================
# Figure
# ==========================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 14)
)

axes = axes.flatten()


# ==========================================
# Case별 그림 생성
# ==========================================

for ax, case_info in zip(
    axes,
    selected_cases
):

    path = case_info["path"]

    row = df[
        df["path_to_image"] == path
    ]

    if len(row) != 1:

        raise ValueError(
            f"{path} 결과가 "
            f"{len(row)}개 검색되었습니다."
        )

    row = row.iloc[0]


    # ======================================
    # 실제 PNG 경로
    # ======================================

    png_relative = (
        path
        .replace("train/", "")
        .replace("valid/", "")
        .replace(".jpg", ".png")
    )

    image_path = (
        IMAGE_DIR
        / png_relative
    )

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )


    # ======================================
    # X-ray 출력
    # ======================================

    image = Image.open(
        image_path
    ).convert("L")

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.axis("off")


    # ======================================
    # 데이터
    # ======================================

    label = int(
        row["label"]
    )

    b_prob = float(
        row["b_probability"]
    )

    d_prob = float(
        row["d_probability"]
    )

    text_logit = float(
        row["text_contribution"]
    )


    reference_label = (
        "Positive"
        if label == 1
        else "Negative"
    )


    # ======================================
    # threshold 기준 prediction
    # ======================================

    b_pred = (
        1
        if b_prob >= B_THRESHOLD
        else 0
    )

    d_pred = (
        1
        if d_prob >= D_THRESHOLD
        else 0
    )


    b_correct = (
        b_pred == label
    )

    d_correct = (
        d_pred == label
    )


    # ======================================
    # Correct / Incorrect 표시
    # ======================================

    b_status = (
        "Correct"
        if b_correct
        else "Incorrect"
    )

    d_status = (
        "Correct"
        if d_correct
        else "Incorrect"
    )


    # ======================================
    # Clinical context
    # ======================================

    clinical_text = textwrap.fill(
        case_info["display_context"],
        width=45
    )


    # ======================================
    # Case 제목
    # ======================================

    ax.set_title(

        f"{case_info['case']}\n"

        f"Reference label: "
        f"{reference_label}\n"

        f"Image-only: "
        f"{b_prob:.3f} "
        f"({b_status})\n"

        f"Image + Text: "
        f"{d_prob:.3f} "
        f"({d_status})\n"

        f"Text logit contribution: "
        f"{text_logit:+.3f}",

        fontsize=11,
        pad=10
    )


    # ======================================
    # Clinical context 아래 출력
    # ======================================

    ax.text(

        0.5,
        -0.07,

        "Clinical context:\n"
        + clinical_text,

        transform=ax.transAxes,

        ha="center",
        va="top",

        fontsize=9,

        wrap=True
    )


# ==========================================
# 전체 제목
# ==========================================

fig.suptitle(

    "Representative Discordant Cases: "
    "Impact of Clinical Context",

    fontsize=18,

    y=0.98
)


# ==========================================
# Row 제목
# ==========================================

fig.text(

    0.015,
    0.71,

    "Prediction corrected\n"
    "after adding text",

    rotation=90,

    fontsize=14,

    ha="center",
    va="center"
)


fig.text(

    0.015,
    0.27,

    "Prediction worsened\n"
    "after adding text",

    rotation=90,

    fontsize=14,

    ha="center",
    va="center"
)


# ==========================================
# 하단 설명
# ==========================================

footnote = (

    "Correct/incorrect classification was determined using "
    "validation-derived Youden thresholds "
    f"(Image-only={B_THRESHOLD:.4f}, "
    f"Image+Text={D_THRESHOLD:.4f}).\n"

    "Text logit contribution represents the contribution of "
    "the BiomedCLIP text embedding to the multimodal "
    "logistic-regression logit."
)


fig.text(

    0.5,
    0.015,

    footnote,

    ha="center",
    va="bottom",

    fontsize=9
)


# ==========================================
# Layout
# ==========================================

plt.subplots_adjust(

    left=0.07,
    right=0.98,

    top=0.90,
    bottom=0.10,

    wspace=0.10,
    hspace=0.38
)


# ==========================================
# 저장
# ==========================================

plt.savefig(

    OUTPUT_PATH,

    dpi=300,

    bbox_inches="tight"
)

plt.close()


print(
    "===== Refined Case Analysis 저장 완료 ====="
)

print(
    "파일:",
    OUTPUT_PATH
)