"""Generate README figures from the aggregated CSV files in ``results/``.

The script intentionally uses only redistributable summary statistics. It does
not read or export CheXpert Plus images, report text, or patient-level records.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

MODEL_COLORS = {
    "A": "#64748B",
    "B": "#2563EB",
    "C": "#F59E0B",
    "D": "#059669",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.alpha": 0.8,
            "grid.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    output_path = RESULTS_DIR / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_model_performance() -> None:
    results = pd.read_csv(RESULTS_DIR / "main_results.csv")
    model_ids = results["model"].tolist()
    labels = [
        "A\nEfficientNet\nimage",
        "B\nBiomedCLIP\nimage",
        "C\nClinical\ntext",
        "D\nImage + text",
    ]
    colors = [MODEL_COLORS[model_id] for model_id in model_ids]
    x = np.arange(len(results))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    width = 0.34
    axes[0].bar(
        x - width / 2,
        results["test_auroc"],
        width,
        label="AUROC",
        color=colors,
        alpha=0.98,
    )
    axes[0].bar(
        x + width / 2,
        results["test_auprc"],
        width,
        label="AUPRC",
        color=colors,
        alpha=0.52,
        hatch="//",
    )
    axes[0].set_title("Discrimination performance")
    axes[0].set_ylabel("Score (higher is better)")
    axes[0].set_ylim(0.60, 0.93)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="x", visible=False)

    for index, row in results.iterrows():
        axes[0].text(
            index - width / 2,
            row["test_auroc"] + 0.006,
            f'{row["test_auroc"]:.4f}',
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0F172A",
        )
        axes[0].text(
            index + width / 2,
            row["test_auprc"] + 0.006,
            f'{row["test_auprc"]:.4f}',
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0F172A",
        )

    axes[1].bar(x, results["brier"], color=colors, width=0.62)
    axes[1].set_title("Probability accuracy")
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_ylim(0.12, 0.24)
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="x", visible=False)

    for index, score in enumerate(results["brier"]):
        axes[1].text(
            index,
            score + 0.003,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0F172A",
        )

    fig.suptitle(
        "Test-set performance across four controlled experiments",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "B and D are nearly identical; the larger gain comes from the biomedical image representation (A → B).",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout()
    save_figure(fig, "model_performance_overview.png")


def plot_bootstrap_effects() -> None:
    summary = pd.read_csv(RESULTS_DIR / "bootstrap_summary.csv")
    auc_rows = summary[summary["metric"] == "delta_auroc"].copy()
    brier_row = summary[summary["metric"] == "delta_brier"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    auc_labels = [
        "A → B\nBiomedical image representation",
        "B → D\nAdd clinical text",
    ]
    y = np.arange(len(auc_rows))[::-1]
    estimates = auc_rows["estimate"].to_numpy()
    lower = auc_rows["ci_lower"].to_numpy()
    upper = auc_rows["ci_upper"].to_numpy()
    xerr = np.vstack([estimates - lower, upper - estimates])

    axes[0].errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        markersize=8,
        color="#2563EB",
        ecolor="#64748B",
        elinewidth=2.2,
        capsize=5,
    )
    axes[0].axvline(0, color="#0F172A", linestyle="--", linewidth=1.2)
    axes[0].set_yticks(y, auc_labels)
    axes[0].set_xlabel("ΔAUROC (95% CI)")
    axes[0].set_title("Discrimination effect")
    axes[0].grid(axis="y", visible=False)
    axes[0].set_xlim(-0.01, 0.06)
    for yi, estimate, lo, hi in zip(y, estimates, lower, upper):
        axes[0].text(
            hi + 0.002,
            yi,
            f"{estimate:+.4f} [{lo:+.4f}, {hi:+.4f}]",
            va="center",
            fontsize=9.5,
            color="#334155",
        )

    estimate = float(brier_row["estimate"])
    lo = float(brier_row["ci_lower"])
    hi = float(brier_row["ci_upper"])
    axes[1].errorbar(
        estimate,
        0,
        xerr=[[estimate - lo], [hi - estimate]],
        fmt="o",
        markersize=8,
        color="#059669",
        ecolor="#64748B",
        elinewidth=2.2,
        capsize=5,
    )
    axes[1].axvline(0, color="#0F172A", linestyle="--", linewidth=1.2)
    axes[1].set_yticks([0], ["B → D\nAdd clinical text"])
    axes[1].set_xlabel("ΔBrier score (95% CI)")
    axes[1].set_title("Calibration effect")
    axes[1].set_xlim(-0.006, 0.006)
    axes[1].grid(axis="y", visible=False)
    axes[1].text(
        hi + 0.00035,
        0,
        f"{estimate:+.4f} [{lo:+.4f}, {hi:+.4f}]",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    fig.suptitle(
        "Patient-level paired bootstrap (5,000 iterations)",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.04,
        "Intervals crossing zero do not establish an incremental benefit in this test cohort.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout()
    save_figure(fig, "bootstrap_effects.png")


def plot_subgroup_forest() -> None:
    projection = pd.read_csv(RESULTS_DIR / "subgroup_ap_pa_results.csv")
    demographics = pd.read_csv(RESULTS_DIR / "subgroup_sex_age_results.csv")

    projection_rows = projection.assign(
        label=projection["subgroup"].map({"AP": "Projection: AP", "PA": "Projection: PA"})
    )
    demographic_rows = demographics.assign(
        label=demographics.apply(
            lambda row: f'{row["variable"]}: {row["subgroup"]}', axis=1
        )
    )
    combined = pd.concat(
        [
            projection_rows[
                ["label", "images", "delta_auc", "delta_ci_lower", "delta_ci_upper"]
            ],
            demographic_rows[
                ["label", "images", "delta_auc", "delta_ci_lower", "delta_ci_upper"]
            ],
        ],
        ignore_index=True,
    )

    y = np.arange(len(combined))[::-1]
    estimates = combined["delta_auc"].to_numpy()
    lower = combined["delta_ci_lower"].to_numpy()
    upper = combined["delta_ci_upper"].to_numpy()
    xerr = np.vstack([estimates - lower, upper - estimates])

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7,
        color="#7C3AED",
        ecolor="#64748B",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0, color="#0F172A", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, combined["label"])
    ax.set_xlabel("ΔAUROC: image + text minus image-only (95% CI)")
    ax.set_title("Exploratory subgroup analysis")
    ax.set_xlim(-0.016, 0.021)
    ax.grid(axis="y", visible=False)

    for yi, row in zip(y, combined.itertuples(index=False)):
        ax.text(
            0.0205,
            yi,
            f"n={row.images:,}",
            ha="right",
            va="center",
            fontsize=9.5,
            color="#475569",
        )

    fig.text(
        0.5,
        0.01,
        "All 95% confidence intervals include zero; signals are exploratory and unadjusted for multiple comparisons.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "subgroup_delta_auroc.png")


def plot_discordant_cases() -> None:
    summary = pd.read_csv(RESULTS_DIR / "discordant_summary.csv")
    label_map = {
        "both_correct": "Both correct",
        "both_wrong": "Both wrong",
        "corrected_after_text": "Corrected after text",
        "worsened_after_text": "Worsened after text",
    }
    color_map = {
        "both_correct": "#2563EB",
        "both_wrong": "#94A3B8",
        "corrected_after_text": "#059669",
        "worsened_after_text": "#DC2626",
    }
    summary["label"] = summary["category"].map(label_map)
    summary["color"] = summary["category"].map(color_map)
    summary["share"] = summary["n_images"] / summary["n_images"].sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    wedges, _ = axes[0].pie(
        summary["n_images"],
        colors=summary["color"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )
    axes[0].text(
        0,
        0.05,
        f'{summary["n_images"].sum():,}',
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#0F172A",
    )
    axes[0].text(
        0,
        -0.14,
        "test images",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#64748B",
    )
    axes[0].set_title("Agreement after adding text")
    axes[0].legend(
        wedges,
        [
            f'{row.label}: {row.n_images:,} ({row.share:.1f}%)'
            for row in summary.itertuples(index=False)
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        fontsize=9.5,
    )

    changed = summary[
        summary["category"].isin(["corrected_after_text", "worsened_after_text"])
    ].copy()
    bars = axes[1].bar(
        changed["label"],
        changed["n_images"],
        color=changed["color"],
        width=0.62,
    )
    axes[1].set_title("Direction of changed decisions")
    axes[1].set_ylabel("Number of test images")
    axes[1].set_ylim(0, 90)
    axes[1].grid(axis="x", visible=False)
    axes[1].bar_label(bars, padding=4, fontsize=12, fontweight="bold")
    axes[1].text(
        0.5,
        0.92,
        "75 corrected vs 62 worsened",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=11,
        color="#334155",
    )

    fig.suptitle(
        "Discordant-case analysis at validation-selected thresholds",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "discordant_cases_overview.png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    set_style()
    plot_model_performance()
    plot_bootstrap_effects()
    plot_subgroup_forest()
    plot_discordant_cases()


if __name__ == "__main__":
    main()
