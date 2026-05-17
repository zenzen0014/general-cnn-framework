from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# Global font setting
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def plot_history(history, output_path: str | Path) -> None:
    """
    Menyimpan grafik training loss dan accuracy
    dengan font Segoe UI dan tampilan lebih rapi.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================
    # Loss Curve
    # =========================
    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        linewidth=2,
        markersize=5,
        label="Train",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        linewidth=2,
        markersize=5,
        label="Validation",
    )

    axes[0].set_title(
        "Loss per Epoch",
        fontname="Segoe UI",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    axes[0].set_xlabel(
        "Epoch",
        fontname="Segoe UI",
        fontsize=12,
        color="black",
    )
    axes[0].set_ylabel(
        "Loss",
        fontname="Segoe UI",
        fontsize=12,
        color="black",
    )
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(prop={"family": "Segoe UI", "size": 10})

    # =========================
    # Accuracy Curve
    # =========================
    axes[1].plot(
        epochs,
        history["train_acc"],
        marker="o",
        linewidth=2,
        markersize=5,
        label="Train",
    )
    axes[1].plot(
        epochs,
        history["val_acc"],
        marker="o",
        linewidth=2,
        markersize=5,
        label="Validation",
    )

    axes[1].set_title(
        "Accuracy per Epoch",
        fontname="Segoe UI",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    axes[1].set_xlabel(
        "Epoch",
        fontname="Segoe UI",
        fontsize=12,
        color="black",
    )
    axes[1].set_ylabel(
        "Accuracy (%)",
        fontname="Segoe UI",
        fontsize=12,
        color="black",
    )
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(prop={"family": "Segoe UI", "size": 10})

    # Tick labels Segoe UI
    for ax in axes:
        ax.tick_params(axis="both", colors="black")
        for label in ax.get_xticklabels():
            label.set_fontname("Segoe UI")
            label.set_fontsize(10)
            label.set_color("black")

        for label in ax.get_yticklabels():
            label.set_fontname("Segoe UI")
            label.set_fontsize(10)
            label.set_color("black")

    fig.suptitle(
        "Training Evaluation Curve",
        fontname="Segoe UI",
        fontsize=16,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(
    cm,
    class_names,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Confusion matrix style:
    - diagonal hijau
    - selain diagonal putih
    - semua text hitam
    - tanpa colorbar
    - font Segoe UI
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    num_classes = len(class_names)

    fig_size = max(5.5, num_classes * 1.15)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    diagonal_color = "#B7D7A8"
    off_diagonal_color = "#FFFFFF"
    grid_color = "#BFBFBF"
    text_color = "#000000"

    for row in range(num_classes):
        for col in range(num_classes):
            cell_color = diagonal_color if row == col else off_diagonal_color

            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cell_color,
                    edgecolor=grid_color,
                    linewidth=1.0,
                )
            )

            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontname="Segoe UI",
                fontweight="bold" if row == col else "normal",
            )

    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect("equal")

    tick_positions = np.arange(num_classes)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    ax.set_xticklabels(
        class_names,
        rotation=0,
        ha="center",
        fontsize=12,
        fontname="Segoe UI",
        color="black",
    )
    ax.set_yticklabels(
        class_names,
        rotation=0,
        fontsize=12,
        fontname="Segoe UI",
        color="black",
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(
        title,
        fontsize=16,
        fontname="Segoe UI",
        fontweight="bold",
        pad=22,
        color="black",
    )
    ax.set_xlabel(
        "Predicted",
        fontsize=12,
        fontname="Segoe UI",
        fontweight="bold",
        labelpad=10,
        color="black",
    )
    ax.set_ylabel(
        "True",
        fontsize=12,
        fontname="Segoe UI",
        fontweight="bold",
        labelpad=10,
        color="black",
    )

    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        pad=7,
        colors="black",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)