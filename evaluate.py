from pathlib import Path

from config import CONFIG
from src.data import prepare_data
from src.evaluation import (
    calculate_accuracy,
    collect_predictions,
    load_checkpoint,
    make_classification_report,
    make_confusion_matrix,
)
from src.models import build_model
from src.utils import get_device, print_section, set_seed
from src.visualization import plot_confusion_matrix


def main() -> None:
    CONFIG.validate()
    set_seed(CONFIG.seed)

    output_dir = Path(CONFIG.output_dir)
    checkpoint_path = output_dir / CONFIG.model_filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model belum ditemukan: {checkpoint_path}. Jalankan train.py terlebih dahulu."
        )

    device = get_device(CONFIG.device)

    print_section("1. Menyiapkan Test Set")
    data = prepare_data(CONFIG)

    print(f"Folder dataset : {data.dataset_dir}")
    print(f"Nama kelas     : {data.class_names}")
    print(f"Jumlah test    : {data.test_size}")

    print_section("2. Memuat Model Terbaik")
    model = build_model(
        model_name=CONFIG.model_name,
        num_classes=len(data.class_names),
        dropout=CONFIG.dropout,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    checkpoint = load_checkpoint(model, checkpoint_path, device)
    print(f"Model dimuat dari       : {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")

    print_section("3. Evaluasi Test Set")
    labels, predictions = collect_predictions(model, data.test_loader, device)

    test_accuracy = calculate_accuracy(labels, predictions)
    report = make_classification_report(labels, predictions, data.class_names)
    cm = make_confusion_matrix(labels, predictions)

    print(f"Test accuracy: {test_accuracy:.2f}%\n")
    print(report)

    report_path = output_dir / "test_report.txt"
    cm_path = output_dir / "confusion_matrix_test.png"

    report_path.write_text(
        f"Test Accuracy: {test_accuracy:.2f}%\n\n{report}",
        encoding="utf-8",
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=data.class_names,
        title="Confusion Matrix - Test Set",
        output_path=cm_path,
    )

    print_section("4. Output Evaluasi")
    print(f"Report          : {report_path}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
