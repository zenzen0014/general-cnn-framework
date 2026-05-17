from pathlib import Path

import torch
import torch.nn as nn

from config import CONFIG
from src.data import prepare_data
from src.engine import train_model
from src.models import build_model
from src.utils import get_device, print_section, save_json, set_seed
from src.visualization import plot_history


def main() -> None:
    CONFIG.validate()
    set_seed(CONFIG.seed)

    output_dir = CONFIG.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(CONFIG.device)

    print_section("1. Menyiapkan Dataset")
    data = prepare_data(CONFIG)

    print(f"Folder dataset : {data.dataset_dir}")
    print(f"Nama kelas     : {data.class_names}")
    print(f"Total gambar   : {data.total_images}")
    print(
        f"Distribusi     : "
        f"train={data.train_size}, val={data.val_size}, test={data.test_size}"
    )

    print_section("2. Membuat Model")
    model = build_model(
        model_name=CONFIG.model_name,
        num_classes=len(data.class_names),
        dropout=CONFIG.dropout,
        pretrained=CONFIG.pretrained,
        freeze_backbone=CONFIG.freeze_backbone,
    ).to(device)

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model              : {CONFIG.model_name}")
    print(f"Pretrained         : {CONFIG.pretrained}")
    print(f"Device             : {device}")
    print(f"Total parameter    : {total_params:,}")
    print(f"Trainable parameter: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    print_section("3. Training")
    checkpoint_path = output_dir / CONFIG.model_filename

    history, best_val_accuracy = train_model(
        model=model,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=CONFIG.epochs,
        save_path=checkpoint_path,
        class_names=data.class_names,
        config=CONFIG,
    )

    history_path = output_dir / "training_history.json"
    curve_path = output_dir / "training_curve.png"

    save_json(history, history_path)
    plot_history(history, curve_path)

    print_section("4. Ringkasan")
    print(f"Best validation accuracy : {best_val_accuracy:.2f}%")
    print(f"Model terbaik            : {checkpoint_path}")
    print(f"History training         : {history_path}")
    print(f"Grafik training          : {curve_path}")


if __name__ == "__main__":
    main()
