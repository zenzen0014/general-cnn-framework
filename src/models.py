from typing import Iterable

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN sederhana untuk klasifikasi gambar.

    Mahasiswa dapat mengubah arsitektur di class ini.
    Bagian akhir model tetap harus menghasilkan output sebanyak jumlah kelas.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(2),

            self._conv_block(32, 64),
            nn.MaxPool2d(2),

            self._conv_block(64, 128),
            nn.MaxPool2d(2),

            self._conv_block(128, 256),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Blok dasar: Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def freeze_parameters(parameters: Iterable) -> None:
    """Membekukan parameter backbone pada pretrained model."""
    for parameter in parameters:
        parameter.requires_grad = False


def build_pretrained_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """
    Membuat model pretrained dari torchvision.

    Classifier terakhir otomatis diganti sesuai jumlah kelas dataset.
    """
    from torchvision import models

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        if freeze_backbone:
            freeze_parameters(model.parameters())

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            freeze_parameters(model.parameters())

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            freeze_parameters(model.parameters())

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Pretrained model '{model_name}' belum tersedia.")


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float = 0.5,
    pretrained: bool = False,
    freeze_backbone: bool = False,
):
    """
    Factory function untuk membuat model.

    Untuk menambah model sendiri:
    1. Buat class model baru di file ini.
    2. Tambahkan pilihan model_name pada fungsi ini.
    3. Ubah model_name di config.py.
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    return build_pretrained_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
