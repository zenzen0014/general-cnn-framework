from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import os


# Membaca file .env secara otomatis jika file tersebut tersedia.
# Jika .env tidak ada, konfigurasi tetap memakai nilai default di bawah ini.
load_dotenv()


def get_str(key: str, default: str) -> str:
    """Membaca nilai string dari .env."""
    return os.getenv(key, default)


def get_optional_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Membaca nilai string opsional dari .env."""
    value = os.getenv(key)
    if value is None or value.strip() == "" or value.lower() in {"none", "null"}:
        return default
    return value


def get_int(key: str, default: int) -> int:
    """Membaca nilai integer dari .env."""
    return int(os.getenv(key, default))


def get_optional_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """Membaca nilai integer opsional dari .env."""
    value = os.getenv(key)
    if value is None or value.strip() == "" or value.lower() in {"none", "null"}:
        return default
    return int(value)


def get_float(key: str, default: float) -> float:
    """Membaca nilai float dari .env."""
    return float(os.getenv(key, default))


def get_bool(key: str, default: bool) -> bool:
    """Membaca nilai boolean dari .env.

    Nilai yang dianggap True:
    true, 1, yes, y

    Nilai yang dianggap False:
    false, 0, no, n
    """
    value = os.getenv(key)
    if value is None:
        return default

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Nilai boolean untuk {key} tidak valid: {value}. "
        "Gunakan true/false, 1/0, atau yes/no."
    )


@dataclass
class Config:
    """
    File ini adalah pusat konfigurasi.

    Nilai konfigurasi dibaca dari file .env agar mahasiswa cukup mengubah
    parameter eksperimen tanpa perlu mengubah kode Python.
    """

    # ============================================================
    # 1. DATASET
    # ============================================================
    # Pilihan:
    # - "kaggle"       : dataset diunduh otomatis dari Kaggle
    # - "folder_lokal" : dataset sudah ada di komputer/laptop
    sumber_dataset: str = get_str("SUMBER_DATASET", "kaggle")

    # Digunakan jika SUMBER_DATASET=kaggle
    kaggle_dataset: str = get_str(
        "KAGGLE_DATASET",
        "wiratrnn/banana-ripeness-image-dataset",
    )

    # Digunakan jika SUMBER_DATASET=folder_lokal
    # Contoh Windows : D:/dataset/banana_ripeness
    # Contoh Linux   : /home/user/dataset/banana_ripeness
    folder_lokal: Optional[str] = get_optional_str("FOLDER_LOKAL", None)

    # Batasi jumlah gambar per kelas.
    # Gunakan kosong, None, atau null jika ingin memakai semua gambar.
    max_per_class: Optional[int] = get_optional_int("MAX_PER_CLASS", 400)

    # Rasio pembagian dataset.
    # Total harus sama dengan 1.0
    train_ratio: float = get_float("TRAIN_RATIO", 0.70)
    val_ratio: float = get_float("VAL_RATIO", 0.15)
    test_ratio: float = get_float("TEST_RATIO", 0.15)

    # ============================================================
    # 2. PREPROCESSING GAMBAR
    # ============================================================
    image_size: int = get_int("IMAGE_SIZE", 224)
    augment_train: bool = get_bool("AUGMENT_TRAIN", True)

    # ============================================================
    # 3. TRAINING
    # ============================================================
    batch_size: int = get_int("BATCH_SIZE", 32)
    epochs: int = get_int("EPOCHS", 30)
    learning_rate: float = get_float("LEARNING_RATE", 0.001)
    weight_decay: float = get_float("WEIGHT_DECAY", 0.0)
    num_workers: int = get_int("NUM_WORKERS", 2)

    # Pilihan:
    # - "auto" : otomatis GPU jika tersedia, jika tidak CPU
    # - "cuda" : paksa GPU
    # - "cpu"  : paksa CPU
    device: str = get_str("DEVICE", "auto")

    # ============================================================
    # 4. MODEL
    # ============================================================
    # Pilihan:
    # - "simple_cnn"         : CNN sederhana buatan sendiri
    # - "resnet18"           : pretrained/transfer learning
    # - "mobilenet_v3_small" : pretrained/transfer learning
    # - "efficientnet_b0"    : pretrained/transfer learning
    model_name: str = get_str("MODEL_NAME", "simple_cnn")

    # True  : memakai bobot pretrained ImageNet
    # False : training dari awal
    pretrained: bool = get_bool("PRETRAINED", False)

    # True  : backbone pretrained dibekukan, hanya classifier dilatih
    # False : semua layer dilatih
    freeze_backbone: bool = get_bool("FREEZE_BACKBONE", False)

    dropout: float = get_float("DROPOUT", 0.5)

    # ============================================================
    # 5. OUTPUT
    # ============================================================
    output_dir: str = get_str("OUTPUT_DIR", "outputs")
    model_filename: str = get_str("MODEL_FILENAME", "best_model.pth")
    seed: int = get_int("SEED", 42)

    def validate(self) -> None:
        """Memastikan konfigurasi tidak salah sebelum training dimulai."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "TRAIN_RATIO + VAL_RATIO + TEST_RATIO harus sama dengan 1.0"
            )

        if self.sumber_dataset not in {"kaggle", "folder_lokal"}:
            raise ValueError("SUMBER_DATASET harus 'kaggle' atau 'folder_lokal'.")

        if self.sumber_dataset == "folder_lokal" and not self.folder_lokal:
            raise ValueError(
                "FOLDER_LOKAL belum diisi. Isi path dataset lokal di file .env."
            )

        if self.model_name not in {
            "simple_cnn",
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
        }:
            raise ValueError(f"MODEL_NAME '{self.model_name}' belum tersedia.")


CONFIG = Config()
