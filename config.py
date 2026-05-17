from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


# ============================================================
# LOAD .env DENGAN LOKASI YANG STABIL
# ============================================================
# load_dotenv() biasa membaca .env dari current working directory.
# Jika notebook dijalankan dari folder berbeda, .env bisa tidak terbaca.
# Karena itu, .env dibaca dari folder yang sama dengan config.py.
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_str(key: str, default: str) -> str:
    """Membaca nilai string dari .env."""
    return os.getenv(key, default).strip()


def get_optional_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Membaca nilai string opsional dari .env."""
    value = os.getenv(key)
    if value is None or value.strip() == "" or value.lower() in {"none", "null"}:
        return default
    return value.strip()


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
    """Membaca nilai boolean dari .env."""
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


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Mengubah path relatif menjadi path absolut berbasis folder project.

    Contoh:
    - ./dataset/banana_ripeness
      akan dibaca sebagai <project>/dataset/banana_ripeness
    - D:/dataset/banana_ripeness
      tetap dibaca sebagai path absolut.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


@dataclass
class Config:
    """
    Pusat konfigurasi project.

    Mahasiswa cukup mengubah file .env untuk:
    - memilih dataset lokal,
    - mengatur split train/val/test,
    - memilih model custom CNN atau pretrained,
    - mengatur parameter training.
    """

    # ============================================================
    # 1. DATASET
    # ============================================================
    # Pilihan:
    # - folder_lokal : memakai dataset yang sudah ada di komputer
    # - kaggle       : download dataset dari Kaggle
    sumber_dataset: str = get_str("SUMBER_DATASET", "folder_lokal").lower()

    # Root folder yang menampung banyak dataset.
    # Contoh struktur:
    # dataset/
    # ├── banana_ripeness/
    # ├── cxr/
    # └── retinal_oct/
    dataset_root: str = get_str("DATASET_ROOT", "./dataset")

    # Nama dataset aktif di dalam DATASET_ROOT.
    # Jika DATASET_NAME=banana_ripeness, maka path dataset adalah:
    # ./dataset/banana_ripeness
    dataset_name: str = get_str("DATASET_NAME", "banana_ripeness")

    # Optional override.
    # Jika FOLDER_LOKAL diisi, path ini akan dipakai langsung.
    # Jika kosong, framework memakai DATASET_ROOT / DATASET_NAME.
    folder_lokal: Optional[str] = get_optional_str("FOLDER_LOKAL", None)

    # Digunakan hanya jika SUMBER_DATASET=kaggle.
    kaggle_dataset: str = get_str(
        "KAGGLE_DATASET",
        "barkataliarbab/banana-ripeness-classification-classification",
    )

    # Lokasi penyimpanan hasil Kaggle jika mode kaggle dipakai.
    # Tidak dipakai sama sekali pada mode folder_lokal.
    kaggle_cache_dir: str = get_str("KAGGLE_CACHE_DIR", "./dataset/_kaggle_cache")

    # Safety switch.
    # Jika false, framework tidak akan download Kaggle walaupun SUMBER_DATASET=kaggle.
    allow_kaggle_download: bool = get_bool("ALLOW_KAGGLE_DOWNLOAD", False)

    # Batasi jumlah gambar per kelas.
    # Gunakan kosong, None, atau null jika ingin memakai semua gambar.
    max_per_class: Optional[int] = get_optional_int("MAX_PER_CLASS", 400)

    # Rasio pembagian dataset. Total harus sama dengan 1.0.
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
    device: str = get_str("DEVICE", "auto")

    # ============================================================
    # 4. MODEL
    # ============================================================
    model_name: str = get_str("MODEL_NAME", "simple_cnn")
    pretrained: bool = get_bool("PRETRAINED", False)
    freeze_backbone: bool = get_bool("FREEZE_BACKBONE", False)
    dropout: float = get_float("DROPOUT", 0.5)

    # ============================================================
    # 5. OUTPUT
    # ============================================================
    output_dir: str = get_str("OUTPUT_DIR", "outputs")
    model_filename: str = get_str("MODEL_FILENAME", "best_model.pth")
    seed: int = get_int("SEED", 42)

    @property
    def dataset_path(self) -> Path:
        """
        Path dataset lokal yang aktif.

        Prioritas:
        1. FOLDER_LOKAL jika diisi.
        2. DATASET_ROOT / DATASET_NAME jika FOLDER_LOKAL kosong.
        """
        if self.folder_lokal:
            return resolve_project_path(self.folder_lokal)
        return resolve_project_path(Path(self.dataset_root) / self.dataset_name)

    @property
    def resolved_output_dir(self) -> Path:
        """Path output absolut berbasis folder project."""
        return resolve_project_path(self.output_dir)

    @property
    def resolved_kaggle_cache_dir(self) -> Path:
        """Path cache Kaggle absolut berbasis folder project."""
        return resolve_project_path(self.kaggle_cache_dir)

    def validate(self) -> None:
        """Memastikan konfigurasi tidak salah sebelum training dimulai."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "TRAIN_RATIO + VAL_RATIO + TEST_RATIO harus sama dengan 1.0"
            )

        if self.sumber_dataset not in {"folder_lokal", "kaggle"}:
            raise ValueError("SUMBER_DATASET harus 'folder_lokal' atau 'kaggle'.")

        if self.sumber_dataset == "folder_lokal":
            if not self.dataset_path.exists():
                raise FileNotFoundError(
                    "Dataset lokal tidak ditemukan.\n"
                    f"Path yang dibaca: {self.dataset_path}\n"
                    "Solusi: ubah DATASET_ROOT/DATASET_NAME atau FOLDER_LOKAL di .env.\n"
                    "Catatan: mode folder_lokal tidak akan download dari Kaggle."
                )

        if self.sumber_dataset == "kaggle" and not self.allow_kaggle_download:
            raise RuntimeError(
                "SUMBER_DATASET=kaggle, tetapi ALLOW_KAGGLE_DOWNLOAD=false.\n"
                "Ubah SUMBER_DATASET=folder_lokal jika ingin memakai dataset lokal, "
                "atau set ALLOW_KAGGLE_DOWNLOAD=true jika memang ingin download Kaggle."
            )

        if self.model_name not in {
            "simple_cnn",
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
        }:
            raise ValueError(f"MODEL_NAME '{self.model_name}' belum tersedia.")


CONFIG = Config()
