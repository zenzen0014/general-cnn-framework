import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DataBundle:
    """Kumpulan output data yang dibutuhkan proses training dan evaluasi."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    dataset_dir: Path
    total_images: int
    train_size: int
    val_size: int
    test_size: int


def download_kaggle_dataset(config) -> Path:
    """
    Mengunduh dataset dari Kaggle.

    Fungsi ini hanya boleh dipanggil jika:
    - SUMBER_DATASET=kaggle
    - ALLOW_KAGGLE_DOWNLOAD=true

    Pada mode folder_lokal, fungsi ini tidak pernah dipanggil.
    """
    if config.sumber_dataset != "kaggle":
        raise RuntimeError(
            "download_kaggle_dataset() terpanggil padahal SUMBER_DATASET bukan kaggle."
        )

    if not config.allow_kaggle_download:
        raise RuntimeError(
            "Download Kaggle diblokir. Set ALLOW_KAGGLE_DOWNLOAD=true jika diperlukan."
        )

    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "Package kagglehub belum terinstall. Jalankan: pip install kagglehub"
        ) from error

    config.resolved_kaggle_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(config.resolved_kaggle_cache_dir)

    print(f"Mengunduh dataset Kaggle: {config.kaggle_dataset}")
    print(f"Cache KaggleHub        : {config.resolved_kaggle_cache_dir}")

    dataset_path = kagglehub.dataset_download(config.kaggle_dataset)
    return Path(dataset_path)


def folder_contains_images(folder: Path) -> bool:
    """Mengecek apakah folder berisi file gambar langsung."""
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder.iterdir()
    )


def find_class_folder(base_dir: str | Path) -> Path:
    """
    Mencari folder yang berisi sub-folder kelas.

    Contoh struktur ideal:
        dataset/banana_ripeness/
        ├── unripe/
        ├── early_ripe/
        ├── ripe/
        └── overripe/

    Jika dataset Kaggle punya nested folder, fungsi ini akan mencari folder
    pertama yang sub-foldernya berisi gambar.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {base_dir}")

    # Jika base_dir sendiri sudah berisi sub-folder kelas, pakai langsung.
    direct_class_dirs = [
        child for child in base_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".") and folder_contains_images(child)
    ]
    if len(direct_class_dirs) >= 2:
        return base_dir

    # Jika tidak, cari di nested folder.
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        class_dirs = []

        for dirname in sorted(dirs):
            class_path = root_path / dirname
            if folder_contains_images(class_path):
                class_dirs.append(dirname)

        root_has_images = any(
            Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in files
        )

        if len(class_dirs) >= 2 and not root_has_images:
            return root_path

    raise ValueError(
        "Tidak ditemukan struktur dataset klasifikasi gambar.\n"
        f"Folder yang diperiksa: {base_dir}\n"
        "Struktur yang diharapkan: satu sub-folder untuk setiap kelas.\n"
        "Contoh: ./dataset/banana_ripeness/ripe/*.jpg"
    )


class ImageFolderDataset(Dataset):
    """
    Dataset universal untuk klasifikasi gambar.

    Aturan:
    - Satu sub-folder = satu kelas.
    - Nama sub-folder = nama kelas.
    """

    def __init__(
        self,
        root_dir: str | Path,
        max_per_class: Optional[int] = None,
        transform=None,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        self.class_names = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith(".")
        )

        if not self.class_names:
            raise ValueError(
                f"Tidak ditemukan sub-folder kelas pada: {self.root_dir}"
            )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        rng = random.Random(seed)

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            image_paths = [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            rng.shuffle(image_paths)

            if max_per_class is not None:
                image_paths = image_paths[:max_per_class]

            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise ValueError(f"Tidak ditemukan file gambar pada: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DatasetWithTransform(Dataset):
    """Wrapper agar train, validation, dan test memakai transform berbeda."""

    def __init__(self, subset: torch.utils.data.Subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_transforms(image_size: int, augment_train: bool = True):
    """Membuat transform untuk train dan validasi/test."""
    train_transform_steps = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment_train:
        train_transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                ),
            ]
        )

    train_transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_transform = transforms.Compose(train_transform_steps)

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def get_dataset_dir(config) -> Path:
    """Menentukan folder dataset berdasarkan config."""
    if config.sumber_dataset == "folder_lokal":
        base_dir = config.dataset_path
        print("Mode dataset        : folder_lokal")
        print(f"Dataset aktif       : {config.dataset_name}")
        print(f"Path dataset lokal  : {base_dir}")
    elif config.sumber_dataset == "kaggle":
        base_dir = download_kaggle_dataset(config)
        print("Mode dataset        : kaggle")
        print(f"Path dataset Kaggle : {base_dir}")
    else:
        raise ValueError("SUMBER_DATASET harus 'folder_lokal' atau 'kaggle'.")

    class_folder = find_class_folder(base_dir)
    print(f"Folder kelas terbaca: {class_folder}")
    return class_folder


def prepare_data(config) -> DataBundle:
    """Menyiapkan dataset, split, transform, dan DataLoader."""
    dataset_dir = get_dataset_dir(config)

    full_dataset = ImageFolderDataset(
        root_dir=dataset_dir,
        max_per_class=config.max_per_class,
        transform=None,
        seed=config.seed,
    )

    total_images = len(full_dataset)
    train_size = int(total_images * config.train_ratio)
    val_size = int(total_images * config.val_ratio)
    test_size = total_images - train_size - val_size

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "Jumlah data terlalu kecil untuk split train/val/test. "
            "Kurangi rasio split atau tambahkan gambar."
        )

    generator = torch.Generator().manual_seed(config.seed)
    train_raw, val_raw, test_raw = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_transform, eval_transform = build_transforms(
        image_size=config.image_size,
        augment_train=config.augment_train,
    )

    train_set = DatasetWithTransform(train_raw, train_transform)
    val_set = DatasetWithTransform(val_raw, eval_transform)
    test_set = DatasetWithTransform(test_raw, eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=full_dataset.class_names,
        dataset_dir=dataset_dir,
        total_images=total_images,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
