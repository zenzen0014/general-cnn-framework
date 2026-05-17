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


def download_kaggle_dataset(dataset_name: str) -> Path:
    """
    Mengunduh dataset dari Kaggle menggunakan kagglehub.

    Fungsi ini hanya dipanggil jika config.sumber_dataset = "kaggle".
    """
    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "Package kagglehub belum terinstall. Jalankan: pip install kagglehub"
        ) from error

    print(f"Mengunduh dataset Kaggle: {dataset_name}")
    dataset_path = kagglehub.dataset_download(dataset_name)
    return Path(dataset_path)


def find_class_folder(base_dir: str | Path) -> Path:
    """
    Mencari folder yang berisi sub-folder kelas.

    Contoh struktur yang benar:
        dataset/
        ├── ripe/
        ├── unripe/
        └── overripe/
    """
    base_dir = Path(base_dir)

    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)

        class_dirs = []
        for dirname in sorted(dirs):
            class_path = root_path / dirname
            if any(
                file_path.suffix.lower() in IMAGE_EXTENSIONS
                for file_path in class_path.iterdir()
                if file_path.is_file()
            ):
                class_dirs.append(dirname)

        root_has_images = any(
            Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in files
        )

        if class_dirs and not root_has_images:
            return root_path

    return base_dir


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
    """
    Wrapper agar train, validation, dan test dapat memakai transform berbeda.
    """

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
    if config.sumber_dataset == "kaggle":
        base_dir = download_kaggle_dataset(config.kaggle_dataset)
    else:
        base_dir = Path(config.folder_lokal)

    return find_class_folder(base_dir)


def prepare_data(config) -> DataBundle:
    """
    Menyiapkan dataset, split, transform, dan DataLoader.

    Fungsi ini menjadi pintu utama untuk seluruh proses data.
    """
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
