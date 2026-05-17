# CNN Framework Sederhana - Multi Dataset Lokal

Framework ini dibuat agar mahasiswa cukup mengganti konfigurasi di `.env` tanpa mengubah kode Python.

## Struktur dataset yang disarankan

Simpan semua dataset di dalam folder `dataset/`:

```text
project/
├── dataset/
│   ├── banana_ripeness/
│   │   ├── unripe/
│   │   ├── early_ripe/
│   │   ├── ripe/
│   │   └── overripe/
│   ├── cxr/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── retinal_oct/
│       ├── class_1/
│       └── class_2/
├── config.py
├── train.py
├── evaluate.py
└── .env
```

Aturan utama:

```text
1 sub-folder = 1 kelas
nama sub-folder = nama kelas
```

## Memilih dataset lokal

Ubah `.env`:

```env
SUMBER_DATASET=folder_lokal
DATASET_ROOT=./dataset
DATASET_NAME=banana_ripeness
FOLDER_LOKAL=
ALLOW_KAGGLE_DOWNLOAD=false
```

Jika ingin memakai dataset CXR:

```env
SUMBER_DATASET=folder_lokal
DATASET_ROOT=./dataset
DATASET_NAME=cxr
FOLDER_LOKAL=
ALLOW_KAGGLE_DOWNLOAD=false
```

Dengan konfigurasi ini, framework akan membaca:

```text
./dataset/cxr
```

Framework tidak akan download dari Kaggle ketika `SUMBER_DATASET=folder_lokal`.

## Jika ingin memakai path manual

Gunakan `FOLDER_LOKAL`:

```env
SUMBER_DATASET=folder_lokal
FOLDER_LOKAL=D:/dataset/banana_ripeness
```

Jika `FOLDER_LOKAL` diisi, maka `DATASET_ROOT` dan `DATASET_NAME` akan diabaikan.

## Jika benar-benar ingin download Kaggle

Download Kaggle sengaja diblokir secara default agar tidak terjadi download tidak sengaja.

Untuk download Kaggle, ubah `.env`:

```env
SUMBER_DATASET=kaggle
ALLOW_KAGGLE_DOWNLOAD=true
KAGGLE_DATASET=barkataliarbab/banana-ripeness-classification-classification
KAGGLE_CACHE_DIR=./dataset/_kaggle_cache
```

## Menjalankan training

```bash
python train.py
```

## Menjalankan evaluasi

```bash
python evaluate.py
```

## Output

Output training dan evaluasi disimpan di folder:

```text
outputs/
├── best_model.pth
├── training_history.json
├── training_curve.png
├── test_report.txt
└── confusion_matrix_test.png
```

## Catatan penting

Jika sudah memilih `SUMBER_DATASET=folder_lokal`, lalu program masih download Kaggle, biasanya penyebabnya:

1. `.env` tidak terbaca karena notebook dijalankan dari folder berbeda.
2. `SUMBER_DATASET` masih bernilai `kaggle` di environment lama.
3. Ada cell notebook yang masih menjalankan `kagglehub.dataset_download()` secara manual.

Versi ini membaca `.env` dari folder yang sama dengan `config.py`, sehingga lebih stabil untuk notebook maupun script.
