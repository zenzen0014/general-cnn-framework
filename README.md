# Framework CNN PyTorch Sederhana

Framework ini dibuat agar mahasiswa dapat fokus pada lima hal utama:

1. Mengganti folder dataset.
2. Mengatur distribusi dataset training, validation, dan testing.
3. Memilih model sendiri atau model pretrained.
4. Menjalankan training.
5. Mengevaluasi model dan menganalisis grafik performa.

Konfigurasi eksperimen sekarang dibaca dari file `.env`, sehingga mahasiswa tidak perlu mengubah kode Python di `config.py`.

Struktur dataset yang digunakan:

```text
dataset/
├── class_1/
│   ├── image_001.jpg
│   └── image_002.jpg
├── class_2/
│   ├── image_001.jpg
│   └── image_002.jpg
└── class_3/
    ├── image_001.jpg
    └── image_002.jpg
```

## Cara menggunakan

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Ubah konfigurasi eksperimen

Buka file:

```text
.env
```

Bagian yang paling sering diganti:

```env
SUMBER_DATASET=folder_lokal
FOLDER_LOKAL=D:/dataset/banana_ripeness

TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15

MODEL_NAME=simple_cnn
PRETRAINED=false
FREEZE_BACKBONE=false

EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.001
```

Jika file `.env` belum ada, salin dari template:

```bash
cp .env.example .env
```

Pada Windows, bisa copy manual file `.env.example`, lalu ubah namanya menjadi `.env`.

### 3. Jalankan training

```bash
python train.py
```

Output training akan tersimpan di folder:

```text
outputs/
├── best_model.pth
├── training_curve.png
└── training_history.json
```

### 4. Jalankan evaluasi test set

```bash
python evaluate.py
```

Output evaluasi akan tersimpan di:

```text
outputs/
├── confusion_matrix_test.png
└── test_report.txt
```

## Mengganti sumber dataset

### Menggunakan dataset Kaggle

```env
SUMBER_DATASET=kaggle
KAGGLE_DATASET=wiratrnn/banana-ripeness-image-dataset
```

### Menggunakan dataset lokal

```env
SUMBER_DATASET=folder_lokal
FOLDER_LOKAL=D:/dataset/banana_ripeness
```

Pastikan struktur folder dataset lokal mengikuti format berikut:

```text
banana_ripeness/
├── ripe/
├── unripe/
└── overripe/
```

Nama sub-folder otomatis menjadi nama kelas.

## Mengatur distribusi dataset

Contoh 70% training, 15% validation, dan 15% testing:

```env
TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15
```

Jumlah ketiganya harus sama dengan `1.0`.

## Mengganti model

Untuk model CNN buatan sendiri, edit class `SimpleCNN` di:

```text
src/models.py
```

Lalu gunakan konfigurasi berikut di `.env`:

```env
MODEL_NAME=simple_cnn
PRETRAINED=false
```

Untuk pretrained model, gunakan contoh berikut:

```env
MODEL_NAME=resnet18
PRETRAINED=true
FREEZE_BACKBONE=true
```

Model pretrained yang sudah disiapkan:

- `resnet18`
- `mobilenet_v3_small`
- `efficientnet_b0`

## Catatan penting

- `config.py` tetap menjadi pusat konfigurasi program, tetapi nilainya dibaca dari `.env`.
- `.env` cocok untuk mahasiswa karena parameter eksperimen terlihat jelas dan mudah diganti.
- Nama sub-folder otomatis menjadi nama kelas.
- Pembagian train/validation/test dilakukan otomatis berdasarkan rasio.
- Model terbaik disimpan berdasarkan akurasi validation tertinggi.
- Grafik loss dan akurasi dibuat otomatis setelah training selesai.
- Confusion matrix dibuat menggunakan `matplotlib`, sehingga tidak memerlukan `seaborn`.
