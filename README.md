# Anemia Detection via Palpebral Conjunctiva Vision
### Controlled Fair Benchmark: YOLO + U-Net vs. MediaPipe Face Mesh

Sistem skrining dan deteksi anemia non-invasif berbasis citra konjungtiva palpebra mata (*palpebral conjunctiva*). Proyek ini mengimplementasikan komparasi setara (*controlled fair benchmark*) antara pendekatan Deep Learning Two-Stage (**YOLO + U-Net**) dan pendekatan Landmark-Based (**MediaPipe Face Mesh**) menggunakan partisi dataset identik dan downstream classifier medis yang sama.

---

## 📌 Arsitektur & Metodologi Fair Benchmark

```
                                  [ Input Citra Mata/Wajah ]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       [ Pipeline A: YOLO + U-Net ]                       [ Pipeline B: MediaPipe Face Mesh ]
     • Stage 1: YOLO Eyelid Detection                   • 468/478 Landmark Triangulation
     • Stage 2: U-Net Pixel Segmentation                • Lower Palpebral Sulcus Polygon
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                 [ Ekstraksi Fitur Identik ]
                      • Kolorimetri: Erythema Index, CIE Lab a*, HSV, RGB
                      • Tekstur: Haralick GLCM (Contrast, Homogeneity, dll.)
                                              │
                                              ▼
                                [ Downstream Classifier Identik ]
                               • XGBoost, SVM (RBF), Random Forest
                                              │
                                              ▼
                                [ Evaluasi 2 Tingkat (Level) ]
                      • Level 1: Segmentasi (Mean IoU, Dice Score, Latensi ms)
                      • Level 2: Diagnostik Medis (Sensitivity, Specificity, ROC-AUC)
```

---

## 📁 Struktur Direktori

```text
anemia-detection/
├── configs/
│   └── config.yaml           # Konfigurasi hyperparameter, paths, dan seed
├── data/
│   ├── raw/                  # Citra mentah dan mask ground truth
│   ├── processed/            # Citra hasil preprocessing / cropping
│   └── splits/               # Partisi terstandarisasi (train.csv, val.csv, test.csv)
├── docs/                     # Dokumen kajian pustaka dan proposal
├── src/
│   ├── preprocessing/        # DatasetLoader & stratified split generator
│   ├── segmentation/         # Modul YOLO, U-Net, dan MediaPipe Face Mesh
│   ├── features/             # Ekstraksi fitur Erythema Index, Lab a*, HSV
│   ├── classification/       # XGBoost, SVM RBF, Random Forest benchmark
│   └── evaluation/           # Penghitung metrik IoU, Dice, Sensitivity, ROC-AUC
├── scripts/
│   ├── run_benchmark.py      # Eksekutor benchmark komparatif end-to-end
│   └── download_dataset.py   # Panduan & penyiapan dataset publik
├── reports/                  # Hasil evaluasi benchmark dalam format JSON/CSV
└── requirements.txt          # Daftar pustaka Python yang dibutuhkan
```

---

## 🚀 Cara Menjalankan

### 1. Aktivasi Environment
```bash
# Menggunakan venv Python 3.11 lokal
.\.venv\Scripts\Activate.ps1
```

### 2. Menjalankan Evaluasi Benchmark
```bash
python scripts/run_benchmark.py
```
Output laporan akan otomatis menampilkan:
1. Perbandingan kualitas ROI (*Mean IoU*, *Dice Score*, *Latency ms*).
2. Perbandingan akurasi klinis (*Sensitivity / Recall*, *Specificity*, *Accuracy*, *ROC-AUC*).
3. Ringkasan laporan tersimpan di `reports/benchmark_summary.json`.
