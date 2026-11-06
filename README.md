# Anemia Detection via Palpebral Conjunctiva Vision
### Perbandingan Setara: YOLOv8n + U-Net vs. MediaPipe Murni

Sistem skrining anemia non-invasif berbasis citra konjungtiva palpebral. Dua cara menemukan area konjungtiva
dibandingkan dengan data, pembagian data, dan classifier (MobileNetV3-Small) yang sama, sehingga perbedaan hasil
hanya berasal dari kualitas ROI.

---

## Arsitektur

```
                              [ Citra RGB ]
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
  Pipeline 1: YOLOv8n + U-Net                Pipeline 2: MediaPipe murni
  • YOLOv8n mendeteksi area konjungtiva      • Face Landmarker (BlazeFace + Face Mesh)
  • crop + padding 10%, CLAHE                • 9 titik kelopak bawah per mata
  • U-Net (encoder ResNet34) → mask piksel   • diperluas ×1,15 + convex hull → mask
              │                                           │
              └──────────────► ROI = citra × mask ◄───────┘
                                    │
                    MobileNetV3-Small (setelan identik)
                                    │
           Evaluasi tingkat 1: tingkat deteksi, Dice, IoU, latensi
           Evaluasi tingkat 2: confusion matrix, sensitivity, specificity,
                               F1, ROC-AUC, CI bootstrap, uji McNemar P1 vs P2
```

Tidak ada jalur cadangan yang memakai mask ground truth. Bila YOLO atau MediaPipe gagal, citra dicatat sebagai
kegagalan deteksi.

---

## Data

Dataset utama: **Eyes-defy-anemia** (Dimauro et al., 2023; IEEE DataPort, DOI 10.21227/t5s2-4j73),
218 citra konjungtiva pasien Italia dan India dengan nilai Hb laboratorium dan mask konjungtiva buatan ahli.

1. Unduh dari IEEE DataPort (perlu login).
2. Ekstrak sehingga strukturnya menjadi:
   ```
   data/raw/eyes_defy_anemia/Italy/Italy.xlsx, Italy/1/..., Italy/123/
   data/raw/eyes_defy_anemia/India/India.xlsx, India/1/..., India/95/
   ```
3. Label: Anemia bila Hb < 12 g/dL (perempuan) atau < 13 g/dL (laki-laki), lihat `configs/config.yaml`.

Data **sintetis** (`--source synthetic`) hanya untuk menguji bahwa kode berjalan. Angkanya bukan hasil penelitian.

---

## Cara menjalankan

```powershell
.\.venv\Scripts\Activate.ps1

# Seluruh tahap dengan data penelitian
python scripts/run_pipeline.py

# Uji fungsi kode: data sintetis, tanpa bobot pretrained, epoch sedikit
python scripts/run_pipeline.py --source synthetic --smoke
```

### Menjalankan di Google Colab (GPU T4, hasil tersimpan di Google Drive)

1. Unggah `archive.zip` dataset ke `MyDrive/anemia-detection-colab/archive.zip`.
2. Buka [notebooks/train_colab.ipynb](https://colab.research.google.com/github/Hatif4706/anemia-detection/blob/dev/notebooks/train_colab.ipynb),
   pilih **Runtime → Change runtime type → T4 GPU**, lalu **Runtime → Run all**.
3. Hasil tersimpan per tahap di `MyDrive/anemia-detection-colab/hasil/<RUN_NAME>/` (models, reports, runs, log).
   Bila sesi terputus, set `START_STEP` ke tahap yang belum selesai lalu Run all lagi.

Lanjutkan dari tahap tertentu secara lokal: `python scripts/run_pipeline.py --start 3b` (opsional `--stop 4`).

| Tahap | Skrip | Keluaran utama |
|---|---|---|
| 1. Metadata + split per subjek | `prepare_data.py` | `data/raw/metadata.csv`, `data/splits/*.csv` |
| 2. Anotasi bbox otomatis dari mask | `make_yolo_labels.py` | `data/processed/yolo/`, `reports/figures/anotasi_bbox_dari_mask.png` |
| 3a. Pelatihan YOLOv8n | `train_yolo.py` | `models/yolo_conjunctiva.pt`, `reports/yolo_summary.json` |
| 3b. Pelatihan U-Net | `train_unet.py` | `models/unet_resnet34.pt`, `reports/unet_training_log.csv` |
| 4. ROI kedua pipeline + metrik segmentasi | `extract_rois.py` | `data/processed/rois/`, `reports/segmentation_summary.json` |
| 5. Classifier MobileNetV3-Small | `train_classifier.py` | `models/mobilenetv3_*.pt`, `reports/classifier_test_predictions.csv` |
| 6. Evaluasi + uji statistik | `evaluate.py` | `reports/results.json`, `reports/results_table.md` |
| 7. Laporan progres | `make_progress_report.py` | `reports/progress/laporan_progres.html` |

---

## Struktur direktori

```text
anemia-detection/
├── configs/config.yaml            # semua hiperparameter, batas Hb, path
├── scripts/                       # tahap 1–7 dan run_pipeline.py
├── src/
│   ├── preprocessing/             # pembaca Eyes-defy-anemia, label Hb, split per subjek, data sintetis
│   ├── segmentation/              # YOLOv8n, U-Net, MediaPipe
│   ├── pipelines/roi_pipelines.py # Pipeline 1 dan Pipeline 2
│   ├── evaluation/metrics.py      # IoU/Dice, metrik klinis, McNemar, bootstrap
│   ├── features/, classification/ # fitur warna + XGBoost/SVM (analisis tambahan, opsional)
│   └── utils.py
├── models/                        # bobot hasil pelatihan (tidak di-commit)
└── reports/                       # ringkasan JSON, gambar, laporan progres
```
