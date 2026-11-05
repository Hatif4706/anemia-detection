Sumber data: **synthetic**  |  commit `2505f7d`

| Pipeline | Deteksi ROI (uji) | Dice (uji) | Latensi ROI (ms) | Cakupan klasifikasi | Akurasi | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Mask ahli (batas atas) | - | - | - | 100% | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pipeline 1: YOLOv8n + U-Net | 100% | 0.981 | 90.0 | 100% | 0.667 | 0.500 | 0.833 | 0.600 | 0.833 |
| Pipeline 2: MediaPipe Face Mesh | 0% | 0.000 | 3.52 | 0% | - | - | - | - | - |
| Pipeline 2b: MediaPipe MagicTouch | 100% | 0.995 | 105.85 | 100% | 0.917 | 0.833 | 1.000 | 0.909 | 1.000 |

Uji McNemar P1 vs P2: n berpasangan = 0 (tidak ada citra uji yang diproses kedua pipeline)

Uji McNemar P1 vs P2b: n berpasangan = 12, p = 0.2500
