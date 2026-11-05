Sumber data: **synthetic**  |  commit `3949a29`

| Pipeline | Deteksi ROI (uji) | Dice (uji) | Latensi ROI (ms) | Cakupan klasifikasi | Akurasi | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Mask ahli (batas atas) | - | - | - | 100% | 0.833 | 0.667 | 1.000 | 0.800 | 0.944 |
| Pipeline 1: YOLOv8n + U-Net | 100% | 0.981 | 74.85 | 100% | 0.750 | 0.667 | 0.833 | 0.727 | 0.889 |
| Pipeline 2: MediaPipe murni | 0% | 0.000 | 5.53 | 0% | - | - | - | - | - |

Uji McNemar P1 vs P2: n berpasangan = 0 (tidak ada citra uji yang diproses kedua pipeline)
