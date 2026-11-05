"""Dua cara mendapatkan mask konjungtiva yang dibandingkan.

Pipeline 1: YOLOv8n (kotak) -> crop + padding -> U-Net (mask piksel) -> mask seukuran citra.
Pipeline 2: MediaPipe Face Landmarker -> poligon kelopak bawah -> mask seukuran citra.
Keduanya mengembalikan (mask uint8 {0,255}, info) dengan info['detected'] dan info['latency_ms'].
Tidak ada jalur cadangan yang memakai mask ground truth: kegagalan dicatat apa adanya.
"""
import time
from typing import Dict, Tuple

import numpy as np

from src.utils import ROOT, apply_clahe, pad_box


class Pipeline1YoloUNet:
    name = "p1_yolo_unet"

    def __init__(self, cfg: dict, device: str):
        from src.segmentation.unet_segmenter import UNetConjunctivaSegmenter
        from src.segmentation.yolo_detector import YOLOConjunctivaDetector
        y, u = cfg["yolo_detector"], cfg["unet_segmenter"]
        self.detector = YOLOConjunctivaDetector(str(ROOT / y["weights_out"]), y["confidence_threshold"],
                                                y["iou_threshold"], device)
        self.segmenter = UNetConjunctivaSegmenter(str(ROOT / u["weights_out"]), u["encoder_name"], u["input_size"],
                                                  u["mask_threshold"], device)
        self.pad = y["crop_padding"]
        self.clahe = cfg["preprocessing"]["clahe_for_segmentation"]
        dummy = np.zeros((480, 640, 3), np.uint8)  # pemanasan agar latensi citra pertama tidak bias
        self.detector.detect(dummy)
        self.segmenter.predict_mask(dummy[:256, :256])

    def __call__(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        x = apply_clahe(image_bgr) if self.clahe else image_bgr
        mask = np.zeros((h, w), np.uint8)
        box, dinfo = self.detector.detect(x)
        if box is None:
            return mask, {"detected": False, "latency_ms": (time.perf_counter() - t0) * 1000, "stage_failed": "yolo"}
        x1, y1, x2, y2 = pad_box(box, self.pad, w, h)
        crop_mask, _ = self.segmenter.predict_mask(x[y1:y2, x1:x2])
        mask[y1:y2, x1:x2] = crop_mask
        ok = bool(mask.any())
        return mask, {"detected": ok, "latency_ms": (time.perf_counter() - t0) * 1000,
                      "yolo_conf": dinfo["confidence"], "stage_failed": None if ok else "unet"}


class Pipeline2MediaPipe:
    name = "p2_mediapipe"

    def __init__(self, cfg: dict):
        from src.segmentation.mediapipe_extractor import MediaPipeConjunctivaExtractor
        m = cfg["mediapipe"]
        self.extractor = MediaPipeConjunctivaExtractor(str(ROOT / m["model_asset_path"]), m["num_faces"],
                                                       m["expansion_factor"])
        self.extractor._get_detector()  # muat model di awal agar latensi tidak memuat waktu inisialisasi

    def __call__(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        mask, meta = self.extractor.extract_mask(image_bgr)
        ok = bool(meta.get("detected")) and bool(mask.any())
        return mask, {"detected": ok, "latency_ms": (time.perf_counter() - t0) * 1000,
                      "stage_failed": None if ok else "face_landmarker"}
