"""Pipeline 2b: MediaPipe tanpa pelatihan yang tetap bekerja pada citra close-up mata.

Langkah:
  1. Titik petunjuk (seed):
     a. Bila MediaPipe Face Landmarker menemukan wajah: titik di bawah tengah kelopak bawah
        (landmark 145 kiri / 374 kanan), digeser ke bawah sebesar tinggi celah mata.
     b. Bila tidak ada wajah (citra close-up): puncak kemerahan, yaitu nilai maksimum kanal a*
        (CIELAB) setelah dihaluskan. Konjungtiva yang dieversi adalah area paling merah pada citra.
  2. MediaPipe Interactive Segmenter (model MagicTouch) memisahkan objek di sekitar titik itu.
  3. Penyaringan warna: di dalam objek, hanya piksel yang lebih merah dari ambang Otsu kanal a*
     yang dipertahankan (membuang sklera putih dan iris gelap), lalu diambil komponen terhubung
     yang memuat/terdekat dengan titik petunjuk.
"""
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOWER_LID = {"left": (145, 159), "right": (374, 386)}  # (kelopak bawah, kelopak atas)


class MagicTouchConjunctivaExtractor:
    def __init__(self, magic_touch_path: str, face_landmarker_path: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mediapipe.tasks.python.vision import InteractiveSegmenterLegacy, InteractiveSegmenterLegacyOptions
        self.mp = mp
        self.segmenter = InteractiveSegmenterLegacy.create_from_options(InteractiveSegmenterLegacyOptions(
            base_options=BaseOptions(model_asset_path=magic_touch_path), output_confidence_masks=True,
            output_category_mask=False))
        self.landmarker = None
        if face_landmarker_path:
            self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_landmarker_path),
                running_mode=vision.RunningMode.IMAGE, num_faces=1))
        self.threshold = confidence_threshold

    # ---------- titik petunjuk ----------
    def _seeds_from_face(self, mp_img, w, h) -> List[Tuple[int, int]]:
        if self.landmarker is None:
            return []
        res = self.landmarker.detect(mp_img)
        if not res.face_landmarks:
            return []
        lm = res.face_landmarks[0]
        seeds = []
        for low, up in LOWER_LID.values():
            lx, ly = lm[low].x * w, lm[low].y * h
            gap = abs(ly - lm[up].y * h)
            seeds.append((int(lx), int(min(h - 1, ly + max(4.0, 0.8 * gap)))))
        return seeds

    @staticmethod
    def _seed_from_redness(image_bgr) -> Tuple[int, int]:
        a = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
        k = int(max(image_bgr.shape[:2]) / 15) | 1
        a = cv2.GaussianBlur(a, (k, k), 0)
        y, x = np.unravel_index(int(np.argmax(a)), a.shape)
        return int(x), int(y)

    # ---------- segmentasi ----------
    def _segment_at(self, mp_img, image_bgr, seed) -> np.ndarray:
        from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
        from mediapipe.tasks.python.vision import InteractiveSegmenterLegacyRegionOfInterest as ROI
        h, w = image_bgr.shape[:2]
        roi = ROI(format=ROI.Format.KEYPOINT, keypoint=NormalizedKeypoint(x=seed[0] / w, y=seed[1] / h))
        conf = np.squeeze(self.segmenter.segment(mp_img, roi).confidence_masks[0].numpy_view())
        obj = (conf > self.threshold).astype(np.uint8)
        if obj.sum() == 0:
            return obj * 255
        a = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
        vals = a[obj > 0].reshape(-1, 1)
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        red = ((a > thr) & (obj > 0)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red = cv2.morphologyEx(cv2.morphologyEx(red, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab = cv2.connectedComponents(red)
        if n <= 1:
            return np.zeros_like(red)
        sx, sy = seed
        idx = lab[sy, sx]
        if idx == 0:  # titik tidak jatuh di komponen merah: ambil komponen terdekat
            ys, xs = np.nonzero(lab)
            j = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
            idx = lab[ys[j], xs[j]]
        return (lab == idx).astype(np.uint8) * 255

    def extract_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        seeds = self._seeds_from_face(mp_img, w, h)
        seed_source = "face_landmarker" if seeds else "redness_peak"
        if not seeds:
            seeds = [self._seed_from_redness(image_bgr)]
        mask = np.zeros((h, w), np.uint8)
        for s in seeds:
            mask |= self._segment_at(mp_img, image_bgr, s)
        return mask, {"detected": bool(mask.any()), "latency_ms": (time.perf_counter() - t0) * 1000,
                      "seed_source": seed_source, "seeds": seeds}
