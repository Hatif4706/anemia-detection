"""YOLOv8n untuk mendeteksi area konjungtiva palpebral, Tahap 1 Pipeline 1."""
import time
from typing import Dict, Optional, Tuple

import numpy as np


class YOLOConjunctivaDetector:
    def __init__(self, weights_path: str, confidence: float = 0.25, iou_threshold: float = 0.5, device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.confidence = confidence
        self.iou = iou_threshold
        self.device = 0 if device == "cuda" else "cpu"

    def detect(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Dict]:
        """Mengembalikan kotak dengan skor tertinggi (x1, y1, x2, y2) atau None bila tidak ada deteksi."""
        t0 = time.perf_counter()
        res = self.model.predict(image_bgr, conf=self.confidence, iou=self.iou, device=self.device, verbose=False)
        latency = (time.perf_counter() - t0) * 1000.0
        boxes = res[0].boxes if res else None
        if boxes is None or len(boxes) == 0:
            return None, {"latency_ms": latency, "confidence": 0.0}
        i = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        return (x1, y1, x2, y2), {"latency_ms": latency, "confidence": float(boxes.conf[i])}
