import time
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class YOLOEyelidDetector:
    """
    Detektor bounding box area mata / kelopak mata bawah menggunakan YOLO (Ultralytics).
    Berfungsi sebagai Tahap 1 pada arsitektur Two-Stage YOLO + U-Net.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect_and_crop(
        self, 
        image_bgr: np.ndarray, 
        padding_ratio: float = 0.1
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Dict[str, Any]]:
        """
        Mendeteksi mata/kelopak mata dan memotong (crop) area tersebut dengan padding.
        
        Args:
            image_bgr: Citra masukan format BGR.
            padding_ratio: Rasio perluasan bounding box agar konjungtiva tidak terpotong.
            
        Returns:
            Tuple of:
            - List cropped images (BGR)
            - List coordinates (x1, y1, x2, y2)
            - Metadata (latency_ms, detections_count)
        """
        model = self._load_model()
        h, w = image_bgr.shape[:2]

        start_time = time.perf_counter()
        results = model.predict(
            image_bgr, 
            conf=self.confidence, 
            iou=self.iou_threshold, 
            device=self.device, 
            verbose=False
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        crops = []
        boxes_coords = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Berikan padding
                bw = x2 - x1
                bh = y2 - y1
                pad_w = int(bw * padding_ratio)
                pad_h = int(bh * padding_ratio)

                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_h)

                crop = image_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
                crops.append(crop)
                boxes_coords.append((x1_pad, y1_pad, x2_pad, y2_pad))

        # Fallback jika tidak ada deteksi: gunakan citra penuh
        if len(crops) == 0:
            crops.append(image_bgr.copy())
            boxes_coords.append((0, 0, w, h))

        metadata = {
            "latency_ms": latency_ms,
            "num_detections": len(boxes_coords)
        }

        return crops, boxes_coords, metadata
