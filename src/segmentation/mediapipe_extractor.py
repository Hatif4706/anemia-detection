import os
import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

class MediaPipeConjunctivaExtractor:
    """
    Ekstraktor Region of Interest (ROI) konjungtiva palpebra berbasis MediaPipe Face Landmarker (Tasks API).
    Menggunakan titik-titik landmark kelopak mata bawah (inferior palpebral eyelid)
    dan membentuk masker poligon yang dapat disesuaikan faktor ekspansinya.
    """

    # Indeks landmark kelopak mata bawah MediaPipe (468/478 landmarks)
    LEFT_LOWER_EYELID_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_LOWER_EYELID_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362]

    # Titik sulkus / kantung bawah mata untuk memperluas area eversi konjungtiva
    LEFT_SULCUS_INDICES = [111, 117, 118, 119, 120, 100, 126]
    RIGHT_SULCUS_INDICES = [340, 346, 347, 348, 349, 329, 355]

    def __init__(
        self,
        model_asset_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        expansion_factor: float = 1.15
    ):
        self.model_asset_path = os.path.abspath(model_asset_path)
        self.num_faces = num_faces
        self.expansion_factor = expansion_factor
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions

            if not os.path.exists(self.model_asset_path):
                raise FileNotFoundError(f"Model file {self.model_asset_path} tidak ditemukan.")

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_asset_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.num_faces
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)

        return self._detector

    def extract_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Menghasilkan masker biner konjungtiva (uint8, nilai 255) dari citra input.
        
        Args:
            image_bgr: Citra wajah / mata berformat BGR.
            
        Returns:
            Tuple of (binary_mask, metadata_dict)
            metadata_dict berisi latency_ms, status_deteksi, dll.
        """
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        start_time = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detector = self._get_detector()
        detection_result = detector.detect(mp_image)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            # Fallback jika deteksi landmark tidak menemukan wajah (misal citra crop mata lokal)
            return mask, {
                "detected": False,
                "latency_ms": latency_ms,
                "message": "Tidak ada wajah/landmarks terdeteksi"
            }

        landmarks = detection_result.face_landmarks[0]

        # Fungsi pembantu untuk membuat poligon dari daftar indeks
        def build_eye_polygon(eyelid_indices: List[int], sulcus_indices: List[int]) -> np.ndarray:
            eyelid_pts = []
            for idx in eyelid_indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    eyelid_pts.append([int(pt.x * w), int(pt.y * h)])

            if not eyelid_pts:
                return np.array([], dtype=np.int32)

            pts_array = np.array(eyelid_pts, dtype=np.int32)
            centroid_y = np.mean(pts_array[:, 1])

            expanded_pts = []
            for p in pts_array:
                x, y = p
                dy = y - centroid_y
                new_y = int(y + max(0, dy) * (self.expansion_factor - 1.0) * 1.5)
                expanded_pts.append([x, new_y])

            sulcus_pts = []
            for idx in sulcus_indices:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    sulcus_pts.append([int(pt.x * w), int(pt.y * h)])

            if sulcus_pts:
                all_pts = np.vstack([np.array(expanded_pts), np.array(sulcus_pts)])
            else:
                all_pts = np.array(expanded_pts)

            hull = cv2.convexHull(all_pts)
            return hull

        left_hull = build_eye_polygon(self.LEFT_LOWER_EYELID_INDICES, self.LEFT_SULCUS_INDICES)
        right_hull = build_eye_polygon(self.RIGHT_LOWER_EYELID_INDICES, self.RIGHT_SULCUS_INDICES)

        if len(left_hull) > 0:
            cv2.fillPoly(mask, [left_hull], 255)
        if len(right_hull) > 0:
            cv2.fillPoly(mask, [right_hull], 255)

        return mask, {
            "detected": True,
            "latency_ms": latency_ms,
            "message": "Deteksi landmark berhasil"
        }

    def close(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None
