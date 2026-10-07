import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

class MediaPipeConjunctivaExtractor:
    """
    Ekstraktor Region of Interest (ROI) konjungtiva palpebra berbasis MediaPipe Face Mesh.
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
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        expansion_factor: float = 1.15
    ):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.expansion_factor = expansion_factor
        self._face_mesh = None

    def _get_face_mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        return self._face_mesh

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
        
        face_mesh = self._get_face_mesh()
        results = face_mesh.process(image_rgb)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not results.multi_face_landmarks:
            return mask, {
                "detected": False,
                "latency_ms": latency_ms,
                "message": "Tidak ada wajah/landmarks terdeteksi"
            }

        landmarks = results.multi_face_landmarks[0].landmark

        # Fungsi pembantu untuk membuat poligon dari daftar indeks
        def build_eye_polygon(eyelid_indices: List[int], sulcus_indices: List[int]) -> np.ndarray:
            eyelid_pts = []
            for idx in eyelid_indices:
                pt = landmarks[idx]
                eyelid_pts.append([int(pt.x * w), int(pt.y * h)])

            # Ambil titik tengah dan perlebar sedikit ke arah bawah untuk menangkap konjungtiva eversi
            pts_array = np.array(eyelid_pts, dtype=np.int32)
            centroid_y = np.mean(pts_array[:, 1])

            expanded_pts = []
            for p in pts_array:
                x, y = p
                dy = y - centroid_y
                # Ekspan ke bawah jika titik berada di bagian bawah kelopak
                new_y = int(y + max(0, dy) * (self.expansion_factor - 1.0) * 1.5)
                expanded_pts.append([x, new_y])

            # Gabungkan dengan titik sulkus untuk menutup poligon kantung mata bawah
            sulcus_pts = []
            for idx in sulcus_indices:
                pt = landmarks[idx]
                sulcus_pts.append([int(pt.x * w), int(pt.y * h)])

            all_pts = np.vstack([np.array(expanded_pts), np.array(sulcus_pts)])
            hull = cv2.convexHull(all_pts)
            return hull

        left_hull = build_eye_polygon(self.LEFT_LOWER_EYELID_INDICES, self.LEFT_SULCUS_INDICES)
        right_hull = build_eye_polygon(self.RIGHT_LOWER_EYELID_INDICES, self.RIGHT_SULCUS_INDICES)

        cv2.fillPoly(mask, [left_hull], 255)
        cv2.fillPoly(mask, [right_hull], 255)

        return mask, {
            "detected": True,
            "latency_ms": latency_ms,
            "message": "Deteksi landmark berhasil"
        }

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
