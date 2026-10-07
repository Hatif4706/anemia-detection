import numpy as np
import cv2
from typing import Dict, Any, Optional
from skimage.feature import graycomatrix, graycoprops

class ConjunctivaFeatureExtractor:
    """
    Ekstraktor fitur kolorimetri medis dan tekstur khusus untuk area konjungtiva palpebra.
    Mengekstraksi fitur warna yang sensitif terhadap kadar hemoglobin / kepucatan (pallor),
    termasuk CIE Lab a*, Erythema Index, HSV, dan Haralick GLCM features.
    """

    def __init__(self, use_glcm: bool = True):
        self.use_glcm = use_glcm

    def extract_features_from_roi(
        self, 
        image_bgr: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Mengekstrak kumpulan fitur terstandarisasi dari area ROI konjungtiva.
        
        Args:
            image_bgr: Citra BGR (format OpenCV).
            mask: Masker biner (0 atau 255/1). Jika None, seluruh area dihitung.
            
        Returns:
            Dictionary fitur numerik terstandarisasi.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image_bgr kosong atau tidak valid.")

        if mask is not None:
            # Pastikan mask berupa biner boolean
            binary_mask = (mask > 0).astype(bool)
            if not np.any(binary_mask):
                # Fallback jika mask kosong
                binary_mask = np.ones(image_bgr.shape[:2], dtype=bool)
        else:
            binary_mask = np.ones(image_bgr.shape[:2], dtype=bool)

        # 1. Ruang Warna RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        r_pixels = image_rgb[:, :, 0][binary_mask].astype(np.float64)
        g_pixels = image_rgb[:, :, 1][binary_mask].astype(np.float64)
        b_pixels = image_rgb[:, :, 2][binary_mask].astype(np.float64)

        # 2. Ruang Warna HSV
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h_pixels = image_hsv[:, :, 0][binary_mask].astype(np.float64)
        s_pixels = image_hsv[:, :, 1][binary_mask].astype(np.float64)
        v_pixels = image_hsv[:, :, 2][binary_mask].astype(np.float64)

        # 3. Ruang Warna CIE L*a*b* (Krusial untuk kemerahan konjungtiva: a*)
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_pixels = image_lab[:, :, 0][binary_mask].astype(np.float64)
        a_pixels = image_lab[:, :, 1][binary_mask].astype(np.float64)
        b_lab_pixels = image_lab[:, :, 2][binary_mask].astype(np.float64)

        eps = 1e-6
        mean_r = float(np.mean(r_pixels))
        mean_g = float(np.mean(g_pixels))
        mean_b = float(np.mean(b_pixels))

        # Fitur Kolorimetri Klinis & Indeks Kepucatan
        # Erythema Index = log(R) - log(G)
        safe_r = np.clip(r_pixels, 1.0, 255.0)
        safe_g = np.clip(g_pixels, 1.0, 255.0)
        erythema_values = np.log(safe_r) - np.log(safe_g)
        erythema_index = float(np.mean(erythema_values))

        # Redness Ratio: R / (R + G + B)
        total_rgb = mean_r + mean_g + mean_b + eps
        red_ratio = mean_r / total_rgb
        green_ratio = mean_g / total_rgb
        blue_ratio = mean_b / total_rgb

        # Pallor Index (Normalized Difference): (R - G) / (R + G)
        pallor_index = (mean_r - mean_g) / (mean_r + mean_g + eps)

        # Red-to-Green Ratio: R / G
        rg_ratio = mean_r / (mean_g + eps)

        features: Dict[str, float] = {
            # RGB Stats
            "rgb_r_mean": mean_r,
            "rgb_r_std": float(np.std(r_pixels)),
            "rgb_g_mean": mean_g,
            "rgb_g_std": float(np.std(g_pixels)),
            "rgb_b_mean": mean_b,
            "rgb_b_std": float(np.std(b_pixels)),
            "red_ratio": red_ratio,
            "green_ratio": green_ratio,
            "blue_ratio": blue_ratio,
            "rg_ratio": rg_ratio,
            "pallor_index": pallor_index,
            "erythema_index": erythema_index,

            # HSV Stats
            "hsv_h_mean": float(np.mean(h_pixels)),
            "hsv_h_std": float(np.std(h_pixels)),
            "hsv_s_mean": float(np.mean(s_pixels)),
            "hsv_s_std": float(np.std(s_pixels)),
            "hsv_v_mean": float(np.mean(v_pixels)),
            "hsv_v_std": float(np.std(v_pixels)),

            # CIE Lab Stats (a* adalah indikator utama rona kemerahan)
            "lab_l_mean": float(np.mean(l_pixels)),
            "lab_l_std": float(np.std(l_pixels)),
            "lab_a_mean": float(np.mean(a_pixels)),
            "lab_a_std": float(np.std(a_pixels)),
            "lab_b_mean": float(np.mean(b_lab_pixels)),
            "lab_b_std": float(np.std(b_lab_pixels)),
        }

        # 4. Fitur Tekstur GLCM (Haralick Features)
        if self.use_glcm:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # Batasi GLCM pada bounding box area bertopeng
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                cropped_gray = gray[y_min:y_max+1, x_min:x_max+1]
                
                # Rescale jika terlalu besar untuk performa GLCM
                if cropped_gray.shape[0] > 128 or cropped_gray.shape[1] > 128:
                    cropped_gray = cv2.resize(cropped_gray, (128, 128))

                try:
                    glcm = graycomatrix(
                        cropped_gray, 
                        distances=[1, 3], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, 
                        symmetric=True, 
                        normed=True
                    )
                    features["glcm_contrast"] = float(np.mean(graycoprops(glcm, 'contrast')))
                    features["glcm_dissimilarity"] = float(np.mean(graycoprops(glcm, 'dissimilarity')))
                    features["glcm_homogeneity"] = float(np.mean(graycoprops(glcm, 'homogeneity')))
                    features["glcm_energy"] = float(np.mean(graycoprops(glcm, 'energy')))
                    features["glcm_correlation"] = float(np.mean(graycoprops(glcm, 'correlation')))
                except Exception:
                    features["glcm_contrast"] = 0.0
                    features["glcm_dissimilarity"] = 0.0
                    features["glcm_homogeneity"] = 0.0
                    features["glcm_energy"] = 0.0
                    features["glcm_correlation"] = 0.0

        return features
