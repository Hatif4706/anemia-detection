import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

class UNetConjunctivaSegmenter:
    """
    Model segmentasi semantik U-Net untuk isolasi piksel presisi
    pada konjungtiva palpebra mata (Tahap 2 dari Two-Stage Pipeline).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        input_size: Tuple[int, int] = (256, 256),
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.input_size = input_size
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self._model = None

    def _build_or_load_model(self):
        if self._model is None:
            import segmentation_models_pytorch as smp
            self._model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights if self.model_path is None else None,
                in_channels=self.in_channels,
                classes=self.classes,
                activation="sigmoid"
            )
            if self.model_path:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self._model.load_state_dict(state_dict)

            self._model.to(self.device)
            self._model.eval()

        return self._model

    def predict_mask(
        self, 
        image_crop_bgr: np.ndarray, 
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Memprediksi mask konjungtiva pada citra crop.
        
        Args:
            image_crop_bgr: Citra crop mata/kelopak BGR.
            threshold: Batas probabilitas biner.
            
        Returns:
            Tuple: (mask_biner uint8 resolusi crop asli, waktu_inferensi_ms)
        """
        orig_h, orig_w = image_crop_bgr.shape[:2]
        model = self._build_or_load_model()

        # Preprocessing input
        resized = cv2.resize(image_crop_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Standarisasi ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
            pred_prob = output.squeeze().cpu().numpy()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        binary_mask_small = (pred_prob > threshold).astype(np.uint8) * 255
        full_mask = cv2.resize(binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return full_mask, latency_ms
