"""U-Net (encoder ResNet34) untuk segmentasi piksel konjungtiva, Tahap 2 Pipeline 1."""
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_unet(encoder_name="resnet34", encoder_weights: Optional[str] = "imagenet", in_channels=3, classes=1):
    import segmentation_models_pytorch as smp
    # Keluaran berupa logit; sigmoid diterapkan saat inferensi dan di dalam loss.
    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels,
                    classes=classes, activation=None)


def to_tensor(image_bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class UNetConjunctivaSegmenter:
    def __init__(self, weights_path: str, encoder_name="resnet34", input_size=256, threshold=0.5, device="cuda"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model = build_unet(encoder_name, encoder_weights=None)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.input_size = input_size
        self.threshold = threshold

    @torch.no_grad()
    def predict_mask(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = crop_bgr.shape[:2]
        x = to_tensor(crop_bgr, self.input_size).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob = torch.sigmoid(self.model(x))[0, 0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000.0
        m = (prob.cpu().numpy() > self.threshold).astype(np.uint8) * 255
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST), latency
