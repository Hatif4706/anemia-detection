"""Utilitas bersama: konfigurasi, seed, perangkat, dan prapemrosesan sederhana."""
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "cuda") -> str:
    try:
        import torch
        return "cuda" if preferred == "cuda" and torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE pada kanal L (ruang warna LAB) agar rona warna tidak bergeser."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def read_image(path: str) -> np.ndarray:
    """cv2.imread yang aman untuk path Windows dengan karakter non-ASCII."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Gagal membaca citra: {path}")
    return img


def read_mask(path: str, shape=None) -> np.ndarray:
    """Membaca mask; piksel tidak hitam dianggap konjungtiva. Mengembalikan uint8 {0,255}."""
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Gagal membaca mask: {path}")
    if m.ndim == 3:
        if m.shape[2] == 4:  # PNG dengan alpha: pakai alpha bila informatif
            alpha = m[:, :, 3]
            m = alpha if alpha.min() < 250 else m[:, :, :3].max(axis=2)
        else:
            m = m.max(axis=2)
    m = (m > 10).astype(np.uint8) * 255
    if shape is not None and m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def write_image(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Gagal menyimpan {path}")
    buf.tofile(str(path))


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_box(box, pad_ratio: float, w: int, h: int):
    x1, y1, x2, y2 = box
    pw, ph = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    return max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)


def roi_from_mask(image_bgr: np.ndarray, mask: np.ndarray, size: int = 224):
    """Potong citra × mask ke kotak mask, lalu letterbox ke size×size. None bila mask kosong."""
    box = mask_bbox(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2].copy()
    crop[mask[y1:y2, x1:x2] == 0] = 0
    h, w = crop.shape[:2]
    s = size / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"
