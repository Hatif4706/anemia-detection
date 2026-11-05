"""Penyusunan metadata, pembagian data per subjek, dan data sintetis untuk uji fungsi kode."""
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.labels import anemia_label
from src.utils import write_image


def create_subject_splits(df: pd.DataFrame, splits_dir: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split di tingkat subjek: semua citra satu subjek berada di bagian yang sama."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    subj = df.groupby("subject_id")["anemia_label"].max().reset_index()
    tr_s, tmp_s = train_test_split(subj, test_size=val_ratio + test_ratio, stratify=subj["anemia_label"],
                                   random_state=seed)
    va_s, te_s = train_test_split(tmp_s, test_size=test_ratio / (val_ratio + test_ratio),
                                  stratify=tmp_s["anemia_label"], random_state=seed)
    out = []
    os.makedirs(splits_dir, exist_ok=True)
    for name, s in [("train", tr_s), ("val", va_s), ("test", te_s)]:
        part = df[df["subject_id"].isin(s["subject_id"])].copy()
        part.to_csv(Path(splits_dir) / f"{name}.csv", index=False)
        out.append(part)
    return tuple(out)


def load_split(splits_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / f"{name}.csv")


def generate_synthetic_dataset(out_dir: str, label_cfg: dict, num_samples: int = 80, seed: int = 42) -> pd.DataFrame:
    """Citra close-up kelopak mata bawah buatan, meniru tata letak Eyes-defy-anemia.

    HANYA untuk menguji bahwa kode berjalan dari awal sampai akhir. Hasil pada data ini
    tidak boleh dilaporkan sebagai hasil penelitian.
    """
    rng = np.random.default_rng(seed)
    img_dir, mask_dir = Path(out_dir) / "images", Path(out_dir) / "masks"
    rows = []
    for i in range(num_samples):
        sex = "female" if rng.random() < 0.5 else "male"
        hb = float(np.clip(rng.normal(12.2, 2.0), 6.0, 17.5))
        H, W = 600, 800
        skin = np.array([150, 170, 205]) + rng.normal(0, 12, 3)  # BGR
        img = np.ones((H, W, 3), np.float32) * skin
        cx, cy = int(W / 2 + rng.normal(0, 40)), int(H * 0.42 + rng.normal(0, 25))
        ax, ay = int(rng.uniform(230, 300)), int(rng.uniform(80, 110))
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (235, 238, 240), -1)  # sklera
        cv2.circle(img, (cx + int(rng.normal(0, 20)), cy), int(ay * 0.85), (40, 55, 75), -1)  # iris
        # konjungtiva palpebral: bulan sabit di bawah mata (kelopak ditarik ke bawah)
        mask = np.zeros((H, W), np.uint8)
        outer = cv2.ellipse2Poly((cx, cy + 10), (ax, ay + int(rng.uniform(70, 110))), 0, 10, 170, 5)
        inner = cv2.ellipse2Poly((cx, cy), (int(ax * 0.97), ay), 0, 10, 170, 5)[::-1]
        cv2.fillPoly(mask, [np.vstack([outer, inner]).astype(np.int32)], 255)
        redness = np.clip((hb - 6.0) / 11.0 + rng.normal(0, 0.12), 0, 1)  # makin rendah Hb makin pucat
        pale, red = np.array([175, 180, 225]), np.array([70, 75, 200])
        conj = pale * (1 - redness) + red * redness
        noise = rng.normal(0, 10, (H, W, 3))
        img[mask > 0] = conj + noise[mask > 0]
        for _ in range(6):  # pembuluh darah
            x0 = int(cx + rng.uniform(-ax * 0.8, ax * 0.8))
            cv2.line(img, (x0, cy + ay), (x0 + int(rng.normal(0, 30)), cy + ay + 50), tuple(conj * 0.8), 2)
        grad = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.15), W)[None, :, None]  # pencahayaan
        img = np.clip(img * grad + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        sid = f"synthetic_{i:03d}"
        ip, mp = img_dir / f"{sid}.jpg", mask_dir / f"{sid}.png"
        write_image(ip, img)
        write_image(mp, mask)
        rows.append({"sample_id": sid, "subject_id": sid, "country": "synthetic", "image_path": str(ip),
                     "mask_path": str(mp), "hb": round(hb, 2), "sex": sex, "age": None,
                     "anemia_label": anemia_label(hb, sex, None, label_cfg)})
    return pd.DataFrame(rows)
