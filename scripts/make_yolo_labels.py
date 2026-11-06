"""Langkah 2: anotasi bounding box otomatis dari mask ahli (format YOLO).

Kotak = batas terluar mask konjungtiva palpebral. Hasil disimpan di data/processed/yolo
beserta data.yaml dan contoh visualisasi untuk validasi manual.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset_loader import load_split
from src.utils import ROOT, apply_clahe, load_config, mask_bbox, read_image, read_mask, write_image


def main():
    cfg = load_config()
    out = ROOT / cfg["dataset"]["processed_dir"] / "yolo"
    if out.exists():
        shutil.rmtree(out)
    stats, preview = {}, []
    for split in ["train", "val", "test"]:
        df = load_split(str(ROOT / cfg["dataset"]["splits_dir"]), split)
        n_ok = 0
        for _, r in df.iterrows():
            img = read_image(r["image_path"])
            mask = read_mask(r["mask_path"], img.shape)
            box = mask_bbox(mask)
            if box is None:
                continue
            if cfg["preprocessing"]["clahe_for_segmentation"]:
                img = apply_clahe(img)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            line = f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
            write_image(out / "images" / split / f"{r['sample_id']}.jpg", img)
            lp = out / "labels" / split / f"{r['sample_id']}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(line, encoding="utf-8")
            n_ok += 1
            if len(preview) < 6:
                vis = img.copy()
                t = max(3, w // 150)  # tebal garis mengikuti resolusi agar tetap terlihat setelah diperkecil
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), t)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (255, 0, 255), t)
                preview.append(cv2.resize(vis, (400, int(400 * h / w))))
        stats[split] = {"images": int(len(df)), "labeled": n_ok}
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {cfg['yolo_detector']['class_name']}\n", encoding="utf-8")
    if preview:
        import numpy as np
        hmax = max(p.shape[0] for p in preview)
        preview = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                   for p in preview]
        rows = [np.hstack(preview[i:i + 3]) for i in range(0, len(preview), 3)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
        write_image(ROOT / "reports" / "figures" / "anotasi_bbox_dari_mask.png", np.vstack(rows))
    (ROOT / "reports" / "annotation_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
