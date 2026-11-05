"""Langkah 4: menjalankan Pipeline 1 dan Pipeline 2 pada semua citra, menyimpan mask + ROI,
serta menghitung metrik segmentasi terhadap mask ahli (evaluasi tingkat 1).

Keluaran:
  data/processed/rois/{gt,p1_yolo_unet,p2_mediapipe}/<sample_id>.png   (ROI 224x224 untuk classifier)
  data/processed/masks/{p1_yolo_unet,p2_mediapipe}/<sample_id>.png
  reports/segmentation_per_image.csv, reports/segmentation_summary.json
  reports/figures/contoh_roi_kedua_pipeline.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import EvaluationMetrics
from src.pipelines.roi_pipelines import build_pipelines
from src.preprocessing.dataset_loader import load_split
from src.utils import ROOT, get_device, load_config, read_image, read_mask, roi_from_mask, write_image


def overlay(img, mask, color):
    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tint = vis.copy()
    tint[mask > 0] = color
    vis = cv2.addWeighted(tint, 0.35, vis, 0.65, 0)
    cv2.drawContours(vis, cnts, -1, color, 3)
    return vis


def label(img, text):
    out = cv2.copyMakeBorder(img, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def main():
    cfg = load_config()
    device = get_device(cfg["experiment"]["device"])
    sd = str(ROOT / cfg["dataset"]["splits_dir"])
    df = pd.concat([load_split(sd, s).assign(split=s) for s in ["train", "val", "test"]], ignore_index=True)
    pipes = build_pipelines(cfg, device)
    styles = {"p1_yolo_unet": ((200, 60, 60), "P1 YOLO+U-Net"), "p2_mediapipe": ((20, 110, 230), "P2 Face Mesh"),
              "p2b_mediapipe_magictouch": ((0, 160, 255), "P2b MagicTouch")}
    proc = ROOT / cfg["dataset"]["processed_dir"]
    size = cfg["classifier"]["input_size"]
    rows, gallery = [], []
    for _, r in df.iterrows():
        img = read_image(r["image_path"])
        gt = read_mask(r["mask_path"], img.shape)
        roi = roi_from_mask(img, gt, size)
        if roi is not None:
            write_image(proc / "rois" / "gt" / f"{r['sample_id']}.png", roi)
        panels = [label(overlay(img, gt, (0, 200, 0)), "Mask ahli (GT)")]
        for p in pipes:
            mask, info = p(img)
            seg = EvaluationMetrics.calculate_segmentation_metrics(mask, gt) if info["detected"] else \
                {"iou": 0.0, "dice_score": 0.0, "precision": 0.0, "recall": 0.0}
            write_image(proc / "masks" / p.name / f"{r['sample_id']}.png", mask)
            proi = roi_from_mask(img, mask, size) if info["detected"] else None
            if proi is not None:
                write_image(proc / "rois" / p.name / f"{r['sample_id']}.png", proi)
            rows.append({"sample_id": r["sample_id"], "split": r["split"], "pipeline": p.name,
                         "detected": info["detected"], "stage_failed": info.get("stage_failed"),
                         "seed_source": info.get("seed_source"),
                         "latency_ms": round(info["latency_ms"], 2), **{k: round(v, 4) for k, v in seg.items()}})
            color, name = styles[p.name]
            title = name + (f"  Dice {seg['dice_score']:.2f}" if info["detected"] else "  GAGAL")
            panels.append(label(overlay(img, mask, color), title))
        if r["split"] == "test" and len(gallery) < 4:
            gallery.append(np.hstack([cv2.resize(p, (300, int(300 * p.shape[0] / p.shape[1]))) for p in panels]))

    per = pd.DataFrame(rows)
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    per.to_csv(rep / "segmentation_per_image.csv", index=False)
    summary = {}
    for name, g in per.groupby("pipeline"):
        for split, gs in [("all", g), ("test", g[g["split"] == "test"])]:
            det = gs[gs["detected"]]
            summary.setdefault(name, {})[split] = {
                "n": int(len(gs)), "detection_rate": round(float(gs["detected"].mean()), 4),
                "mean_iou_all": round(float(gs["iou"].mean()), 4),
                "mean_dice_all": round(float(gs["dice_score"].mean()), 4),
                "mean_dice_detected": round(float(det["dice_score"].mean()), 4) if len(det) else None,
                "mean_latency_ms": round(float(gs["latency_ms"].mean()), 2),
                "failures_by_stage": gs["stage_failed"].value_counts().to_dict(),
            }
    (rep / "segmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if gallery:
        wmax = max(g.shape[1] for g in gallery)
        gallery = [cv2.copyMakeBorder(g, 0, 6, 0, wmax - g.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                   for g in gallery]
        write_image(rep / "figures" / "contoh_roi_kedua_pipeline.png", np.vstack(gallery))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
