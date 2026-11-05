"""Langkah 3a: fine-tuning YOLOv8n untuk mendeteksi area konjungtiva palpebral (1 kelas).

    python scripts/train_yolo.py            # pretrained COCO (yolov8n.pt), sesuai config
    python scripts/train_yolo.py --smoke    # uji fungsi: tanpa bobot pretrained, epoch sedikit
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config()
    y = cfg["yolo_detector"]
    set_seed(cfg["experiment"]["seed"])
    from ultralytics import YOLO

    data_yaml = ROOT / cfg["dataset"]["processed_dir"] / "yolo" / "data.yaml"
    model = YOLO("yolov8n.yaml" if args.smoke else y["model_type"])
    # Tanpa bobot pretrained, YOLO butuh ~150 epoch pada data sintetis agar skornya melewati ambang deteksi.
    epochs = args.epochs or (150 if args.smoke else y["epochs"])
    device = 0 if get_device(cfg["experiment"]["device"]) == "cuda" else "cpu"
    res = model.train(data=str(data_yaml), epochs=epochs, imgsz=y["imgsz"], batch=y["batch_size"], device=device,
                      seed=cfg["experiment"]["seed"], project=str(ROOT / "runs" / "yolo"), name="train",
                      exist_ok=True, plots=not args.smoke, amp=not args.smoke, verbose=False, workers=0,
                      pretrained=not args.smoke)
    best = Path(res.save_dir) / "weights" / "best.pt"
    out = ROOT / y["weights_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    metrics = YOLO(str(out)).val(data=str(data_yaml), split="test", imgsz=y["imgsz"], device=device,
                                 plots=False, verbose=False, workers=0)
    summary = {"epochs": epochs, "pretrained": not args.smoke, "weights": str(out.relative_to(ROOT)),
               "test_mAP50": float(metrics.box.map50), "test_mAP50_95": float(metrics.box.map),
               "test_precision": float(metrics.box.mp), "test_recall": float(metrics.box.mr)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "yolo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
