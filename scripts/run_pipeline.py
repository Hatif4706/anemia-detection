"""Menjalankan seluruh tahap secara berurutan.

    python scripts/run_pipeline.py                      # data penelitian (config: dataset.source)
    python scripts/run_pipeline.py --source synthetic --smoke   # uji fungsi kode, cepat, tanpa unduhan bobot
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(name, args):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Tahap gagal: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["eyes_defy_anemia", "synthetic"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    smoke = ["--smoke"] if a.smoke else []
    step("1. Metadata dan pembagian data per subjek", ["prepare_data.py"] + (["--source", a.source] if a.source else []))
    step("2. Anotasi bounding box otomatis dari mask", ["make_yolo_labels.py"])
    step("3a. Pelatihan YOLOv8n", ["train_yolo.py"] + smoke)
    step("3b. Pelatihan U-Net (encoder ResNet34)", ["train_unet.py"] + smoke)
    step("4. Ekstraksi ROI Pipeline 1 dan Pipeline 2 + metrik segmentasi", ["extract_rois.py"])
    step("5. Pelatihan classifier MobileNetV3-Small (setelan identik)", ["train_classifier.py"] + smoke)
    step("6. Evaluasi dan uji statistik", ["evaluate.py"])
    step("7. Laporan progres", ["make_progress_report.py"])


if __name__ == "__main__":
    main()
