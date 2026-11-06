"""Menjalankan seluruh tahap secara berurutan.

    python scripts/run_pipeline.py                      # data penelitian (config: dataset.source)
    python scripts/run_pipeline.py --start 3b           # lanjut dari tahap 3b (tahap sebelumnya dilewati)
    python scripts/run_pipeline.py --source synthetic --smoke   # uji fungsi kode, cepat, tanpa unduhan bobot
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["1", "2", "3a", "3b", "4", "5", "6", "7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["eyes_defy_anemia", "synthetic"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start", choices=STEPS, default="1", help="mulai dari tahap ini")
    ap.add_argument("--stop", choices=STEPS, default="7", help="berhenti setelah tahap ini")
    a = ap.parse_args()
    smoke = ["--smoke"] if a.smoke else []
    plan = [
        ("1", "Metadata dan pembagian data per subjek", ["prepare_data.py"] + (["--source", a.source] if a.source else [])),
        ("2", "Anotasi bounding box otomatis dari mask", ["make_yolo_labels.py"]),
        ("3a", "Pelatihan YOLOv8n", ["train_yolo.py"] + smoke),
        ("3b", "Pelatihan U-Net (encoder ResNet34)", ["train_unet.py"] + smoke),
        ("4", "Ekstraksi ROI kedua pipeline + metrik segmentasi", ["extract_rois.py"]),
        ("5", "Pelatihan classifier MobileNetV3-Small (setelan identik)", ["train_classifier.py"] + smoke),
        ("6", "Evaluasi dan uji statistik", ["evaluate.py"]),
        ("7", "Laporan progres", ["make_progress_report.py"]),
    ]
    for sid, name, args in plan[STEPS.index(a.start):STEPS.index(a.stop) + 1]:
        print(f"\n{'=' * 70}\n{sid}. {name}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Tahap gagal: {sid}. {name}  (lanjutkan dengan: python scripts/run_pipeline.py --start {sid})")


if __name__ == "__main__":
    main()
