"""Cek bukti target minggu 12: anotasi selesai, YOLOv8n terlatih, IoU/Dice Pipeline 2 terukur.

    python scripts/cek_progres.py                                  # hasil di folder repo
    python scripts/cek_progres.py --dir hasil_colab/run_utama      # hasil run Colab (salinan ringkasan)

Bila folder berisi drive_manifest.json, file besar (bobot .pt, gambar) yang hanya tersimpan di
Google Drive dianggap ada sesuai manifest tersebut. Hanya memakai library bawaan Python.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, PART, NO = "[SELESAI]  ", "[SEBAGIAN] ", "[BELUM]    "


class Bukti:
    def __init__(self, base):
        self.base = base
        m = base / "drive_manifest.json"
        self.manifest = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {"files": {}}

    def exists(self, rel):
        return (self.base / rel).exists() or rel in self.manifest["files"]

    def load(self, rel):
        p = self.base / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def stamp(self, rel):
        d = self.manifest["files"].get(rel)
        if d:
            size = d["size"] / 1e6
            size = f"{size:.1f} MB" if size >= 0.1 else f"{d['size']} B"
            return f"{rel}  (Google Drive, {size}, {d['modified_wib']} WIB)"
        p = self.base / rel
        if not p.exists():
            return f"{rel}: TIDAK ADA"
        t = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d %b %Y %H:%M")
        return f"{rel}  (lokal, diubah {t})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT), help="folder yang berisi reports/ dan models/")
    base = Path(ap.parse_args().dir).resolve()
    b = Bukti(base)
    status = []

    print("=" * 74)
    print(" CEK TARGET MINGGU 12: Anotasi selesai dan pelatihan YOLO")
    print(" Folder:", base)
    if b.manifest.get("drive_folder"):
        print(f" Sumber: {b.manifest['drive_folder']}  (commit {b.manifest.get('commit', '?')})")
    print("=" * 74)
    data = b.load("reports/data_summary.json")
    if not data:
        print(NO, "reports/data_summary.json tidak ada. Jalankan tahap 1 dulu.")
        return
    n_total, src = data["n_images"], data["source"]
    print(f"\nDataset: {src}  |  {n_total} citra ({data['n_anemia']} anemia, {data['n_non_anemia']} non-anemia)")
    print(f"Split per subjek: {data['split']}")
    if src != "eyes_defy_anemia":
        print("PERINGATAN: ini data sintetis, bukan bukti penelitian.")

    # 1. Anotasi
    print("\n--- 1. Anotasi 100% tervalidasi " + "-" * 41)
    fig = "reports/figures/anotasi_bbox_dari_mask.png"
    ann = b.load("reports/annotation_summary.json")
    print("Bukti:", b.stamp("reports/annotation_summary.json"))
    print("      ", b.stamp(fig))
    if ann:
        lab = sum(v["labeled"] for v in ann.values())
        img = sum(v["images"] for v in ann.values())
        for k, v in ann.items():
            print(f"   {k:5s}: {v['labeled']:3d}/{v['images']:3d} citra berlabel")
        pct = 100 * lab / img if img else 0
        print(f"   Total: {lab}/{img} = {pct:.0f}%  (bbox dibuat otomatis dari mask ahli dataset)")
        done = lab == img == n_total and b.exists(fig) and src == "eyes_defy_anemia"
        status.append((OK if done else PART, "Anotasi bbox + mask", f"{lab}/{img} citra ({pct:.0f}%)"))
    else:
        status.append((NO, "Anotasi bbox + mask", "annotation_summary.json tidak ada"))

    # 2. YOLO
    print("\n--- 2. YOLOv8n terlatih (mAP) " + "-" * 43)
    w = "models/yolo_conjunctiva.pt"
    yolo = b.load("reports/yolo_summary.json")
    print("Bukti:", b.stamp("reports/yolo_summary.json"))
    print("      ", b.stamp(w))
    if yolo:
        print(f"   Epoch {yolo['epochs']}, bobot awal pretrained COCO: {yolo['pretrained']}")
        print(f"   Data uji: mAP50 = {yolo['test_mAP50']:.3f} | mAP50-95 = {yolo['test_mAP50_95']:.3f} | "
              f"precision = {yolo['test_precision']:.3f} | recall = {yolo['test_recall']:.3f}")
        if b.exists(w):
            status.append((OK, "YOLOv8n terlatih", f"mAP50 uji {yolo['test_mAP50']:.3f}"))
        else:
            status.append((PART, "YOLOv8n terlatih", f"mAP50 uji {yolo['test_mAP50']:.3f}, file bobot tidak ada"))
    else:
        status.append((NO, "YOLOv8n terlatih", "yolo_summary.json tidak ada"))
    unet = b.load("reports/unet_summary.json")
    if unet:
        print(f"   (U-Net ikut terlatih: {unet['epochs']} epoch, Dice validasi terbaik {unet['best_val_dice']:.3f})")

    # 3. IoU/Dice Pipeline 2
    print("\n--- 3. IoU/Dice Pipeline 2 terukur " + "-" * 38)
    seg = b.load("reports/segmentation_summary.json")
    print("Bukti:", b.stamp("reports/segmentation_summary.json"))
    if seg:
        n_seg = seg["p2_mediapipe"]["all"]["n"]
        names = {"p1_yolo_unet": "P1  YOLO+U-Net", "p2_mediapipe": "P2  Face Mesh", "p2b_mediapipe_magictouch": "P2b MagicTouch"}
        for part, judul in [("all", "semua citra"), ("test", "data uji")]:
            print(f"   [{judul}]")
            print(f"   {'Pipeline':16s} {'n':>4s} {'deteksi':>8s} {'IoU':>6s} {'Dice':>6s} {'latensi':>10s}")
            for k, label in names.items():
                if k in seg:
                    a = seg[k][part]
                    print(f"   {label:16s} {a['n']:4d} {a['detection_rate']*100:7.1f}% {a['mean_iou_all']:6.3f} "
                          f"{a['mean_dice_all']:6.3f} {a['mean_latency_ms']:8.1f}ms")
        fail = seg["p2_mediapipe"]["all"]["failures_by_stage"]
        if fail:
            print(f"   Catatan P2: gagal di tahap {fail} (wajah tidak terdeteksi pada foto close-up).")
        if n_seg == n_total and src == "eyes_defy_anemia":
            p2 = seg["p2_mediapipe"]["all"]
            status.append((OK, "IoU/Dice Pipeline 2", f"IoU {p2['mean_iou_all']:.3f}, Dice {p2['mean_dice_all']:.3f} pada {n_seg} citra"))
        else:
            print(f"   Catatan: diukur pada {n_seg} citra, sedangkan dataset asli {n_total} citra (data sintetis).")
            status.append((NO, "IoU/Dice Pipeline 2", f"baru data sintetis ({n_seg} citra), belum data asli"))
    else:
        status.append((NO, "IoU/Dice Pipeline 2", "segmentation_summary.json tidak ada (tahap 4)"))

    print("\n" + "=" * 74)
    print(" RINGKASAN")
    print("=" * 74)
    for s, name, info in status:
        print(f" {s}{name:22s} {info}")
    done = sum(s == OK for s, _, _ in status)
    print(f"\n Target minggu 12: {done}/{len(status)} kriteria selesai.")
    if done < len(status):
        print(" Lanjutkan dengan menjalankan notebooks/train_colab.ipynb (tahap 1-4) di Google Colab.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
