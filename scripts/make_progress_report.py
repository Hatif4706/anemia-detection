"""Langkah 7: laporan progres mandiri (HTML, gambar tertanam) dari keluaran tahap 1-6.

Keluaran: reports/progress/laporan_progres.html (bisa dibuka di browser atau dicetak ke PDF).
Status setiap target dihitung dari berkas hasil, bukan diisi manual.
"""
import base64
import datetime as dt
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, git_commit, load_config

REP = ROOT / "reports"


def jload(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def img_tag(path, caption):
    p = REP / "figures" / path
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def unet_curve():
    log = REP / "unet_training_log.csv"
    if not log.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(log)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(d["epoch"], d["train_loss"], label="train")
    ax[0].plot(d["epoch"], d["val_loss"], label="val")
    ax[0].set_title("Loss Dice + BCE")
    ax[0].legend()
    ax[1].plot(d["epoch"], d["val_dice"], color="#3f35a8")
    ax[1].set_title("Dice validasi")
    ax[1].set_ylim(0, 1)
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    (REP / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REP / "figures" / "kurva_pelatihan_unet.png", dpi=150)
    plt.close(fig)


def table(rows, header, cls=""):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main():
    cfg = load_config()
    unet_curve()
    data, ann = jload("data_summary.json"), jload("annotation_summary.json")
    yolo, unet, seg = jload("yolo_summary.json"), jload("unet_summary.json"), jload("segmentation_summary.json")
    cls, res = jload("classifier_summary.json"), jload("results.json")
    synthetic = (data or {}).get("source") == "synthetic"
    smoke = bool(unet and not unet.get("pretrained_encoder"))

    def status(ok, ran=False):
        return "Selesai" if ok else ("Kode berjalan, hasil perlu tindak lanjut" if ran else "Belum")

    n_lab = sum(v["labeled"] for v in ann.values()) if ann else 0
    n_img = sum(v["images"] for v in ann.values()) if ann else 0
    p2 = (seg or {}).get("p2_mediapipe", {}).get("all", {})
    p1 = (seg or {}).get("p1_yolo_unet", {}).get("all", {})
    targets = [
        ["Kode memakai U-Net asli (tanpa mask tiruan dari ground truth)", status(bool(unet)),
         f"U-Net ResNet34 dilatih {unet['epochs']} epoch, Dice validasi terbaik {unet['best_val_dice']:.3f}" if unet else "-"],
        ["Anotasi bounding box (min. 50% citra)", status(n_img and n_lab / n_img >= 0.5),
         f"{n_lab}/{n_img} citra berlabel otomatis dari mask ahli" if ann else "-"],
        ["Mask konjungtiva tersedia dan tervalidasi", status(bool(ann)),
         "Mask dari dataset; contoh kotak + kontur di Gambar 1 untuk validasi visual" if ann else "-"],
        ["Pipeline 2 (MediaPipe) menghasilkan ROI untuk semua citra", status(p2.get("detection_rate", 0) == 1.0,
                                                                            bool(p2)),
         f"dijalankan pada {p2.get('n', 0)} citra, ROI terbentuk pada {p2.get('detection_rate', 0):.0%}" if p2 else "-"],
        ["Pipeline 1 (YOLOv8n + U-Net) berjalan ujung ke ujung", status(bool(p1) and p1.get("detection_rate", 0) > 0,
                                                                       bool(p1)),
         f"ROI terbentuk pada {p1.get('detection_rate', 0):.0%} citra, Dice rata-rata {p1.get('mean_dice_all', 0):.3f}" if p1 else "-"],
        ["Evaluasi dua tingkat + uji McNemar tersedia", status(bool(res)), "reports/results.json" if res else "-"],
    ]
    p2b = (seg or {}).get("p2b_mediapipe_magictouch", {}).get("all", {})
    if p2b:
        targets.insert(4, ["Pipeline 2b (MediaPipe MagicTouch, tanpa pelatihan) untuk citra close-up",
                           status(p2b.get("detection_rate", 0) > 0, True),
                           f"ROI terbentuk pada {p2b['detection_rate']:.0%} citra, Dice rata-rata {p2b['mean_dice_all']:.3f}"])
    banner = ""
    if synthetic or smoke:
        banner = ('<div class="warn"><b>Uji fungsi kode.</b> Laporan ini memakai '
                  + ("data <b>sintetis</b>" if synthetic else "data asli")
                  + (" dan mode <b>smoke</b> (tanpa bobot pretrained, epoch sedikit)" if smoke else "")
                  + ". Angka di bawah hanya membuktikan bahwa seluruh tahap berjalan, "
                    "<b>bukan hasil penelitian</b> dan tidak boleh dikutip di skripsi.</div>")
    sections = [f"<h1>Laporan Progres Sistem Deteksi Anemia</h1>",
                f"<p class='meta'>Dibuat {dt.datetime.now():%d %B %Y %H:%M} &middot; commit <code>{git_commit()}</code> "
                f"&middot; sumber data <b>{html.escape((data or {}).get('source', '-'))}</b></p>", banner,
                "<h2>1. Status target minggu ke-11</h2>",
                table(targets, ["Target", "Status", "Bukti"], "status")]
    if data:
        sections += ["<h2>2. Data</h2>", table([[data["n_images"], data["n_anemia"], data["n_non_anemia"],
                                                  data["split"]["train"], data["split"]["val"], data["split"]["test"]]],
                                                ["Total citra", "Anemia", "Non-anemia", "Latih", "Validasi", "Uji"])]
        if data.get("skipped"):
            sections.append(f"<p>Dilewati: {html.escape('; '.join(data['skipped']))}</p>")
    sections += ["<h2>3. Anotasi</h2>", img_tag("anotasi_bbox_dari_mask.png",
                 "Gambar 1. Kotak YOLO (hijau) dibuat otomatis dari batas mask ahli (ungu).")]
    rows = []
    if yolo:
        rows.append(["YOLOv8n", yolo["epochs"], "ya" if yolo["pretrained"] else "tidak",
                     f"mAP50 uji {yolo['test_mAP50']:.3f}, recall {yolo['test_recall']:.3f}"])
    if unet:
        rows.append(["U-Net ResNet34", unet["epochs"], "ya" if unet["pretrained_encoder"] else "tidak",
                     f"Dice validasi terbaik {unet['best_val_dice']:.3f}"])
    if cls:
        for k, v in cls.items():
            rows.append([f"MobileNetV3-Small ({k})", v.get("epochs_run", "-"), "-", v.get("status", f"val loss {v.get('best_val_loss', '-')}")])
    sections += ["<h2>4. Pelatihan model</h2>", table(rows, ["Model", "Epoch", "Bobot pretrained", "Hasil"]),
                 img_tag("kurva_pelatihan_unet.png", "Gambar 2. Kurva pelatihan U-Net.")]
    sections += ["<h2>5. Hasil kedua pipeline</h2>",
                 img_tag("contoh_roi_kedua_pipeline.png",
                         "Gambar 3. Data uji: mask ahli (hijau), Pipeline 1 (ungu), Pipeline 2 Face Mesh, "
                         "Pipeline 2b MagicTouch (oranye).")]
    tbl = REP / "results_table.md"
    if tbl.exists():
        md = [l for l in tbl.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
        hdr = [c.strip() for c in md[0].strip("|").split("|")]
        body = [[c.strip() for c in l.strip("|").split("|")] for l in md[2:]]
        sections.append(table(body, hdr))
        extra = [l for l in tbl.read_text(encoding="utf-8").splitlines() if l.startswith("Uji McNemar")]
        sections += [f"<p>{html.escape(e)}</p>" for e in extra]
    if p2 and p2.get("detection_rate", 1) < 0.5:
        sections += ["<h2>Temuan yang perlu dibahas dengan pembimbing</h2>",
                     f"<p>MediaPipe hanya membentuk ROI pada {p2['detection_rate']:.0%} citra. Face Landmarker "
                     "membutuhkan wajah yang terlihat, sedangkan citra konjungtiva umumnya berupa foto close-up mata. "
                     "Perbandingan kedua pipeline baru adil bila citra uji memuat wajah, sehingga sumber data untuk "
                     "Pipeline 2 perlu diputuskan bersama pembimbing.</p>"
                     + ("<p>Sebagai alternatif, Pipeline 2b memakai MediaPipe Interactive Segmenter (MagicTouch) "
                        "dengan titik petunjuk dari landmark wajah atau dari puncak kemerahan pada citra close-up, "
                        "lalu disaring berdasarkan kanal a* CIELAB. Cara ini tetap tanpa pelatihan.</p>" if p2b else "")]
    sections += ["<h2>6. Cara mengulang</h2>",
                 "<pre>.venv\\Scripts\\python scripts\\run_pipeline.py            # data Eyes-defy-anemia\n"
                 ".venv\\Scripts\\python scripts\\run_pipeline.py --source synthetic --smoke   # uji fungsi</pre>"]
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#111827}"
           "h1{color:#3f35a8;margin-bottom:4px}h2{border-bottom:2px solid #e5e7eb;padding-bottom:4px;margin-top:28px}"
           ".meta{color:#6b7280}table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}"
           "th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f3f4f6}"
           ".warn{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:6px;margin:12px 0}"
           "figure{margin:12px 0}img{max-width:100%;border:1px solid #e5e7eb}figcaption{color:#4b5563;font-size:13px}"
           "pre{background:#f3f4f6;padding:10px;border-radius:6px;overflow-x:auto}"
           ".status td:nth-child(2){font-weight:600}")
    out = REP / "progress" / "laporan_progres.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html lang='id'><head><meta charset='utf-8'><title>Laporan Progres</title>"
                   f"<style>{css}</style></head><body>{''.join(sections)}</body></html>", encoding="utf-8")
    print(f"Laporan: {out}")
    for t in targets:
        print(f" - [{t[1]}] {t[0]}: {t[2]}")


if __name__ == "__main__":
    main()
