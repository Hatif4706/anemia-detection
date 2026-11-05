"""Langkah 6: evaluasi dua tingkat dan uji statistik.

Tingkat 1 (segmentasi): dari reports/segmentation_summary.json.
Tingkat 2 (diagnostik): confusion matrix, accuracy, sensitivity, specificity, precision, F1, ROC-AUC,
selang kepercayaan bootstrap 95%, cakupan (persentase citra uji yang punya ROI), dan uji McNemar
P1 vs P2 pada citra uji yang berhasil diproses kedua pipeline.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import EvaluationMetrics, bootstrap_ci, mcnemar_exact
from src.preprocessing.dataset_loader import load_split
from src.utils import ROOT, git_commit, load_config


def main():
    cfg = load_config()
    rep = ROOT / "reports"
    test = load_split(str(ROOT / cfg["dataset"]["splits_dir"]), "test")[["sample_id", "anemia_label"]]
    preds = pd.read_csv(rep / "classifier_test_predictions.csv").merge(test, on="sample_id")
    data = json.loads((rep / "data_summary.json").read_text(encoding="utf-8"))
    out = {"data_source": data["source"], "git_commit": git_commit(),
           "segmentation": json.loads((rep / "segmentation_summary.json").read_text(encoding="utf-8")),
           "classification": {}, "comparison": {}}
    for src, g in preds.groupby("source"):
        ok = g[g["has_roi"] == True]  # noqa: E712
        entry = {"n_test": int(len(g)), "coverage": round(float(len(ok) / max(1, len(g))), 4)}
        if len(ok) and ok["pred"].notna().all():
            y, p, s = ok["anemia_label"].values, ok["pred"].astype(int).values, ok["prob_anemia"].values
            m = EvaluationMetrics.calculate_classification_metrics(y, p, np.c_[1 - s, s])
            entry.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
            entry["ci95"] = bootstrap_ci(y, p, s, cfg["evaluation"]["bootstrap_iterations"], cfg["evaluation"]["alpha"])
        out["classification"][src] = entry
    piv = preds.pivot(index="sample_id", columns="source", values="pred")
    for other, key in [("p2_mediapipe", "p1_vs_p2_mcnemar"), ("p2b_mediapipe_magictouch", "p1_vs_p2b_mcnemar")]:
        if {"p1_yolo_unet", other} <= set(piv.columns):
            both = piv.dropna(subset=["p1_yolo_unet", other]).join(test.set_index("sample_id"))
            out["comparison"][key] = (
                {"n_paired": int(len(both)), **mcnemar_exact(both["anemia_label"], both["p1_yolo_unet"].astype(int),
                                                             both[other].astype(int))}
                if len(both) else {"n_paired": 0, "note": "tidak ada citra uji yang diproses kedua pipeline"})
    (rep / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Ringkasan tabel Markdown untuk laporan
    lines = [f"Sumber data: **{data['source']}**  |  commit `{out['git_commit']}`", "",
             "| Pipeline | Deteksi ROI (uji) | Dice (uji) | Latensi ROI (ms) | Cakupan klasifikasi | Akurasi | Sensitivity | Specificity | F1 | ROC-AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"gt": "Mask ahli (batas atas)", "p1_yolo_unet": "Pipeline 1: YOLOv8n + U-Net",
             "p2_mediapipe": "Pipeline 2: MediaPipe Face Mesh", "p2b_mediapipe_magictouch": "Pipeline 2b: MediaPipe MagicTouch"}
    for src in [s for s in names if s in out["classification"] or s in out["segmentation"]]:
        c = out["classification"].get(src, {})
        s = out["segmentation"].get(src, {}).get("test", {})
        f = lambda k: "-" if c.get(k) is None else "{:.3f}".format(c[k])
        det = "-" if not s else "{:.0%}".format(s["detection_rate"])
        dice = "-" if not s else "{:.3f}".format(s["mean_dice_all"])
        lat = "-" if not s else str(s["mean_latency_ms"])
        cov = "{:.0%}".format(c.get("coverage", 0))
        lines.append(f"| {names[src]} | {det} | {dice} | {lat} | {cov} | {f('accuracy')} | {f('sensitivity')} | "
                     f"{f('specificity')} | {f('f1_score')} | {f('roc_auc')} |")
    for key, label in [("p1_vs_p2_mcnemar", "P1 vs P2"), ("p1_vs_p2b_mcnemar", "P1 vs P2b")]:
        mc = out["comparison"].get(key)
        if mc:
            lines += ["", f"Uji McNemar {label}: n berpasangan = {mc['n_paired']}"
                      + (f", p = {mc['p_value']:.4f}" if "p_value" in mc else f" ({mc.get('note')})")]
    (rep / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
