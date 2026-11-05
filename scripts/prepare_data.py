"""Langkah 1: menyusun metadata.csv dan pembagian data per subjek.

    python scripts/prepare_data.py                 # memakai dataset.source di config
    python scripts/prepare_data.py --source synthetic   # data uji fungsi
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset_loader import create_subject_splits, generate_synthetic_dataset
from src.utils import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["eyes_defy_anemia", "synthetic"])
    ap.add_argument("--n", type=int, default=80, help="jumlah sampel sintetis")
    args = ap.parse_args()
    cfg = load_config()
    source = args.source or cfg["dataset"]["source"]
    raw = ROOT / cfg["dataset"]["raw_dir"]

    if source == "eyes_defy_anemia":
        from src.preprocessing.eyes_defy_anemia import build_metadata
        df = build_metadata(ROOT / cfg["dataset"]["eyes_defy_anemia_dir"], cfg["labeling"], cfg["dataset"]["mask_name"])
        skipped = df.attrs.get("skipped", [])
    else:
        df = generate_synthetic_dataset(str(raw / "synthetic"), cfg["labeling"], num_samples=args.n,
                                        seed=cfg["experiment"]["seed"])
        skipped = []
    raw.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / "metadata.csv", index=False)
    d = cfg["dataset"]
    tr, va, te = create_subject_splits(df, str(ROOT / d["splits_dir"]), d["train_ratio"], d["val_ratio"],
                                       d["test_ratio"], cfg["experiment"]["seed"])
    summary = {
        "source": source,
        "n_images": int(len(df)),
        "n_anemia": int(df["anemia_label"].sum()),
        "n_non_anemia": int((df["anemia_label"] == 0).sum()),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "skipped": skipped,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
