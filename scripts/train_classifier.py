"""Langkah 5: melatih classifier MobileNetV3-Small dengan setelan identik untuk tiap sumber ROI.

Sumber ROI: gt (batas atas, mask ahli), p1_yolo_unet, p2_mediapipe.
Citra yang ROI-nya gagal dibuat oleh suatu pipeline tidak ikut dilatih/diuji untuk pipeline itu
dan dicatat sebagai kegagalan pada evaluasi.

    python scripts/train_classifier.py            # bobot ImageNet
    python scripts/train_classifier.py --smoke    # uji fungsi: tanpa bobot pretrained, epoch sedikit
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset_loader import load_split
from src.utils import ROOT, get_device, load_config, read_image, set_seed

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SOURCES = ["gt", "p1_yolo_unet", "p2_mediapipe"]


def build_mobilenet(pretrained: bool):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)  # 1000 kelas ImageNet -> 2 kelas
    return m


class RoiDataset(Dataset):
    def __init__(self, items, train, seed=0):
        self.items, self.train = items, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = read_image(path)
        if self.train:  # augmentasi sesuai Bab III: rotasi ±15°, flip, kecerahan ±15%
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), self.rng.uniform(-15, 15), 1.0)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (img.shape[1], img.shape[0]))
            img = np.clip(img.astype(np.float32) * self.rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), y


def items_for(df, roi_dir):
    out, missing = [], []
    for _, r in df.iterrows():
        p = roi_dir / f"{r['sample_id']}.png"
        (out if p.exists() else missing).append((str(p), int(r["anemia_label"])) if p.exists() else r["sample_id"])
    return out, missing


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Hitung ulang statistik BatchNorm dari data latih (tanpa augmentasi).

    Dengan data sedikit, running mean/var hasil momentum belum stabil sehingga keluaran mode eval
    bisa runtuh ke satu nilai. Rata-rata kumulatif (momentum=None) memperbaikinya.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for x, _ in loader:
        model(x.to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


def run_epoch(model, loader, device, opt=None, weights=None):
    train = opt is not None
    model.train(train)
    losses, probs, ys = [], [], []
    lossf = torch.nn.CrossEntropyLoss(weight=weights)
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = lossf(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss) * len(y))
            probs.append(torch.softmax(out, 1)[:, 1].detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    return sum(losses) / max(1, len(loader.dataset)), np.concatenate(probs) if probs else np.array([]), \
        np.concatenate(ys) if ys else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config()
    c = cfg["classifier"]
    device = get_device(cfg["experiment"]["device"])
    sd = str(ROOT / cfg["dataset"]["splits_dir"])
    splits = {s: load_split(sd, s) for s in ["train", "val", "test"]}
    rois = ROOT / cfg["dataset"]["processed_dir"] / "rois"
    epochs = args.epochs or (30 if args.smoke else c["epochs"])
    # Mode smoke dilatih dari nol (tanpa bobot ImageNet), jadi butuh laju belajar lebih besar dan tanpa early stop.
    lr = c["learning_rate"] * (3 if args.smoke else 1)
    patience = epochs if args.smoke else c["early_stopping_patience"]
    preds, summary = [], {}
    for src in SOURCES:
        set_seed(cfg["experiment"]["seed"])  # setelan identik untuk setiap sumber ROI
        tr, tr_miss = items_for(splits["train"], rois / src)
        va, _ = items_for(splits["val"], rois / src)
        te, te_miss = items_for(splits["test"], rois / src)
        summary[src] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                        "test_without_roi": len(te_miss), "train_without_roi": len(tr_miss)}
        if len(tr) < 4 or len({y for _, y in tr}) < 2 or not te:
            summary[src]["status"] = "dilewati: ROI terlalu sedikit untuk dilatih"
            for sid in splits["test"]["sample_id"]:
                preds.append({"sample_id": sid, "source": src, "has_roi": False, "prob_anemia": None, "pred": None})
            print(f"[{src}] dilewati ({summary[src]})")
            continue
        model = build_mobilenet(c["pretrained"] and not args.smoke).to(device)
        ytr = np.array([y for _, y in tr])
        w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2).clip(min=1)), dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
        tl = DataLoader(RoiDataset(tr, True, cfg["experiment"]["seed"]), batch_size=c["batch_size"], shuffle=True,
                        num_workers=0)
        vl = DataLoader(RoiDataset(va, False), batch_size=c["batch_size"], num_workers=0)
        cal = DataLoader(RoiDataset(tr, False), batch_size=c["batch_size"], num_workers=0)
        best, best_state, wait, log = 1e9, None, 0, []
        for ep in range(1, epochs + 1):
            tl_loss, _, _ = run_epoch(model, tl, device, opt, w)
            recalibrate_bn(model, cal, device)
            vl_loss, _, _ = run_epoch(model, vl, device, None, w) if va else (tl_loss, None, None)
            log.append({"epoch": ep, "train_loss": round(tl_loss, 4), "val_loss": round(vl_loss, 4)})
            if vl_loss < best:
                best, wait = vl_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        out_w = ROOT / "models" / f"mobilenetv3_{src}.pt"
        torch.save(model.state_dict(), out_w)
        pd.DataFrame(log).to_csv(ROOT / "reports" / f"classifier_log_{src}.csv", index=False)
        t0 = time.perf_counter()
        _, probs, ys = run_epoch(model, DataLoader(RoiDataset(te, False), batch_size=1), device)
        lat = (time.perf_counter() - t0) * 1000 / max(1, len(te))
        by_id = {Path(p).stem: pr for (p, _), pr in zip(te, probs)}
        for sid in splits["test"]["sample_id"]:
            pr = by_id.get(sid)
            preds.append({"sample_id": sid, "source": src, "has_roi": pr is not None,
                          "prob_anemia": None if pr is None else float(pr),
                          "pred": None if pr is None else int(pr >= c["decision_threshold"])})
        summary[src].update({"epochs_run": len(log), "best_val_loss": round(best, 4),
                             "classifier_latency_ms": round(lat, 2), "weights": str(out_w.relative_to(ROOT))})
        print(f"[{src}] {summary[src]}")
    pd.DataFrame(preds).to_csv(ROOT / "reports" / "classifier_test_predictions.csv", index=False)
    (ROOT / "reports" / "classifier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
