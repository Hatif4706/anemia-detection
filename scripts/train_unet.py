"""Langkah 3b: melatih U-Net (encoder ResNet34) pada potongan area konjungtiva.

Masukan latih = potongan di sekitar kotak mask dengan padding acak (meniru kotak YOLO),
target = mask ahli pada potongan yang sama. Loss = Dice + BCE.

    python scripts/train_unet.py            # encoder pretrained ImageNet
    python scripts/train_unet.py --smoke    # uji fungsi: tanpa bobot pretrained, epoch sedikit
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import EvaluationMetrics
from src.preprocessing.dataset_loader import load_split
from src.segmentation.unet_segmenter import build_unet, to_tensor
from src.utils import ROOT, apply_clahe, get_device, load_config, mask_bbox, pad_box, read_image, read_mask, set_seed


class CropDataset(Dataset):
    """Menyimpan hanya area di sekitar konjungtiva yang sudah diperkecil.

    Foto Eyes-defy-anemia berukuran ~12 MP; memuat semuanya utuh ke RAM butuh ~9 GB.
    Area yang disimpan = kotak mask + margin (cukup untuk padding acak + geseran), lalu sisi
    terpanjangnya dibatasi `max_side` piksel. Ini tidak mengubah apa yang dilihat U-Net karena
    masukannya memang diperkecil ke 256x256.
    """

    def __init__(self, df, size, pad_range, clahe, train, max_side=768):
        self.items, self.size, self.pad_range, self.train = [], size, pad_range, train
        margin = pad_range[1] + 0.2
        for _, r in df.iterrows():
            img = read_image(r["image_path"])
            m = read_mask(r["mask_path"], img.shape)
            box = mask_bbox(m)
            if box is None:
                continue
            x1, y1, x2, y2 = pad_box(box, margin, m.shape[1], m.shape[0])
            ci, cm = img[y1:y2, x1:x2], m[y1:y2, x1:x2]
            s = min(1.0, max_side / max(ci.shape[:2]))
            if s < 1.0:
                ci = cv2.resize(ci, (int(ci.shape[1] * s), int(ci.shape[0] * s)), interpolation=cv2.INTER_AREA)
                cm = cv2.resize(cm, (ci.shape[1], ci.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.items.append((apply_clahe(ci) if clahe else ci.copy(), cm.copy()))
            del img, m
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        h, w = m.shape
        pad = self.rng.uniform(*self.pad_range) if self.train else float(np.mean(self.pad_range))
        x1, y1, x2, y2 = pad_box(mask_bbox(m), pad, w, h)
        if self.train:  # geser kotak sedikit, seperti prediksi YOLO yang tidak sempurna
            dx, dy = (self.rng.normal(0, 0.04, 2) * [x2 - x1, y2 - y1]).astype(int)
            x1, x2 = np.clip([x1 + dx, x2 + dx], 0, w)
            y1, y2 = np.clip([y1 + dy, y2 + dy], 0, h)
        ci, cm = img[y1:y2, x1:x2], m[y1:y2, x1:x2]
        if self.train:
            if self.rng.random() < 0.5:
                ci, cm = ci[:, ::-1], cm[:, ::-1]
            ang = self.rng.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((ci.shape[1] / 2, ci.shape[0] / 2), ang, 1.0)
            ci = cv2.warpAffine(np.ascontiguousarray(ci), M, (ci.shape[1], ci.shape[0]), borderMode=cv2.BORDER_REFLECT)
            cm = cv2.warpAffine(np.ascontiguousarray(cm), M, (cm.shape[1], cm.shape[0]), flags=cv2.INTER_NEAREST)
            ci = cv2.convertScaleAbs(ci, alpha=float(self.rng.uniform(0.85, 1.15)), beta=0)  # tanpa salinan float32
        x = to_tensor(np.ascontiguousarray(ci), self.size)
        y = cv2.resize(np.ascontiguousarray(cm), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return x, torch.from_numpy((y > 0).astype(np.float32))[None]


def dice_bce(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def evaluate(model, loader, device):
    model.eval()
    dices, losses = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lo = model(x)
            losses.append(float(dice_bce(lo, y)))
            pred = (torch.sigmoid(lo) > 0.5).cpu().numpy()
            for p, t in zip(pred, y.cpu().numpy()):
                dices.append(EvaluationMetrics.calculate_segmentation_metrics(p[0], t[0])["dice_score"])
    return float(np.mean(losses)), float(np.mean(dices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config()
    u = cfg["unet_segmenter"]
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    sd = str(ROOT / cfg["dataset"]["splits_dir"])
    clahe = cfg["preprocessing"]["clahe_for_segmentation"]
    tr = CropDataset(load_split(sd, "train"), u["input_size"], u["train_crop_padding"], clahe, True)
    va = CropDataset(load_split(sd, "val"), u["input_size"], u["train_crop_padding"], clahe, False)
    tl = DataLoader(tr, batch_size=u["batch_size"], shuffle=True, num_workers=0, drop_last=len(tr) > u["batch_size"])
    vl = DataLoader(va, batch_size=u["batch_size"], shuffle=False, num_workers=0)

    model = build_unet(u["encoder_name"], None if args.smoke else u["encoder_weights"], u["in_channels"], u["classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=u["learning_rate"], weight_decay=1e-4)
    epochs = args.epochs or (25 if args.smoke else u["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    out = ROOT / u["weights_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "reports" / "unet_training_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    best = -1.0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for ep in range(1, epochs + 1):
            model.train()
            tot = []
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = dice_bce(model(x), y)
                loss.backward()
                opt.step()
                tot.append(float(loss))
            sched.step()
            vloss, vdice = evaluate(model, vl, device)
            w.writerow([ep, round(np.mean(tot), 5), round(vloss, 5), round(vdice, 5)])
            if vdice > best:
                best = vdice
                torch.save(model.state_dict(), out)
            print(f"epoch {ep:3d}  train_loss {np.mean(tot):.4f}  val_loss {vloss:.4f}  val_dice {vdice:.4f}")
    summary = {"epochs": epochs, "pretrained_encoder": not args.smoke, "best_val_dice": best,
               "n_train": len(tr), "n_val": len(va), "weights": str(out.relative_to(ROOT))}
    (ROOT / "reports" / "unet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
