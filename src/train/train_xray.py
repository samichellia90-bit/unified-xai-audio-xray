"""Train a CNN to separate normal from pneumonia chest X-rays."""

from __future__ import annotations
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .common import read_labels_csv, split_train_val

class XRayDataset(Dataset):
    """Chest X-ray images resized to 224 x 224 and normalized.

    With `augment`, training images get a random crop, a small rotation and a
    small brightness and contrast change.
    """
    def __init__(self, items, augment: bool = False):
        self.items = items
        resize = [
            T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(7),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ] if augment else [T.Resize((224,224))]
        self.tf = T.Compose(resize + [
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        pil = Image.open(it.path).convert("RGB")
        x = self.tf(pil)
        y = torch.tensor(it.label, dtype=torch.long)
        return x, y

def build_model(arch: str, pretrained: bool = False) -> nn.Module:
    """Build AlexNet or DenseNet121 with a two-class head, optionally from ImageNet weights."""
    weights = "IMAGENET1K_V1" if pretrained else None
    if arch == "alexnet":
        m = tvm.alexnet(weights=weights)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
        return m
    if arch == "densenet121":
        m = tvm.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m
    raise ValueError("arch must be alexnet|densenet121")

def main():
    """Train, keep the weights with the best validation accuracy and save the run metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data/xray folder (images/ + labels.csv)")
    ap.add_argument("--arch", type=str, default="densenet121", choices=["alexnet","densenet121"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--run_dir", type=str, default=None, help="Optional output folder for history/metrics")
    ap.add_argument("--seed", type=int, default=0, help="Seed of the weight initialisation and batch order")
    ap.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights")
    ap.add_argument("--augment", action="store_true", help="Random crop, rotation and contrast on training images")
    ap.add_argument("--balanced_loss", action="store_true", help="Weight each class by the inverse of its frequency")
    ap.add_argument("--workers", type=int, default=0, help="Data loading processes")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    img_root = data_dir / "images"
    csv_path = data_dir / "labels.csv"
    items = read_labels_csv(csv_path, img_root)
    tr, va = split_train_val(items, val_ratio=0.15, seed=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.arch, pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weight = None
    if args.balanced_loss:
        counts = Counter(it.label for it in tr)
        class_weight = torch.tensor([len(tr) / (2 * counts[c]) for c in (0, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    tr_dl = DataLoader(XRayDataset(tr, augment=args.augment), batch_size=args.batch, shuffle=True, num_workers=args.workers)
    va_dl = DataLoader(XRayDataset(va), batch_size=args.batch, shuffle=False, num_workers=args.workers)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"image_{args.arch}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("weights/image"); weights_dir.mkdir(parents=True, exist_ok=True)
    fname = {"alexnet":"alexnet_xray.pth", "densenet121":"densenet121_xray.pth"}[args.arch]
    weights_path = weights_dir / fname

    best = 0.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f"train {epoch}/{args.epochs}")
        loss_sum, loss_cnt = 0.0, 0
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            loss_sum += float(loss.detach().cpu())
            loss_cnt += 1

        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pr = torch.argmax(model(xb), dim=-1)
                correct += int((pr == yb).sum().item())
                total += int(yb.numel())
        acc = correct / max(1, total)
        avg_loss = loss_sum / max(1, loss_cnt)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": acc})
        print(f"val acc: {acc:.4f}")

        if acc > best:
            best = acc
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)
            print(f"Saved best -> {weights_path}")

    def _count_labels(items):
        c = Counter([it.label for it in items])
        return {str(k): int(v) for k, v in c.items()}

    metrics = {
        "task": "image",
        "arch": args.arch,
        "best_val_acc": best,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "augment": args.augment,
        "balanced_loss": args.balanced_loss,
        "batch": args.batch,
        "lr": args.lr,
        "train_size": len(tr),
        "val_size": len(va),
        "train_label_counts": _count_labels(tr),
        "val_label_counts": _count_labels(va),
        "weights_path": str(weights_path),
        "run_dir": str(run_dir),
        "dataset_dir": str(data_dir),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Run saved -> {run_dir}")

if __name__ == "__main__":
    main()
