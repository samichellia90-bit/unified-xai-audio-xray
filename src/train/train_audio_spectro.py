"""Train a spectrogram CNN to separate bona fide from spoofed audio."""

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
from ..utils.audio_io import read_wav_bytes
from ..utils.audio_features import resample_if_needed, mel_spectrogram_image
from ..config import AUDIO_TARGET_SR

class AudioSpectroDataset(Dataset):
    """WAV files turned into normalized mel-spectrogram images."""
    def __init__(self, items):
        self.items = items
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        x, sr = read_wav_bytes(it.path.read_bytes())
        x, sr = resample_if_needed(x, sr, AUDIO_TARGET_SR)
        rgb = mel_spectrogram_image(x, sr)
        pil = Image.fromarray(rgb).convert("RGB")
        t = self.tf(pil)
        y = torch.tensor(it.label, dtype=torch.long)
        return t, y

def build_model(arch: str) -> nn.Module:
    """Build the selected torchvision architecture with a two-class head."""
    if arch == "vgg16":
        m = tvm.vgg16(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2); return m
    if arch == "resnet18":
        m = tvm.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2); return m
    if arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=None); m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2); return m
    raise ValueError("arch must be vgg16|resnet18|mobilenet_v3_small")

def main():
    """Train, keep the weights with the best validation accuracy and save the run metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data/audio folder (wav/ + labels.csv)")
    ap.add_argument("--arch", type=str, default="resnet18", choices=["vgg16","resnet18","mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--run_dir", type=str, default=None, help="Optional output folder for history/metrics")
    ap.add_argument("--seed", type=int, default=0, help="Seed of the weight initialisation and batch order")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    wav_root = data_dir / "wav"
    csv_path = data_dir / "labels.csv"
    items = read_labels_csv(csv_path, wav_root)
    tr, va = split_train_val(items, val_ratio=0.15, seed=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    tr_dl = DataLoader(AudioSpectroDataset(tr), batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(AudioSpectroDataset(va), batch_size=args.batch, shuffle=False, num_workers=0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"audio_{args.arch}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("weights/audio"); weights_dir.mkdir(parents=True, exist_ok=True)
    fname = {"vgg16":"vgg16_spec.pth", "resnet18":"resnet18_spec.pth", "mobilenet_v3_small":"mobilenetv3_spec.pth"}[args.arch]
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
        "task": "audio",
        "arch": args.arch,
        "best_val_acc": best,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "seed": args.seed,
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
