"""Label file reading and train/validation split."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import List, Tuple

@dataclass
class Item:
    """File path with its integer label."""
    path: Path
    label: int

def read_labels_csv(csv_path: Path, root: Path) -> List[Item]:
    """Read a labels.csv file with filename and label columns."""
    items: List[Item] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row["filename"].strip()
            lab = int(row["label"])
            items.append(Item(path=root / fn, label=lab))
    return items

def split_train_val(items: List[Item], val_ratio: float=0.15, seed: int=0) -> Tuple[List[Item], List[Item]]:
    """Shuffle with a fixed seed and hold out a validation fraction."""
    import random
    rnd = random.Random(seed)
    arr = items[:]
    rnd.shuffle(arr)
    n_val = max(1, int(len(arr)*val_ratio))
    return arr[n_val:], arr[:n_val]
