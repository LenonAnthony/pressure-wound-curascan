#!/usr/bin/env python3
"""Produce the aligned auxiliary predictions the fusions and modular systems need.

The cascade and the tissue fusion combine two heads on the *same* identifiers, so
each head must be run over every evaluation context, not only over its own test
folder. This script writes one ``<context>_aux.npz`` per head, seed and context,
carrying ids, group ids, ground truth and the full probability matrix.

Contexts: ``stage_test``, ``tissue_val``, ``tissue_test``, ``joint_test``.

Usage:
    python scripts/infer.py --family timm --seeds 42,43,44
    python scripts/infer.py --family yolo --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M1,
    M2,
    STAGE5,
    STAGE12,
    STAGE34,
    ImageIndex,
    add_index_argument,
    add_work_argument,
    manifest_items,
    parse_seeds,
)

TIMM_JOBS = [
    ("stage_head12_effnet", 3, STAGE12, ["stage_test", "joint_test"]),
    ("stage_head34_effnet", 3, STAGE34, ["stage_test", "joint_test"]),
    ("tissue_m1_effnet", 4, M1, ["tissue_val", "tissue_test", "joint_test"]),
    ("tissue_m2_effnet_os", 4, M2, ["tissue_val", "tissue_test", "joint_test"]),
]
YOLO_JOBS = [
    ("stage_head12_yolo", ["stage_test", "joint_test"]),
    ("stage_head34_yolo", ["stage_test", "joint_test"]),
    ("tissue_m1_yolo", ["tissue_val", "tissue_test", "joint_test"]),
    ("tissue_m2_yolo_os", ["tissue_val", "tissue_test", "joint_test"]),
]


def contexts(index: ImageIndex) -> dict[str, list[dict[str, Any]]]:
    return {
        "stage_test": index.resolve(manifest_items("stage", "test")),
        "tissue_val": index.resolve(manifest_items("tissue", "val")),
        "tissue_test": index.resolve(manifest_items("tissue", "test")),
        "joint_test": index.resolve(manifest_items("multitask", "test")),
    }


def save(path: Path, items: list[dict[str, Any]], probabilities: np.ndarray, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray([str(item.get("id") or item.get("name") or item["identity"]) for item in items]),
        group_ids=np.asarray([item["group_id"] for item in items]),
        stage_y_true=np.asarray(
            [STAGE5.index(item["label_5"]) if item.get("label_5") in STAGE5 else -1 for item in items]
        ),
        tissue_y_true=np.asarray([item.get("tissue", [0, 0, 0, 0, 0]) for item in items], dtype=np.int32),
        probabilities=probabilities,
        y_pred=probabilities.argmax(axis=1),
        class_names=np.asarray(classes),
    )


def run_timm(runs: Path, seeds: list[int], data: dict[str, list[dict[str, Any]]]) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    import train_timm as training

    class Images(Dataset):
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self.items = items
            self.transform = training.build_transform(224, False)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, position: int) -> "torch.Tensor":
            return self.transform(Image.open(self.items[position]["path"]).convert("RGB"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def predict(model: "torch.nn.Module", items: list[dict[str, Any]]) -> np.ndarray:
        loader = DataLoader(Images(items), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        model.eval()
        values = []
        for images in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            values.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(values)

    for seed in seeds:
        for exp_id, outputs, classes, names in TIMM_JOBS:
            run = runs / "timm" / f"seed_{seed}" / exp_id
            checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
            model = training.build_backbone(checkpoint["model_name"], outputs, pretrained=False).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            for name in names:
                items = data[name]
                save(run / f"{name}_aux.npz", items, predict(model, items), classes)
                print(f"[{seed}:{exp_id}:{name}] n={len(items)}", flush=True)
            del model
            torch.cuda.empty_cache()


def run_yolo(runs: Path, seeds: list[int], data: dict[str, list[dict[str, Any]]]) -> None:
    from ultralytics import YOLO

    def predict(model: "YOLO", items: list[dict[str, Any]]) -> np.ndarray:
        paths = [item["path"] for item in items]
        probabilities = []
        for start in range(0, len(paths), 32):
            results = model.predict(paths[start : start + 32], imgsz=224, batch=32, verbose=False)
            probabilities.extend(result.probs.data.detach().cpu().numpy() for result in results)
        return np.asarray(probabilities)

    for seed in seeds:
        for exp_id, names in YOLO_JOBS:
            run = runs / "yolo" / f"seed_{seed}" / exp_id
            model = YOLO(str(run / "best.pt"))
            classes = [model.names[index] for index in range(len(model.names))]
            for name in names:
                items = data[name]
                save(run / f"{name}_aux.npz", items, predict(model, items), classes)
                print(f"[{seed}:{exp_id}:{name}] n={len(items)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=["timm", "yolo", "both"], default="both")
    parser.add_argument("--seeds", default="42,43,44")
    add_index_argument(parser)
    add_work_argument(parser)
    args = parser.parse_args()

    index = ImageIndex.load(args.index)
    data = contexts(index)
    runs = args.work_dir / "runs"
    seeds = parse_seeds(args.seeds)
    if args.family in ("timm", "both"):
        run_timm(runs, seeds, data)
    if args.family in ("yolo", "both"):
        run_yolo(runs, seeds, data)


if __name__ == "__main__":
    main()
