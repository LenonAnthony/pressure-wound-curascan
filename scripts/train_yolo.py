#!/usr/bin/env python3
"""Train the YOLO26s models: six classification runs per seed, plus the detector.

Classification (``--task classify``, the default) trains the six folder datasets
built by ``build_yolo_datasets.py``: 6 experiments x 3 seeds = 18 runs.

Detection (``--task detect``) fine-tunes ``yolo26s.pt`` on the wound bounding
boxes. That model is reported on its own held-out partition only: its split is
not aligned with the classification outputs and it was not repeated across the
three seeds, so its metrics never enter a classification claim.

Usage:
    python scripts/train_yolo.py --seeds 42,43,44
    python scripts/train_yolo.py --task detect --weights yolo26s.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONFIGS, add_work_argument, parse_seeds, read_json, write_json  # noqa: E402

CLASSIFY_PLAN = [
    ("stage_head12_yolo", "stage_head12"),
    ("stage_head34_yolo", "stage_head34"),
    ("stage_mono_yolo", "stage_mono5"),
    ("tissue_m1_yolo", "tissue_m1"),
    ("tissue_m2_yolo_os", "tissue_m2_os"),
    ("tissue_mono_yolo_sl6", "tissue_mono_sl6"),
]


def classification_metrics(true: np.ndarray, pred: np.ndarray, classes: int) -> dict[str, Any]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for expected, observed in zip(true, pred):
        matrix[int(expected), int(observed)] += 1
    scores = []
    support = matrix.sum(axis=1)
    for index in range(classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float(2 * true_positive / denominator) if denominator else 0.0)
    values = np.asarray(scores)
    return {
        "accuracy": float(np.mean(true == pred)),
        "f1_macro": float(values.mean()),
        "f1_weighted": float(np.sum(values * support) / support.sum()),
        "f1_per_class": scores,
        "confusion_matrix": matrix.tolist(),
    }


def mapping_for(data: Path, dataset: Path, split: str = "test") -> dict[str, dict[str, Any]]:
    mapping_path = dataset / "mapping.json"
    if not mapping_path.exists() and dataset.name.startswith("tissue_m2_os_seed_"):
        mapping_path = data / "tissue_m2" / "mapping.json"
    payload = read_json(mapping_path)
    return {
        Path(item["relative_path"]).name: item
        for item in payload["items"]
        if item["split"] == split
    }


@torch.inference_mode()
def infer_files(model: YOLO, files: list[Path], batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    for start in range(0, len(files), batch):
        results = model.predict(
            [str(path) for path in files[start : start + batch]],
            imgsz=224,
            batch=batch,
            verbose=False,
        )
        for result in results:
            predictions.append(int(result.probs.top1))
            probabilities.append(result.probs.data.detach().cpu().numpy())
    return np.asarray(predictions), np.asarray(probabilities)


def evaluate(model: YOLO, data: Path, dataset: Path, split: str = "test") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names = model.names
    class_names = [names[index] for index in range(len(names))]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    mapping = mapping_for(data, dataset, split)
    files: list[Path] = []
    true: list[int] = []
    ids: list[str] = []
    for class_name in class_names:
        directory = dataset / split / class_name
        for path in sorted(directory.iterdir() if directory.exists() else []):
            if not path.is_file():
                continue
            files.append(path)
            true.append(class_to_index[class_name])
            metadata = mapping.get(path.name)
            ids.append(metadata["id"] if metadata else path.name)
    pred, probabilities = infer_files(model, files)
    y_true = np.asarray(true)
    metrics = {
        **classification_metrics(y_true, pred, len(class_names)),
        "class_names": class_names,
        "n": len(y_true),
    }
    arrays = {
        "ids": np.asarray(ids),
        "y_true": y_true,
        "y_pred": pred,
        "probabilities": probabilities,
    }
    return metrics, arrays


def best_epoch(results_csv: Path) -> tuple[int | None, int]:
    if not results_csv.exists():
        return None, 0
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    if not rows:
        return None, 0
    key = next((value for value in rows[0] if "accuracy_top1" in value), None)
    if key is None:
        return len(rows), len(rows)
    index = max(range(len(rows)), key=lambda row: float(rows[row].get(key) or 0.0))
    return int(float(rows[index].get("epoch", index + 1))), len(rows)


def train_one(
    data: Path,
    runs: Path,
    exp_id: str,
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = data / (f"tissue_m2_os_seed_{seed}" if dataset_name == "tissue_m2_os" else dataset_name)
    output = runs / f"seed_{seed}" / exp_id
    if (output / "metrics.json").exists() and not args.force:
        print(f"[{seed}:{exp_id}] already complete")
        return read_json(output / "metrics.json")
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = YOLO(str(args.weights))
    epochs = min(args.epochs, 3) if args.smoke else args.epochs
    patience = min(args.patience, 2) if args.smoke else args.patience
    started = time.time()
    model.train(
        data=str(dataset.resolve()),
        epochs=epochs,
        patience=patience,
        batch=args.batch,
        workers=args.workers,
        imgsz=224,
        seed=seed,
        deterministic=True,
        project=str(output),
        name="train",
        exist_ok=True,
        pretrained=True,
        plots=False,
        verbose=False,
    )
    best = output / "train/weights/best.pt"
    last = output / "train/weights/last.pt"
    selected = best if best.exists() else last
    trained = YOLO(str(selected))
    metrics, arrays = evaluate(trained, data, dataset)
    np.savez_compressed(output / "test_predictions.npz", **arrays)
    epoch, stopped = best_epoch(output / "train/results.csv")
    shutil.copy2(selected, output / "best.pt")
    payload = {
        "round": "seed_2",
        "seed": seed,
        "exp_id": exp_id,
        "dataset": dataset_name,
        "model": "yolo26s-cls",
        "best_epoch": epoch,
        "stopped_epoch": stopped,
        "seconds": time.time() - started,
        "protocol": {
            "image_size": 224,
            "batch": args.batch,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "seed": seed,
            "optimizer": "Ultralytics auto",
            "initial_weights": str(args.weights),
        },
        "test": metrics,
    }
    write_json(output / "metrics.json", payload)
    print(f"[{seed}:{exp_id}] DONE epoch={epoch} test_f1={metrics['f1_macro']:.4f}", flush=True)
    return payload


def train_detector(runs: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune the wound detector; reported only on its own partition."""
    config = read_json(CONFIGS / "hyperparameters.json")["detector"]
    output = runs / "detect"
    output.mkdir(parents=True, exist_ok=True)
    data_yaml = args.data or (args.work_dir / "yolo_datasets/detector/data.yaml")
    if not Path(data_yaml).exists():
        raise SystemExit(
            f"{data_yaml} not found; run build_yolo_datasets.py first, or pass --data"
        )
    model = YOLO(str(args.weights))
    started = time.time()
    model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=min(config["epochs"], 3) if args.smoke else config["epochs"],
        batch=config["batch"],
        imgsz=config["image_size"],
        seed=config["seed"],
        workers=args.workers,
        deterministic=True,
        project=str(output),
        name="train",
        exist_ok=True,
        pretrained=True,
        plots=False,
        verbose=False,
    )
    validated = model.val(split="test", project=str(output), name="val", exist_ok=True)
    payload = {
        "task": "detect",
        "protocol": {**config, "initial_weights": str(args.weights)},
        "seconds": time.time() - started,
        "test": {
            "precision": float(validated.box.mp),
            "recall": float(validated.box.mr),
            "map50": float(validated.box.map50),
            "map50_95": float(validated.box.map),
        },
        "note": "own held-out partition; not aligned with the classification outputs",
    }
    write_json(output / "metrics.json", payload)
    print(json.dumps(payload["test"], indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["classify", "detect"], default="classify")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--only", default="", help="comma-separated exp_id subset")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="initial checkpoint (default: yolo26s-cls.pt for classify, yolo26s.pt for detect)",
    )
    parser.add_argument("--data", type=Path, default=None, help="detector data.yaml override")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    add_work_argument(parser)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = Path("yolo26s-cls.pt" if args.task == "classify" else "yolo26s.pt")
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("CUDA is required for the full round; use --smoke for a dry run")

    runs = args.work_dir / "runs/yolo"
    if args.task == "detect":
        train_detector(runs, args)
        return

    data = args.work_dir / "yolo_datasets"
    only = {value.strip() for value in args.only.split(",") if value.strip()}
    summary = []
    for seed in parse_seeds(args.seeds):
        for exp_id, dataset in CLASSIFY_PLAN:
            if only and exp_id not in only:
                continue
            result = train_one(data, runs, exp_id, dataset, seed, args)
            summary.append(
                {
                    "seed": seed,
                    "exp_id": exp_id,
                    "best_epoch": result.get("best_epoch"),
                    "test": result.get("test"),
                    "seconds": result.get("seconds"),
                }
            )
    write_json(runs / "summary.json", summary)


if __name__ == "__main__":
    main()
