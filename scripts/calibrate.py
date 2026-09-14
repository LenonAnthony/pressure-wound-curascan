#!/usr/bin/env python3
"""Select multi-label thresholds on validation data and freeze the test outputs.

One threshold per tissue attribute is chosen by grid search over [0.05, 0.95] in
0.01 steps, maximizing validation F1; ties resolve to the value closest to 0.5.
The test set is never consulted. The NC fusion policy (AND / OR / M1 / M2) is
selected the same way inside ``evaluate_bootstrap.py``, also on validation only.

Usage:
    python scripts/calibrate.py --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_timm as training  # noqa: E402
from common import (  # noqa: E402
    TISSUE5,
    ImageIndex,
    add_index_argument,
    add_work_argument,
    parse_seeds,
    write_json,
)


def thresholds(true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    grid = np.linspace(0.05, 0.95, 91)
    selected = []
    details = []
    for column in range(true.shape[1]):
        scores = np.asarray(
            [
                f1_score(true[:, column], probabilities[:, column] >= threshold, zero_division=0)
                for threshold in grid
            ]
        )
        best = scores.max()
        candidates = np.flatnonzero(np.isclose(scores, best))
        index = min(candidates, key=lambda value: (abs(grid[value] - 0.5), grid[value]))
        selected.append(grid[index])
        details.append(
            {
                "threshold": float(grid[index]),
                "validation_f1": float(best),
                "positive_support": int(true[:, column].sum()),
            }
        )
    return np.asarray(selected), details


def save_calibrated(
    output: Path,
    ids: list[str],
    true: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    prefix: str = "",
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "ids": np.asarray(ids),
        f"{prefix}y_true": true.astype(np.int32),
        f"{prefix}y_pred": (probabilities >= selected[None, :]).astype(np.int32),
        f"{prefix}probabilities": probabilities,
        "tissue_thresholds": selected,
    }
    if extra:
        arrays.update(extra)
    np.savez_compressed(output, **arrays)


def calibration_record(details: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "selection_split": "validation",
        "method": "per-attribute grid search over [0.05, 0.95] in 0.01 steps maximizing "
        "validation F1; ties resolve closest to 0.5",
        "attributes": dict(zip(TISSUE5, details)),
        "fixed_threshold_baseline": 0.5,
    }


def calibrate_monolith(runs: Path, seed: int, device: torch.device, index: ImageIndex) -> None:
    run = runs / "timm" / f"seed_{seed}" / "tissue_mono_effnet_ml5"
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = training.build_backbone(checkpoint["model_name"], 5, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    source = training.load_items("tissue", index)
    results = {}
    for split in ("val", "test"):
        dataset = training.ManifestDataset(source[split], "tissue_ml", False, seed=seed)
        results[split] = training.predict_ml(model, training.make_loader(dataset, 32, 4, False, seed), device)
    selected, details = thresholds(results["val"]["y_true"], results["val"]["probabilities"])
    ids = [str(item.get("name") or item.get("id") or item["identity"]) for item in source["test"]]
    save_calibrated(
        run / "test_predictions_calibrated.npz",
        ids,
        results["test"]["y_true"],
        results["test"]["probabilities"],
        selected,
    )
    write_json(run / "calibration.json", calibration_record(details))


def calibrate_multitask(runs: Path, seed: int, exp_id: str, device: torch.device, index: ImageIndex) -> None:
    run = runs / "timm" / f"seed_{seed}" / exp_id
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = training.MultiTaskModel(checkpoint["model_name"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    source = training.load_items("multitask", index)
    results = {}
    for split in ("val", "test"):
        dataset = training.ManifestDataset(source[split], "multitask", False, seed=seed)
        loader = training.make_loader(dataset, 32, 4, False, seed)
        results[split] = training.predict_multitask(model, loader, device)
    validation, test = results["val"], results["test"]
    selected, details = thresholds(validation["tissue_y_true"], validation["tissue_probabilities"])
    ids = [str(item.get("id") or item["identity"]) for item in source["test"]]
    save_calibrated(
        run / "test_predictions_calibrated.npz",
        ids,
        test["tissue_y_true"],
        test["tissue_probabilities"],
        selected,
        prefix="tissue_",
        extra={
            "stage_y_true": test["stage_y_true"],
            "stage_y_pred": test["stage_y_pred"],
            "stage_probabilities": test["stage_probabilities"],
        },
    )
    write_json(run / "calibration.json", calibration_record(details))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="42,43,44")
    add_index_argument(parser)
    add_work_argument(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = ImageIndex.load(args.index)
    runs = args.work_dir / "runs"
    for seed in parse_seeds(args.seeds):
        calibrate_monolith(runs, seed, device, index)
        calibrate_multitask(runs, seed, "multitask_effnet", device, index)
        calibrate_multitask(runs, seed, "multitask_densenet121", device, index)
        print(f"seed={seed} calibrated", flush=True)


if __name__ == "__main__":
    main()
