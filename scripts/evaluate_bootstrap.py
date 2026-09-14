#!/usr/bin/env python3
"""Consolidate every system, then quantify uncertainty with a hierarchical bootstrap.

Three things happen here, in order:

1. **Fusion.** The stage cascade routes Head 12 into Head 34; the tissue fusion
   combines M1 and M2 into the ML5 vector under the NC policy selected on
   validation (AND is preserved on ties).
2. **Point estimates.** Every system is scored on its own test identifiers, once
   per training seed, with a fixed label set and zero division mapped to zero.
3. **Uncertainty.** A hierarchical percentile bootstrap (B = 5,000, seed
   20260812) resamples training seeds *and* paired held-out identifiers, giving
   both 95% CIs per system and paired deltas between systems. An interval that
   contains zero is inconclusive evidence, not equality.

Usage:
    python scripts/evaluate_bootstrap.py --seeds 42,43,44 --bootstrap 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M1,
    M2,
    STAGE12,
    STAGE34,
    add_work_argument,
    parse_seeds,
    write_json,
)

BOOTSTRAP_SEED = 20260812


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def normalize_class(value: str) -> str:
    replacements = {
        "GRANULACAO": "GRANULAÇÃO",
        "NECROSE_SECA": "NECROSE SECA",
        "NAO_CLASSIFICAVEL": "NÃO CLASSIFICÁVEL",
    }
    return replacements.get(str(value), str(value))


def reorder_predictions(values: np.ndarray, source_names: np.ndarray, target_names: list[str]) -> np.ndarray:
    """Map argmax indices from a model's own class order into the canonical order."""
    source_pred = values.argmax(axis=1)
    names = [normalize_class(str(source_names[index])) for index in source_pred]
    return np.asarray([target_names.index(name) for name in names], dtype=np.int64)


def classification(true: np.ndarray, pred: np.ndarray, classes: int) -> dict[str, Any]:
    labels = list(range(classes))
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
        "f1_per_class": f1_score(true, pred, labels=labels, average=None, zero_division=0).tolist(),
        "support": np.bincount(true.astype(int), minlength=classes).tolist(),
    }


def multilabel(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "exact_match": float(accuracy_score(true, pred)),
        "hamming": float(hamming_loss(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(true, pred, average="micro", zero_division=0)),
        "f1_per_class": f1_score(true, pred, average=None, zero_division=0).tolist(),
        "positive_support": true.sum(axis=0).astype(int).tolist(),
    }


def joint_metrics(
    stage_true: np.ndarray, stage_pred: np.ndarray, tissue_true: np.ndarray, tissue_pred: np.ndarray
) -> dict[str, Any]:
    stage = classification(stage_true, stage_pred, 5)
    tissue = multilabel(tissue_true, tissue_pred)
    return {
        "stage_accuracy": stage["accuracy"],
        "stage_f1_macro": stage["f1_macro"],
        "stage_f1_per_class": stage["f1_per_class"],
        "stage_support": stage["support"],
        "tissue_exact_match": tissue["exact_match"],
        "tissue_hamming": tissue["hamming"],
        "tissue_f1_macro": tissue["f1_macro"],
        "tissue_f1_per_class": tissue["f1_per_class"],
        "tissue_positive_support": tissue["positive_support"],
        "avg_f1": (stage["f1_macro"] + tissue["f1_macro"]) / 2.0,
        "joint_exact": float(
            np.mean((stage_true == stage_pred) & np.all(tissue_true == tissue_pred, axis=1))
        ),
    }


def single_record(ids: np.ndarray, true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    ids = ids.astype(str)
    order = np.argsort(ids)
    return {"ids": ids[order], "y_true": true.astype(int)[order], "y_pred": pred.astype(int)[order]}


def joint_record(
    ids: np.ndarray,
    stage_true: np.ndarray,
    stage_pred: np.ndarray,
    tissue_true: np.ndarray,
    tissue_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = ids.astype(str)
    order = np.argsort(ids)
    return {
        "ids": ids[order],
        "stage_y_true": stage_true.astype(int)[order],
        "stage_y_pred": stage_pred.astype(int)[order],
        "tissue_y_true": tissue_true.astype(int)[order],
        "tissue_y_pred": tissue_pred.astype(int)[order],
    }


def assert_aligned(records: list[dict[str, np.ndarray]]) -> None:
    ids = records[0]["ids"].tolist()
    for record in records[1:]:
        if record["ids"].tolist() != ids:
            raise AssertionError("prediction IDs are not aligned across training seeds")


def classification_scalar(true: np.ndarray, pred: np.ndarray, classes: int) -> dict[str, float]:
    """NumPy equivalent of the scalar sklearn classification metrics.

    Used inside the bootstrap loop. The full, per-class result outside the loop
    continues to use sklearn as an independent implementation.
    """
    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    scores = np.zeros(classes, dtype=np.float64)
    support = np.bincount(true, minlength=classes).astype(np.float64)
    for label in range(classes):
        true_label = true == label
        pred_label = pred == label
        tp = np.count_nonzero(true_label & pred_label)
        fp = np.count_nonzero(~true_label & pred_label)
        fn = np.count_nonzero(true_label & ~pred_label)
        denominator = 2 * tp + fp + fn
        scores[label] = (2.0 * tp / denominator) if denominator else 0.0
    return {
        "accuracy": float(np.mean(true == pred)),
        "f1_macro": float(scores.mean()),
        "f1_weighted": float(np.dot(scores, support) / support.sum()),
    }


def multilabel_scalar(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """NumPy equivalent of the scalar sklearn multilabel metrics."""
    true = np.asarray(true, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tp = np.count_nonzero((true == 1) & (pred == 1), axis=0).astype(np.float64)
    fp = np.count_nonzero((true == 0) & (pred == 1), axis=0).astype(np.float64)
    fn = np.count_nonzero((true == 1) & (pred == 0), axis=0).astype(np.float64)
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    micro_denominator = float(2.0 * tp.sum() + fp.sum() + fn.sum())
    return {
        "exact_match": float(np.mean(np.all(true == pred, axis=1))),
        "hamming": float(np.mean(true != pred)),
        "f1_macro": float(per_class.mean()),
        "f1_micro": float(2.0 * tp.sum() / micro_denominator) if micro_denominator else 0.0,
    }


def scalar_metrics(kind: str, record: dict[str, np.ndarray], indexes: np.ndarray | None = None) -> dict[str, float]:
    take = slice(None) if indexes is None else indexes
    if kind in ("stage", "stage12", "stage34", "m1", "m2", "sl6"):
        classes = {"stage": 5, "stage12": 3, "stage34": 3, "m1": 4, "m2": 4, "sl6": 6}[kind]
        return classification_scalar(record["y_true"][take], record["y_pred"][take], classes)
    if kind == "tissue":
        return multilabel_scalar(record["y_true"][take], record["y_pred"][take])
    if kind == "joint":
        stage_true = record["stage_y_true"][take]
        stage_pred = record["stage_y_pred"][take]
        tissue_true = record["tissue_y_true"][take]
        tissue_pred = record["tissue_y_pred"][take]
        stage = classification_scalar(stage_true, stage_pred, 5)
        tissue = multilabel_scalar(tissue_true, tissue_pred)
        return {
            "stage_accuracy": stage["accuracy"],
            "stage_f1_macro": stage["f1_macro"],
            "tissue_exact_match": tissue["exact_match"],
            "tissue_hamming": tissue["hamming"],
            "tissue_f1_macro": tissue["f1_macro"],
            "avg_f1": (stage["f1_macro"] + tissue["f1_macro"]) / 2.0,
            "joint_exact": float(
                np.mean((stage_true == stage_pred) & np.all(tissue_true == tissue_pred, axis=1))
            ),
        }
    raise ValueError(kind)


def full_metrics(kind: str, record: dict[str, np.ndarray]) -> dict[str, Any]:
    if kind in ("stage", "stage12", "stage34", "m1", "m2", "sl6"):
        classes = {"stage": 5, "stage12": 3, "stage34": 3, "m1": 4, "m2": 4, "sl6": 6}[kind]
        return classification(record["y_true"], record["y_pred"], classes)
    if kind == "tissue":
        return multilabel(record["y_true"], record["y_pred"])
    if kind == "joint":
        return joint_metrics(
            record["stage_y_true"], record["stage_y_pred"], record["tissue_y_true"], record["tissue_y_pred"]
        )
    raise ValueError(kind)


def aggregate(
    name: str,
    kind: str,
    records: list[dict[str, np.ndarray]],
    seeds: list[int],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    assert_aligned(records)
    per_seed = [full_metrics(kind, record) for record in records]
    scalar = [scalar_metrics(kind, record) for record in records]
    metric_names = list(scalar[0])
    n = len(records[0]["ids"])
    distributions = {metric: np.empty(bootstrap, dtype=np.float64) for metric in metric_names}
    for replicate in range(bootstrap):
        sampled_seeds = rng.integers(0, len(records), len(records))
        sampled_ids = rng.integers(0, n, n)
        values = [scalar_metrics(kind, records[index], sampled_ids) for index in sampled_seeds]
        for metric in metric_names:
            distributions[metric][replicate] = np.mean([value[metric] for value in values])
    summary = {}
    for metric in metric_names:
        values = np.asarray([value[metric] for value in scalar])
        summary[metric] = {
            "mean": float(values.mean()),
            "sd_training_seeds": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_hierarchical": [float(value) for value in np.percentile(distributions[metric], [2.5, 97.5])],
            "by_seed": {str(seed): float(value) for seed, value in zip(seeds, values)},
        }
    return {
        "name": name,
        "kind": kind,
        "n_test": n,
        "training_seeds": seeds,
        "metrics": summary,
        "per_seed_full": {str(seed): value for seed, value in zip(seeds, per_seed)},
    }


def paired_delta(
    name: str,
    kind: str,
    left: list[dict[str, np.ndarray]],
    right: list[dict[str, np.ndarray]],
    metrics: list[str],
    seeds: list[int],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    assert_aligned(left + right)
    n = len(left[0]["ids"])
    point = []
    for lvalue, rvalue in zip(left, right):
        lm, rm = scalar_metrics(kind, lvalue), scalar_metrics(kind, rvalue)
        point.append({metric: lm[metric] - rm[metric] for metric in metrics})
    distributions = {metric: np.empty(bootstrap) for metric in metrics}
    for replicate in range(bootstrap):
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_ids = rng.integers(0, n, n)
        for metric in metrics:
            deltas = []
            for index in sampled_seeds:
                lm = scalar_metrics(kind, left[index], sampled_ids)[metric]
                rm = scalar_metrics(kind, right[index], sampled_ids)[metric]
                deltas.append(lm - rm)
            distributions[metric][replicate] = np.mean(deltas)
    return {
        "comparison": name,
        "left_minus_right": {
            metric: {
                "mean": float(np.mean([value[metric] for value in point])),
                "sd_training_seeds": (
                    float(np.std([value[metric] for value in point], ddof=1)) if len(point) > 1 else 0.0
                ),
                "ci95_hierarchical_paired": [
                    float(value) for value in np.percentile(distributions[metric], [2.5, 97.5])
                ],
                "by_seed": {str(seed): float(value[metric]) for seed, value in zip(seeds, point)},
            }
            for metric in metrics
        },
    }


class Loader:
    """Reads one round of runs from ``<work-dir>/runs``."""

    def __init__(self, runs: Path) -> None:
        self.runs = runs

    def aux(self, family: str, seed: int, exp_id: str, context: str) -> dict[str, np.ndarray]:
        return load_npz(self.runs / family / f"seed_{seed}" / exp_id / f"{context}_aux.npz")

    def stage_fusion(self, family: str, seed: int, context: str) -> dict[str, np.ndarray]:
        prefix = "effnet" if family == "timm" else "yolo"
        head12 = self.aux(family, seed, f"stage_head12_{prefix}", context)
        head34 = self.aux(family, seed, f"stage_head34_{prefix}", context)
        if head12["ids"].tolist() != head34["ids"].tolist():
            raise AssertionError("stage head IDs differ")
        pred12 = reorder_predictions(head12["probabilities"], head12["class_names"], STAGE12)
        pred34 = reorder_predictions(head34["probabilities"], head34["class_names"], STAGE34)
        fused = np.empty(len(pred12), dtype=np.int64)
        for index, (first, second) in enumerate(zip(pred12, pred34)):
            fused[index] = first if first in (0, 1) else second + 2
        return single_record(head12["ids"], head12["stage_y_true"], fused)

    def projected_stage_head(self, family: str, seed: int, head: int) -> dict[str, np.ndarray]:
        prefix = "effnet" if family == "timm" else "yolo"
        payload = self.aux(family, seed, f"stage_head{head}_{prefix}", "stage_test")
        targets = STAGE12 if head == 12 else STAGE34
        pred = reorder_predictions(payload["probabilities"], payload["class_names"], targets)
        global_true = payload["stage_y_true"].astype(int)
        if head == 12:
            true = np.where(global_true < 2, global_true, 2)
            ids = payload["ids"]
        else:
            selected = global_true >= 2
            true = global_true[selected] - 2
            pred = pred[selected]
            ids = payload["ids"][selected]
        return single_record(ids, true, pred)

    def tissue_fusion(self, family: str, seed: int, context: str, nc_policy: str = "and") -> dict[str, np.ndarray]:
        prefix = "effnet" if family == "timm" else "yolo"
        first = self.aux(family, seed, f"tissue_m1_{prefix}", context)
        second_exp = "tissue_m2_effnet_os" if family == "timm" else "tissue_m2_yolo_os"
        second = self.aux(family, seed, second_exp, context)
        if first["ids"].tolist() != second["ids"].tolist():
            raise AssertionError("tissue head IDs differ")
        pred1 = reorder_predictions(first["probabilities"], first["class_names"], M1)
        pred2 = reorder_predictions(second["probabilities"], second["class_names"], M2)
        output = np.zeros((len(pred1), 5), dtype=np.int32)
        output[:, 0] = np.isin(pred1, [0, 2])
        output[:, 1] = np.isin(pred1, [1, 2])
        output[:, 2] = np.isin(pred2, [1, 2])
        output[:, 3] = np.isin(pred2, [0, 2])
        nc_candidates = {
            "and": (pred1 == 3) & (pred2 == 3),
            "or": (pred1 == 3) | (pred2 == 3),
            "m1": pred1 == 3,
            "m2": pred2 == 3,
        }
        if nc_policy not in nc_candidates:
            raise ValueError(f"unknown NC fusion policy: {nc_policy}")
        output[:, 4] = nc_candidates[nc_policy]
        return single_record(first["ids"], first["tissue_y_true"], output)

    def select_nc_policy(self, family: str, seed: int) -> tuple[str, dict[str, float]]:
        policies = ["and", "or", "m1", "m2"]
        scores = {}
        for policy in policies:
            record = self.tissue_fusion(family, seed, "tissue_val", policy)
            scores[policy] = multilabel(record["y_true"], record["y_pred"])["f1_macro"]
        # Preserve the prespecified AND rule on ties.
        selected = max(policies, key=lambda policy: (scores[policy], -policies.index(policy)))
        return selected, scores

    def projected_tissue_head(self, family: str, seed: int, head: int) -> dict[str, np.ndarray]:
        prefix = "effnet" if family == "timm" else "yolo"
        exp_id = f"tissue_m{head}_{prefix}" + ("_os" if head == 2 else "")
        payload = self.aux(family, seed, exp_id, "tissue_test")
        target = M1 if head == 1 else M2
        pred = reorder_predictions(payload["probabilities"], payload["class_names"], target)
        values = payload["tissue_y_true"]
        if head == 1:
            true = np.where(
                (values[:, 0] == 1) & (values[:, 1] == 1),
                2,
                np.where(values[:, 0] == 1, 0, np.where(values[:, 1] == 1, 1, 3)),
            )
        else:
            true = np.where(
                (values[:, 2] == 1) & (values[:, 3] == 1),
                2,
                np.where(values[:, 3] == 1, 0, np.where(values[:, 2] == 1, 1, 3)),
            )
        return single_record(payload["ids"], true, pred)

    def classifier_from_file(self, family: str, seed: int, exp_id: str) -> dict[str, np.ndarray]:
        payload = load_npz(self.runs / family / f"seed_{seed}" / exp_id / "test_predictions.npz")
        return single_record(payload["ids"], payload["y_true"], payload["y_pred"])

    def multilabel_from_file(self, seed: int, exp_id: str) -> dict[str, np.ndarray]:
        run = self.runs / "timm" / f"seed_{seed}" / exp_id
        selected = run / "test_predictions_calibrated.npz"
        payload = load_npz(selected if selected.exists() else run / "test_predictions.npz")
        return single_record(payload["ids"], payload["y_true"], payload["y_pred"])

    def multitask_from_file(self, seed: int, exp_id: str) -> dict[str, np.ndarray]:
        run = self.runs / "timm" / f"seed_{seed}" / exp_id
        selected = run / "test_predictions_calibrated.npz"
        payload = load_npz(selected if selected.exists() else run / "test_predictions.npz")
        return joint_record(
            payload["ids"],
            payload["stage_y_true"],
            payload["stage_y_pred"],
            payload["tissue_y_true"],
            payload["tissue_y_pred"],
        )

    def modular(self, family: str, seed: int, nc_policy: str) -> dict[str, np.ndarray]:
        stage = self.stage_fusion(family, seed, "joint_test")
        tissue = self.tissue_fusion(family, seed, "joint_test", nc_policy)
        if stage["ids"].tolist() != tissue["ids"].tolist():
            raise AssertionError("modular stage/tissue IDs differ")
        return joint_record(stage["ids"], stage["y_true"], stage["y_pred"], tissue["y_true"], tissue["y_pred"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap", type=int, default=5000)
    add_work_argument(parser)
    parser.add_argument("--out", type=Path, default=None, help="default: <work-dir>/results/results_master.json")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    load = Loader(args.work_dir / "runs")
    selected_nc = {
        family: {seed: load.select_nc_policy(family, seed) for seed in seeds}
        for family in ("timm", "yolo")
    }

    systems: dict[str, tuple[str, list[dict[str, np.ndarray]]]] = {
        "stage_head12_effnet": ("stage12", [load.projected_stage_head("timm", seed, 12) for seed in seeds]),
        "stage_head34_effnet": ("stage34", [load.projected_stage_head("timm", seed, 34) for seed in seeds]),
        "stage_head12_yolo": ("stage12", [load.projected_stage_head("yolo", seed, 12) for seed in seeds]),
        "stage_head34_yolo": ("stage34", [load.projected_stage_head("yolo", seed, 34) for seed in seeds]),
        "stage_cascade_effnet": ("stage", [load.stage_fusion("timm", seed, "stage_test") for seed in seeds]),
        "stage_cascade_yolo": ("stage", [load.stage_fusion("yolo", seed, "stage_test") for seed in seeds]),
        "stage_mono_effnet": ("stage", [load.classifier_from_file("timm", seed, "stage_mono_effnet") for seed in seeds]),
        "stage_mono_yolo": ("stage", [load.classifier_from_file("yolo", seed, "stage_mono_yolo") for seed in seeds]),
        "stage_mono_densenet121": ("stage", [load.classifier_from_file("timm", seed, "stage_mono_densenet121") for seed in seeds]),
        "stage_mono_resnet18": ("stage", [load.classifier_from_file("timm", seed, "stage_mono_resnet18") for seed in seeds]),
        "tissue_m1_effnet": ("m1", [load.projected_tissue_head("timm", seed, 1) for seed in seeds]),
        "tissue_m2_effnet": ("m2", [load.projected_tissue_head("timm", seed, 2) for seed in seeds]),
        "tissue_m1_yolo": ("m1", [load.projected_tissue_head("yolo", seed, 1) for seed in seeds]),
        "tissue_m2_yolo": ("m2", [load.projected_tissue_head("yolo", seed, 2) for seed in seeds]),
        "tissue_fusion_effnet": ("tissue", [load.tissue_fusion("timm", seed, "tissue_test", selected_nc["timm"][seed][0]) for seed in seeds]),
        "tissue_fusion_yolo": ("tissue", [load.tissue_fusion("yolo", seed, "tissue_test", selected_nc["yolo"][seed][0]) for seed in seeds]),
        "tissue_fusion_effnet_and": ("tissue", [load.tissue_fusion("timm", seed, "tissue_test", "and") for seed in seeds]),
        "tissue_fusion_yolo_and": ("tissue", [load.tissue_fusion("yolo", seed, "tissue_test", "and") for seed in seeds]),
        "tissue_mono_effnet_ml5": ("tissue", [load.multilabel_from_file(seed, "tissue_mono_effnet_ml5") for seed in seeds]),
        "tissue_mono_yolo_sl6": ("sl6", [load.classifier_from_file("yolo", seed, "tissue_mono_yolo_sl6") for seed in seeds]),
        "multitask_effnet": ("joint", [load.multitask_from_file(seed, "multitask_effnet") for seed in seeds]),
        "multitask_densenet121": ("joint", [load.multitask_from_file(seed, "multitask_densenet121") for seed in seeds]),
        "modular_effnet": ("joint", [load.modular("timm", seed, selected_nc["timm"][seed][0]) for seed in seeds]),
        "modular_yolo": ("joint", [load.modular("yolo", seed, selected_nc["yolo"][seed][0]) for seed in seeds]),
    }
    aggregated = {
        name: aggregate(name, kind, records, seeds, args.bootstrap, rng)
        for name, (kind, records) in systems.items()
    }
    paired = [
        paired_delta("EffNet stage cascade - monolith", "stage", systems["stage_cascade_effnet"][1], systems["stage_mono_effnet"][1], ["accuracy", "f1_macro"], seeds, args.bootstrap, rng),
        paired_delta("YOLO stage cascade - monolith", "stage", systems["stage_cascade_yolo"][1], systems["stage_mono_yolo"][1], ["accuracy", "f1_macro"], seeds, args.bootstrap, rng),
        paired_delta("EffNet tissue monolith - fusion", "tissue", systems["tissue_mono_effnet_ml5"][1], systems["tissue_fusion_effnet"][1], ["exact_match", "f1_macro"], seeds, args.bootstrap, rng),
        paired_delta("EffNet validation-selected NC fusion - AND", "tissue", systems["tissue_fusion_effnet"][1], systems["tissue_fusion_effnet_and"][1], ["exact_match", "f1_macro"], seeds, args.bootstrap, rng),
        paired_delta("YOLO validation-selected NC fusion - AND", "tissue", systems["tissue_fusion_yolo"][1], systems["tissue_fusion_yolo_and"][1], ["exact_match", "f1_macro"], seeds, args.bootstrap, rng),
        paired_delta("EffNet multitask - DenseNet multitask", "joint", systems["multitask_effnet"][1], systems["multitask_densenet121"][1], ["avg_f1", "joint_exact"], seeds, args.bootstrap, rng),
        paired_delta("EffNet multitask - YOLO modular", "joint", systems["multitask_effnet"][1], systems["modular_yolo"][1], ["avg_f1", "joint_exact"], seeds, args.bootstrap, rng),
        paired_delta("EffNet multitask - EffNet modular", "joint", systems["multitask_effnet"][1], systems["modular_effnet"][1], ["avg_f1", "joint_exact"], seeds, args.bootstrap, rng),
    ]
    payload = {
        "round": "seed_2",
        "training_seeds": seeds,
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": BOOTSTRAP_SEED,
            "method": "hierarchical percentile bootstrap: resample training seeds and paired test IDs",
        },
        "detection": "reported separately on its own partition; never combined with classification",
        "nc_fusion_selection": {
            family: {
                str(seed): {"selected": policy, "validation_f1_by_policy": scores}
                for seed, (policy, scores) in values.items()
            }
            for family, values in selected_nc.items()
        },
        "systems": aggregated,
        "paired_deltas": paired,
    }
    out = args.out or (args.work_dir / "results/results_master.json")
    write_json(out, payload)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
