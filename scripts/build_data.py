#!/usr/bin/env python3
"""Rebuild the leakage-controlled manifests from raw images and label catalogues.

This is the script that produced ``manifests/*_split.json``. It is published so
the audit can be re-executed, not because a reproduction needs it: the frozen
manifests in ``manifests/`` are the canonical partition of the paper. Given the
same input catalogue the search is deterministic and rebuilds that partition
exactly (verified: identical digest set, identical split per digest, identical
group assignment). Given a *different* catalogue it produces a different,
equally valid partition -- and a different test set -- so numbers computed that
way are not comparable with the published ones.

Input: two catalogues listing every candidate image with its label, in the
format described in ``--help`` and in the README. Output: the same six JSON
documents shipped in ``manifests/``.

The grouping rule is deliberately conservative because a patient dictionary was
not available: filenames starting with ``<digits>_`` share the numeric prefix,
and copy suffixes such as ``(2)`` and ``__dup2`` are collapsed into the same
case-level proxy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M1,
    M2,
    MANIFESTS,
    SPLITS,
    STAGE5,
    TISSUE5,
    TISSUE_SL6,
    read_json,
    sha256_file,
    write_json,
)

TARGET = np.asarray([0.80, 0.15, 0.05], dtype=np.float64)
SEARCH_SEED = 20260812


def perceptual_hash(path: str) -> int:
    image = Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(image, dtype=np.float64), type=2, norm="ortho")[:8, :8]
    median = np.median(coefficients.ravel()[1:])
    value = 0
    for bit in (coefficients > median).ravel():
        value = (value << 1) | int(bit)
    return value


def resized_gray(path: str) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((64, 64), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )


def identity(value: str) -> str:
    stem = Path(value).stem if Path(value).suffix else value
    stem = re.sub(r"(?<=\S)\(", " (", stem)
    return re.sub(r"\s+", " ", stem).strip().lower()


def provisional_group(value: str) -> str:
    stem = identity(value)
    stem = re.sub(r"__dup\d+$", "", stem)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"[-_ ]+(?:copy|copia|dup(?:licate)?)(?:[-_ ]*\d+)?$", "", stem)
    numeric = re.match(r"^(\d+)_", stem)
    if numeric:
        return f"clinical:{numeric.group(1)}"
    return f"name:{stem}"


def tissue_key(item: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(float(value) != 0.0) for value in item["tissue"])


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if b < a:
            a, b = b, a
        self.parent[b] = a


def deduplicate(
    items: list[dict[str, Any]],
    label: Callable[[dict[str, Any]], Any],
    id_key: str,
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for original in items:
        item = dict(original)
        item["content_sha256"] = sha256_file(Path(item["path"]))
        item["identity"] = identity(str(item.get(id_key) or item.get("stem") or ""))
        item["provisional_group"] = provisional_group(item["identity"])
        by_hash[item["content_sha256"]].append(item)

    kept: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for digest, group in sorted(by_hash.items()):
        labels = {label(item) for item in group}
        record = {
            "sha256": digest,
            "n": len(group),
            "ids": [item["identity"] for item in group],
            "labels": [str(value) for value in sorted(labels, key=str)],
        }
        if len(labels) > 1:
            conflicts.append(record)
            continue
        # Prefer the shortest stable identifier; source split is intentionally ignored.
        representative = min(
            group,
            key=lambda item: (len(item["identity"]), item["identity"], str(item["path"])),
        )
        kept.append(representative)
        if len(group) > 1:
            duplicate_groups.append(record)

    audit = {
        "task": task,
        "n_input": len(items),
        "n_output": len(kept),
        "n_exact_duplicate_groups": len(duplicate_groups),
        "n_redundant_files_removed": sum(item["n"] - 1 for item in duplicate_groups),
        "n_conflicting_hash_groups_excluded": len(conflicts),
        "n_conflicting_files_excluded": sum(item["n"] for item in conflicts),
        "duplicate_groups": duplicate_groups,
        "conflicting_groups": conflicts,
    }
    return kept, audit


def deduplicate_perceptual(
    items: list[dict[str, Any]],
    label: Callable[[dict[str, Any]], Any],
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse recompressed/resized copies missed by exact SHA-256.

    A distance up to four bits in a 64-bit pHash is treated as equivalent. A
    narrow rescue criterion (distance <=10 and mean 64x64 grayscale error <=10)
    captures the same image with a changed crop/encoding without merging merely
    similar wound photographs.
    """
    values = [perceptual_hash(item["path"]) for item in items]
    dsu = DisjointSet()
    for index in range(len(items)):
        dsu.find(str(index))
    gray_cache: dict[int, np.ndarray] = {}
    edges: list[dict[str, Any]] = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            distance = (values[left] ^ values[right]).bit_count()
            equivalent = distance <= 4
            mean_error = None
            if not equivalent and distance <= 10:
                gray_cache.setdefault(left, resized_gray(items[left]["path"]))
                gray_cache.setdefault(right, resized_gray(items[right]["path"]))
                mean_error = float(np.mean(np.abs(gray_cache[left] - gray_cache[right])))
                equivalent = mean_error <= 10.0
            if equivalent:
                dsu.union(str(left), str(right))
                edges.append(
                    {
                        "left": items[left]["identity"],
                        "right": items[right]["identity"],
                        "hamming": distance,
                        "mean_gray_error": mean_error,
                    }
                )
    components: dict[str, list[int]] = defaultdict(list)
    for index in range(len(items)):
        components[dsu.find(str(index))].append(index)

    kept: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for indexes in components.values():
        group = [items[index] for index in indexes]
        labels = {label(item) for item in group}
        record = {
            "n": len(group),
            "ids": [item["identity"] for item in group],
            "labels": [str(value) for value in sorted(labels, key=str)],
            "phash": [f"{values[index]:016x}" for index in indexes],
        }
        if len(labels) > 1:
            conflicts.append(record)
            continue
        representative = min(
            group,
            key=lambda item: (len(item["identity"]), item["identity"], str(item["path"])),
        )
        representative["perceptual_hash"] = f"{perceptual_hash(representative['path']):016x}"
        kept.append(representative)
        if len(group) > 1:
            duplicates.append(record)
    return kept, {
        "task": task,
        "n_input": len(items),
        "n_output": len(kept),
        "criteria": "pHash Hamming <=4, or Hamming <=10 with mean 64x64 grayscale error <=10",
        "n_candidate_edges": len(edges),
        "n_duplicate_groups": len(duplicates),
        "n_redundant_images_removed": sum(item["n"] - 1 for item in duplicates),
        "n_conflicting_groups_excluded": len(conflicts),
        "n_conflicting_images_excluded": sum(item["n"] for item in conflicts),
        "edges": edges,
        "duplicate_groups": duplicates,
        "conflicting_groups": conflicts,
    }


def merge_cross_task_groups(
    stage: list[dict[str, Any]], tissue: list[dict[str, Any]]
) -> dict[str, Any]:
    """Unify provisional groups that share exact content across task catalogues."""
    dsu = DisjointSet()
    by_hash: dict[str, list[str]] = defaultdict(list)
    for item in stage + tissue:
        group = item["provisional_group"]
        dsu.find(group)
        by_hash[item["content_sha256"]].append(group)
    for values in by_hash.values():
        for value in values[1:]:
            dsu.union(values[0], value)
    perceptual_edges = []
    for task, items in (("stage", stage), ("tissue", tissue)):
        hashes = [int(item["perceptual_hash"], 16) for item in items]
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                distance = (hashes[left] ^ hashes[right]).bit_count()
                if distance <= 8:
                    dsu.union(items[left]["provisional_group"], items[right]["provisional_group"])
                    perceptual_edges.append(
                        {
                            "task": task,
                            "left": items[left]["identity"],
                            "right": items[right]["identity"],
                            "hamming": distance,
                        }
                    )
    for item in stage + tissue:
        item["group_id"] = dsu.find(item["provisional_group"])
    return {
        "threshold": "64-bit pHash Hamming <=8",
        "n_edges": len(perceptual_edges),
        "edges": perceptual_edges,
    }


def m1_label(values: list[float]) -> str:
    h, g = bool(values[0]), bool(values[1])
    if h and g:
        return "AMBAS"
    if h:
        return "HIPEREMIA"
    if g:
        return "GRANULAÇÃO"
    return "NÃO CLASSIFICÁVEL"


def m2_label(values: list[float]) -> str:
    e, n = bool(values[2]), bool(values[3])
    if e and n:
        return "AMBAS"
    if n:
        return "NECROSE SECA"
    if e:
        return "ESFACELO"
    return "NÃO CLASSIFICÁVEL"


def tissue_sl6_label(values: list[float]) -> str:
    positives = [name for name, value in zip(TISSUE5[:4], values[:4]) if value]
    if len(positives) >= 2:
        return "MULTI"
    if positives:
        return positives[0]
    return "NÃO CLASSIFICÁVEL"


def build_multitask(
    stage: list[dict[str, Any]], tissue: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_by_identity = {item["identity"]: item for item in stage}
    stage_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in stage:
        stage_by_hash[item["content_sha256"]].append(item)

    result: list[dict[str, Any]] = []
    matched_hash = 0
    ambiguous_hash = 0
    for tissue_item in tissue:
        stage_item = stage_by_identity.get(tissue_item["identity"])
        match_kind = "identity"
        if stage_item is None:
            candidates = stage_by_hash.get(tissue_item["content_sha256"], [])
            labels = {candidate["label_5"] for candidate in candidates}
            if len(candidates) == 1 or (candidates and len(labels) == 1):
                stage_item = sorted(candidates, key=lambda value: value["identity"])[0]
                match_kind = "sha256"
                matched_hash += 1
            elif candidates:
                ambiguous_hash += 1
        if stage_item is None:
            continue
        # Both inputs belong to the same split, even if a cross-task alias was used.
        group_id = min(stage_item["group_id"], tissue_item["group_id"])
        result.append(
            {
                "id": tissue_item.get("name") or tissue_item["identity"],
                "identity": tissue_item["identity"],
                "group_id": group_id,
                "path": tissue_item["path"],
                "stage_path": stage_item["path"],
                "content_sha256": tissue_item["content_sha256"],
                "stage_content_sha256": stage_item["content_sha256"],
                "stage": stage_item["label_5"],
                "label_5": stage_item["label_5"],
                "label_12": stage_item["label_12"],
                "label_34": stage_item["label_34"],
                "tissue": [int(value) for value in tissue_item["tissue"]],
                "match_kind": match_kind,
            }
        )
    return result, {
        "n": len(result),
        "matched_by_sha256": matched_hash,
        "ambiguous_sha256_not_used": ambiguous_hash,
    }


def feature_names() -> list[str]:
    names = ["stage:n"] + [f"stage:{name}" for name in STAGE5]
    names += ["tissue:n"] + [f"tissue:{name}" for name in TISSUE5]
    names += [f"m1:{name}" for name in M1]
    names += [f"m2:{name}" for name in M2]
    names += [f"tissue-sl6:{name}" for name in TISSUE_SL6]
    names += ["joint:n"] + [f"joint-stage:{name}" for name in STAGE5]
    names += [f"joint-tissue:{name}" for name in TISSUE5]
    return names


def group_features(
    groups: list[str],
    stage: list[dict[str, Any]],
    tissue: list[dict[str, Any]],
    multitask: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    names = feature_names()
    index = {name: col for col, name in enumerate(names)}
    group_index = {name: row for row, name in enumerate(groups)}
    matrix = np.zeros((len(groups), len(names)), dtype=np.float64)

    for item in stage:
        row = group_index[item["group_id"]]
        matrix[row, index["stage:n"]] += 1
        matrix[row, index[f"stage:{item['label_5']}"]] += 1
    for item in tissue:
        row = group_index[item["group_id"]]
        matrix[row, index["tissue:n"]] += 1
        for attr, value in zip(TISSUE5, item["tissue"]):
            matrix[row, index[f"tissue:{attr}"]] += int(value)
        matrix[row, index[f"m1:{m1_label(item['tissue'])}"]] += 1
        matrix[row, index[f"m2:{m2_label(item['tissue'])}"]] += 1
        matrix[row, index[f"tissue-sl6:{tissue_sl6_label(item['tissue'])}"]] += 1
    for item in multitask:
        row = group_index[item["group_id"]]
        matrix[row, index["joint:n"]] += 1
        matrix[row, index[f"joint-stage:{item['label_5']}"]] += 1
        for attr, value in zip(TISSUE5, item["tissue"]):
            matrix[row, index[f"joint-tissue:{attr}"]] += int(value)
    return matrix, names


def assignment_score(counts: np.ndarray, totals: np.ndarray, names: list[str]) -> float:
    target = TARGET[:, None] * totals[None, :]
    relative = (counts - target) / (target + 1.0)
    weights = np.ones(len(names), dtype=np.float64)
    for key in ("stage:n", "tissue:n", "joint:n"):
        weights[names.index(key)] = 4.0
    score = float(np.mean((relative**2) * weights[None, :]))

    # Require representation when the feature is sufficiently frequent. The
    # penalty is finite so the script can still produce a split if impossible.
    for col, total in enumerate(totals):
        if total > 0 and counts[0, col] == 0:
            score += 500.0
        # With at least three observations, one example can in principle be
        # retained in every partition. This particularly protects the rare
        # ``AMBAS`` classes in M1/M2 from silently disappearing at evaluation.
        if total >= 3 and counts[1, col] == 0:
            score += 100.0
        if total >= 3 and counts[2, col] == 0:
            score += 200.0
    for row in range(3):
        if counts[row, names.index("stage:n")] == 0:
            score += 1000.0
        if counts[row, names.index("tissue:n")] == 0:
            score += 1000.0
        if counts[row, names.index("joint:n")] == 0:
            score += 1000.0
    return score


def find_assignment(
    groups: list[str], matrix: np.ndarray, names: list[str], trials: int
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(SEARCH_SEED)
    totals = matrix.sum(axis=0)
    best: np.ndarray | None = None
    best_score = float("inf")

    # Random search is intentionally independent of the training seeds. Whole
    # groups are sampled, and the best multi-objective class balance is frozen.
    for _ in range(trials):
        assignment = rng.choice(3, size=len(groups), p=TARGET)
        counts = np.stack([matrix[assignment == split].sum(axis=0) for split in range(3)])
        score = assignment_score(counts, totals, names)
        if score < best_score:
            best_score = score
            best = assignment.copy()
    assert best is not None
    return best, best_score


def counts_by_split(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        split: dict(Counter(item[key] for item in items if item["split"] == split))
        for split in SPLITS
    }


def verify_no_leakage(*collections: list[dict[str, Any]]) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for items in collections:
        for item in items:
            group_splits[item["group_id"]].add(item["split"])
            hash_splits[item["content_sha256"]].add(item["split"])
    group_leaks = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    hash_leaks = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    perceptual_leaks = []
    for items in collections[:2]:
        hashes = [int(item["perceptual_hash"], 16) for item in items]
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                distance = (hashes[left] ^ hashes[right]).bit_count()
                if distance <= 8 and items[left]["split"] != items[right]["split"]:
                    perceptual_leaks.append(
                        {
                            "left": items[left]["identity"],
                            "right": items[right]["identity"],
                            "hamming": distance,
                            "splits": [items[left]["split"], items[right]["split"]],
                        }
                    )
    return {
        "group_leak_count": len(group_leaks),
        "exact_content_leak_count": len(hash_leaks),
        "perceptual_leak_count": len(perceptual_leaks),
        "group_leaks": group_leaks,
        "exact_content_leaks": hash_leaks,
        "perceptual_leaks": perceptual_leaks,
    }


def public_record(item: dict[str, Any], fields: list[str], root: Path | None) -> dict[str, Any]:
    record = {key: item[key] for key in fields if key in item}
    record["file_name"] = Path(item["path"]).name
    if root is not None:
        try:
            record["source_relative_path"] = str(Path(item["path"]).resolve().relative_to(root))
        except ValueError:
            record["source_relative_path"] = Path(item["path"]).name
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage-catalogue",
        type=Path,
        required=True,
        help='JSON with {"items": [{"id", "path", "label_5"}, ...]}',
    )
    parser.add_argument(
        "--tissue-catalogue",
        type=Path,
        required=True,
        help='JSON with {"items": [{"name", "path", "tissue": [H,G,E,N,NC]}, ...]}',
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MANIFESTS / "rebuilt",
        help="where to write the rebuilt manifests (default: manifests/rebuilt, "
        "so the frozen published manifests are never overwritten by accident)",
    )
    parser.add_argument(
        "--path-root",
        type=Path,
        default=None,
        help="optional base directory recorded as source_relative_path",
    )
    parser.add_argument("--trials", type=int, default=100000)
    args = parser.parse_args()

    root = args.path_root.resolve() if args.path_root else None
    source_stage = read_json(args.stage_catalogue)["items"]
    source_tissue = read_json(args.tissue_catalogue)["items"]
    stage, stage_audit = deduplicate(source_stage, lambda item: item["label_5"], "id", "stage")
    tissue, tissue_audit = deduplicate(source_tissue, tissue_key, "name", "tissue")
    stage, stage_perceptual_audit = deduplicate_perceptual(
        stage, lambda item: item["label_5"], "stage"
    )
    tissue, tissue_perceptual_audit = deduplicate_perceptual(tissue, tissue_key, "tissue")
    for item in tissue:
        item["tissue"] = [int(float(value) != 0.0) for value in item["tissue"]]

    perceptual_group_audit = merge_cross_task_groups(stage, tissue)
    multitask, multitask_audit = build_multitask(stage, tissue)

    groups = sorted({item["group_id"] for item in stage + tissue + multitask})
    matrix, names = group_features(groups, stage, tissue, multitask)
    assignment, score = find_assignment(groups, matrix, names, args.trials)
    split_for_group = {group: SPLITS[int(value)] for group, value in zip(groups, assignment)}
    for items in (stage, tissue, multitask):
        for item in items:
            item["split"] = split_for_group[item["group_id"]]

    # Recreate projected labels after the clean split.
    for item in stage:
        stage_name = item["label_5"]
        item["label_12"] = stage_name if stage_name in STAGE5[:2] else "nao_classificavel"
        item["label_34"] = stage_name if stage_name in STAGE5[2:] else None
        item["in_head_34"] = item["label_34"] is not None

    leakage = verify_no_leakage(stage, tissue, multitask)
    if (
        leakage["group_leak_count"]
        or leakage["exact_content_leak_count"]
        or leakage["perceptual_leak_count"]
    ):
        raise RuntimeError(f"split leakage detected: {leakage}")

    for item in tissue:
        item["m1"] = m1_label(item["tissue"])
        item["m2"] = m2_label(item["tissue"])
        item["sl6"] = tissue_sl6_label(item["tissue"])
    for item in multitask:
        item["m1"] = m1_label(item["tissue"])
        item["m2"] = m2_label(item["tissue"])

    stage_fields = [
        "id", "identity", "content_sha256", "perceptual_hash", "group_id", "split",
        "label_5", "label_12", "label_34", "in_head_34",
    ]
    tissue_fields = [
        "name", "identity", "content_sha256", "perceptual_hash", "group_id", "split",
        "tissue", "m1", "m2", "sl6",
    ]
    multitask_fields = [
        "id", "identity", "content_sha256", "stage_content_sha256", "group_id", "split",
        "stage", "label_5", "label_12", "label_34", "tissue", "m1", "m2", "match_kind",
    ]

    stage_sorted = sorted(stage, key=lambda item: (item["split"], item["identity"]))
    tissue_sorted = sorted(tissue, key=lambda item: (item["split"], item["identity"]))
    multitask_sorted = sorted(multitask, key=lambda item: (item["split"], item["identity"]))
    head34 = [item for item in stage_sorted if item["in_head_34"]]

    stage_payload = {
        "round": "rebuilt",
        "pool": "stage",
        "split_search_seed": SEARCH_SEED,
        "resolution": "match local files to records by content_sha256",
        "classes_5": STAGE5,
        "n_total": len(stage),
        "n_by_split": {split: sum(item["split"] == split for item in stage) for split in SPLITS},
        "counts_5": counts_by_split(stage, "label_5"),
        "counts_12": counts_by_split(stage, "label_12"),
        "head_34": {
            "n_total": len(head34),
            "n_by_split": {
                split: sum(item["split"] == split for item in head34) for split in SPLITS
            },
            "counts": counts_by_split(head34, "label_34"),
        },
        "items": [public_record(item, stage_fields, root) for item in stage_sorted],
    }
    tissue_payload = {
        "round": "rebuilt",
        "pool": "tissue",
        "split_search_seed": SEARCH_SEED,
        "resolution": "match local files to records by content_sha256",
        "classes": TISSUE5,
        "n_total": len(tissue),
        "n_by_split": {split: sum(item["split"] == split for item in tissue) for split in SPLITS},
        "positive_counts": {
            split: {
                attr: sum(int(item["tissue"][col]) for item in tissue if item["split"] == split)
                for col, attr in enumerate(TISSUE5)
            }
            for split in SPLITS
        },
        "m1_counts": counts_by_split(tissue, "m1"),
        "m2_counts": counts_by_split(tissue, "m2"),
        "sl6_counts": counts_by_split(tissue, "sl6"),
        "items": [public_record(item, tissue_fields, root) for item in tissue_sorted],
    }
    multitask_payload = {
        "round": "rebuilt",
        "pool": "multitask",
        "split_search_seed": SEARCH_SEED,
        "resolution": "match local files to records by content_sha256 (tissue image)",
        "n_total": len(multitask),
        "n_by_split": {
            split: sum(item["split"] == split for item in multitask) for split in SPLITS
        },
        "counts_stage_by_split": counts_by_split(multitask, "label_5"),
        "positive_counts": {
            split: {
                attr: sum(int(item["tissue"][col]) for item in multitask if item["split"] == split)
                for col, attr in enumerate(TISSUE5)
            }
            for split in SPLITS
        },
        "items": [public_record(item, multitask_fields, root) for item in multitask_sorted],
    }

    audit = {
        "round": "rebuilt",
        "grouping_rule": {
            "numeric_prefix": "filenames beginning <digits>_ share the leading integer",
            "copy_suffixes": "trailing (N), __dupN, copy/copia/duplicate suffixes are collapsed",
            "interpretation": "conservative case-level proxy; patient IDs were unavailable",
        },
        "deduplication": {
            "stage_exact": stage_audit,
            "tissue_exact": tissue_audit,
            "stage_perceptual": stage_perceptual_audit,
            "tissue_perceptual": tissue_perceptual_audit,
        },
        "perceptual_grouping": perceptual_group_audit,
        "multitask_matching": multitask_audit,
        "split": {
            "search_seed": SEARCH_SEED,
            "trials": args.trials,
            "target_fractions": dict(zip(SPLITS, TARGET.tolist())),
            "n_groups": len(groups),
            "groups_by_split": dict(Counter(split_for_group.values())),
            "objective": score,
            "feature_names": names,
        },
        "leakage": leakage,
    }
    summary = {
        "round": "rebuilt",
        "training_seeds_planned": [42, 43, 44],
        "stage": {
            "n_total": len(stage),
            "n_by_split": stage_payload["n_by_split"],
            "counts": stage_payload["counts_5"],
        },
        "tissue": {
            "n_total": len(tissue),
            "n_by_split": tissue_payload["n_by_split"],
            "positive_counts": tissue_payload["positive_counts"],
        },
        "multitask": {
            "n_total": len(multitask),
            "n_by_split": multitask_payload["n_by_split"],
            "counts": multitask_payload["counts_stage_by_split"],
        },
        "audit": {
            "stage_conflict_groups_excluded": stage_audit["n_conflicting_hash_groups_excluded"],
            "stage_files_excluded_for_conflict": stage_audit["n_conflicting_files_excluded"],
            "stage_redundant_files_removed": stage_audit["n_redundant_files_removed"],
            "stage_perceptual_conflict_groups_excluded": stage_perceptual_audit["n_conflicting_groups_excluded"],
            "stage_images_excluded_for_perceptual_conflict": stage_perceptual_audit["n_conflicting_images_excluded"],
            "stage_perceptual_redundant_images_removed": stage_perceptual_audit["n_redundant_images_removed"],
            "tissue_conflict_groups_excluded": tissue_audit["n_conflicting_hash_groups_excluded"],
            "tissue_redundant_files_removed": tissue_audit["n_redundant_files_removed"],
            "tissue_perceptual_conflict_groups_excluded": tissue_perceptual_audit["n_conflicting_groups_excluded"],
            "tissue_perceptual_redundant_images_removed": tissue_perceptual_audit["n_redundant_images_removed"],
            "group_leak_count": leakage["group_leak_count"],
            "exact_content_leak_count": leakage["exact_content_leak_count"],
            "perceptual_leak_count": leakage["perceptual_leak_count"],
        },
    }

    out = args.out_dir
    write_json(out / "stage_split.json", stage_payload)
    write_json(out / "tissue_split.json", tissue_payload)
    write_json(out / "multitask_split.json", multitask_payload)
    write_json(out / "group_assignment.json", split_for_group)
    write_json(out / "audit_report.json", {**audit, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
