#!/usr/bin/env python3
"""Materialize Ultralytics folders from the manifests and your local images.

Classification datasets follow the ``<root>/<split>/<class>/<file>`` layout that
``ultralytics`` expects; the detector dataset follows ``images/<split>`` plus
``labels/<split>`` with the boxes stored in ``detector_split.json``. Files are
hard-linked where the filesystem allows it, then symlinked, then copied, so no
image is duplicated on disk unnecessarily.

Usage:
    python scripts/build_yolo_datasets.py --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CONFIGS,
    SPLITS,
    STAGE5,
    ImageIndex,
    add_index_argument,
    add_work_argument,
    load_manifest,
    manifest_items,
    parse_seeds,
    write_json,
)

STAGE12 = ["estagio_1", "estagio_2", "nao_classificavel"]
STAGE34 = ["estagio_3", "estagio_4", "nao_classificavel"]
M1_ASCII = ["HIPEREMIA", "GRANULACAO", "AMBAS", "NAO_CLASSIFICAVEL"]
M2_ASCII = ["NECROSE_SECA", "ESFACELO", "AMBAS", "NAO_CLASSIFICAVEL"]
SL6_ASCII = ["HIPEREMIA", "GRANULACAO", "ESFACELO", "NECROSE_SECA", "NAO_CLASSIFICAVEL", "MULTI"]


def ascii_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return normalized.replace(" ", "_").upper()


def m1_ascii(item: dict[str, Any]) -> str:
    h, g = bool(item["tissue"][0]), bool(item["tissue"][1])
    if h and g:
        return "AMBAS"
    if h:
        return "HIPEREMIA"
    if g:
        return "GRANULACAO"
    return "NAO_CLASSIFICAVEL"


def m2_ascii(item: dict[str, Any]) -> str:
    e, n = bool(item["tissue"][2]), bool(item["tissue"][3])
    if e and n:
        return "AMBAS"
    if n:
        return "NECROSE_SECA"
    if e:
        return "ESFACELO"
    return "NAO_CLASSIFICAVEL"


def sl6_ascii(item: dict[str, Any]) -> str:
    names = ["HIPEREMIA", "GRANULACAO", "ESFACELO", "NECROSE_SECA"]
    positives = [name for name, value in zip(names, item["tissue"][:4]) if value]
    if len(positives) >= 2:
        return "MULTI"
    if positives:
        return positives[0]
    return "NAO_CLASSIFICAVEL"


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        try:
            os.symlink(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def materialize(
    out: Path,
    name: str,
    items: list[dict[str, Any]],
    classes: list[str],
    label: Callable[[dict[str, Any]], str],
    index: ImageIndex,
) -> dict[str, Any]:
    root = out / name
    if root.exists():
        shutil.rmtree(root)
    mapping: list[dict[str, Any]] = []
    for split in SPLITS:
        for class_name in classes:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)
    ordered = sorted(items, key=lambda value: (value["split"], value["identity"]))
    for position, item in enumerate(ordered):
        class_name = label(item)
        source = index.path(item)
        suffix = source.suffix.lower() or ".jpg"
        identifier = str(item.get("id") or item.get("name") or item["identity"])
        safe = "".join(char if char.isalnum() else "_" for char in ascii_name(identifier))[:80]
        destination = root / item["split"] / class_name / f"{position:05d}__{safe}{suffix}"
        link(source, destination)
        mapping.append(
            {
                "relative_path": str(destination.relative_to(root)),
                "id": identifier,
                "content_sha256": item["content_sha256"],
                "group_id": item["group_id"],
                "class": class_name,
                "split": item["split"],
            }
        )
    payload = {
        "dataset": name,
        "classes": classes,
        "n": len(mapping),
        "counts": {
            split: dict(Counter(row["class"] for row in mapping if row["split"] == split))
            for split in SPLITS
        },
        "items": mapping,
    }
    write_json(root / "mapping.json", payload)
    return payload


def oversample_m2(out: Path, seed: int, base: Path) -> dict[str, Any]:
    destination = out / f"tissue_m2_os_seed_{seed}"
    if destination.exists():
        shutil.rmtree(destination)
    for split in SPLITS:
        for class_name in M2_ASCII:
            (destination / split / class_name).mkdir(parents=True, exist_ok=True)
            for source in sorted((base / split / class_name).iterdir()):
                if source.is_file():
                    link(source.resolve(), destination / split / class_name / source.name)
    rng = random.Random(seed)
    files = {
        class_name: [path for path in (destination / "train" / class_name).iterdir() if path.is_file()]
        for class_name in M2_ASCII
    }
    target = max(len(values) for values in files.values())
    for class_name, values in files.items():
        for position in range(target - len(values)):
            source = rng.choice(values)
            link(source.resolve(), destination / "train" / class_name / f"os_{position:04d}__{source.name}")
    payload = {
        "seed": seed,
        "strategy": "random hard-link replication of every minority M2 class to the training majority count",
        "original_counts": {key: len(value) for key, value in files.items()},
        "effective_counts": {key: target for key in files},
    }
    write_json(destination / "oversampling.json", payload)
    return payload


def format_coordinate(value: float) -> str:
    """Shortest round-trip form, keeping integral values integral (1, not 1.0).

    This reproduces the original annotation export byte for byte, so a rebuilt
    label file can be diffed against the one it came from.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def materialize_detector(out: Path, index: ImageIndex) -> dict[str, Any]:
    """Rebuild images/<split> and labels/<split> plus a runnable data.yaml."""
    payload = load_manifest("detector")
    root = out / "detector"
    if root.exists():
        shutil.rmtree(root)
    counts: Counter[str] = Counter()
    for item in payload["items"]:
        source = index.path(item)
        split = item["split"]
        image = root / "images" / split / item["file_name"]
        link(source, image)
        label = root / "labels" / split / f"{Path(item['file_name']).stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            " ".join([str(int(box[0]))] + [format_coordinate(value) for value in box[1:]])
            for box in item["boxes_xywhn"]
        ]
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
    data_yaml = root / "data.yaml"
    # `path` must be absolute: Ultralytics resolves a relative one against its
    # own datasets_dir setting, not against the location of this file.
    data_yaml.write_text(
        "task: detect\n"
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {position}: {name}\n" for position, name in enumerate(payload["classes"])),
        encoding="utf-8",
    )
    return {"root": str(root), "counts": dict(counts), "data_yaml": str(data_yaml)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="42,43,44")
    add_index_argument(parser)
    add_work_argument(parser)
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="do not rebuild the detection dataset",
    )
    args = parser.parse_args()

    index = ImageIndex.load(args.index)
    out = args.work_dir / "yolo_datasets"
    out.mkdir(parents=True, exist_ok=True)
    stage = manifest_items("stage")
    tissue = manifest_items("tissue")

    datasets = {
        "stage_head12": materialize(out, "stage_head12", stage, STAGE12, lambda item: item["label_12"], index),
        "stage_head34": materialize(
            out,
            "stage_head34",
            [item for item in stage if item.get("label_34") is not None],
            STAGE34,
            lambda item: item["label_34"],
            index,
        ),
        "stage_mono5": materialize(out, "stage_mono5", stage, STAGE5, lambda item: item["label_5"], index),
        "tissue_m1": materialize(out, "tissue_m1", tissue, M1_ASCII, m1_ascii, index),
        "tissue_m2": materialize(out, "tissue_m2", tissue, M2_ASCII, m2_ascii, index),
        "tissue_mono_sl6": materialize(out, "tissue_mono_sl6", tissue, SL6_ASCII, sl6_ascii, index),
    }
    oversampling = {
        str(seed): oversample_m2(out, seed, out / "tissue_m2") for seed in parse_seeds(args.seeds)
    }
    summary = {
        "round": "seed_2",
        "datasets": {name: {"n": value["n"], "counts": value["counts"]} for name, value in datasets.items()},
        "m2_oversampling": oversampling,
    }
    if not args.skip_detector:
        summary["detector"] = materialize_detector(out, index)
        print(
            f"detector rebuilt at {summary['detector']['root']}; "
            f"a ready-made config is also available at {CONFIGS / 'detector_data.yaml'}"
        )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
