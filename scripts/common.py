#!/usr/bin/env python3
"""Shared helpers: repository layout, manifest loading and image resolution.

No script in this repository stores an absolute path. Manifest records identify
an image by its SHA-256 digest; a local image directory is bound to those
records through the index produced by ``verify_manifests.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "manifests"
CONFIGS = REPO / "configs"
DEFAULT_INDEX = REPO / "manifests/local_index.json"
DEFAULT_WORK = REPO / "work"

SPLITS = ("train", "val", "test")

STAGE5 = ["estagio_1", "estagio_2", "estagio_3", "estagio_4", "nao_classificavel"]
STAGE12 = ["estagio_1", "estagio_2", "nao_classificavel"]
STAGE34 = ["estagio_3", "estagio_4", "nao_classificavel"]
TISSUE5 = ["HIPEREMIA", "GRANULAÇÃO", "ESFACELO", "NECROSE SECA", "NÃO CLASSIFICÁVEL"]
M1 = ["HIPEREMIA", "GRANULAÇÃO", "AMBAS", "NÃO CLASSIFICÁVEL"]
M2 = ["NECROSE SECA", "ESFACELO", "AMBAS", "NÃO CLASSIFICÁVEL"]
TISSUE_SL6 = ["HIPEREMIA", "GRANULAÇÃO", "ESFACELO", "NECROSE SECA", "NÃO CLASSIFICÁVEL", "MULTI"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_manifest(name: str) -> dict[str, Any]:
    """Load one manifest by short name, for example ``stage`` or ``detector``."""
    if not name.endswith(".json"):
        name = f"{name}_split.json"
    return read_json(MANIFESTS / name)


def manifest_items(name: str, split: str | None = None) -> list[dict[str, Any]]:
    items = load_manifest(name)["items"]
    return [item for item in items if split is None or item["split"] == split]


def items_by_split(name: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for item in load_manifest(name)["items"]:
        result[item["split"]].append(item)
    return result


class ImageIndex:
    """Maps a manifest digest to a file on this machine."""

    def __init__(self, mapping: dict[str, str], source: Path | None = None) -> None:
        self.mapping = mapping
        self.source = source

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX) -> "ImageIndex":
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"image index not found at {path}\n"
                "Run: python scripts/verify_manifests.py --data-dir <your image directory>"
            )
        payload = read_json(path)
        return cls(payload["sha256_to_path"], path)

    def path(self, item: dict[str, Any], key: str = "content_sha256") -> Path:
        digest = item[key]
        local = self.mapping.get(digest)
        if local is None:
            raise KeyError(
                f"no local file for {item.get('id', item.get('identity'))} ({digest[:12]}…); "
                "re-run scripts/verify_manifests.py over a complete image directory"
            )
        return Path(local)

    def resolve(self, items: list[dict[str, Any]], key: str = "content_sha256") -> list[dict[str, Any]]:
        """Return copies of ``items`` carrying a usable ``path`` field."""
        resolved = []
        for item in items:
            copy = dict(item)
            copy["path"] = str(self.path(item, key))
            resolved.append(copy)
        return resolved


def add_index_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="image index written by verify_manifests.py (default: manifests/local_index.json)",
    )


def add_work_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK,
        help="directory for datasets, runs and results (default: work/)",
    )


def m1_label(values: list[int]) -> str:
    h, g = bool(values[0]), bool(values[1])
    if h and g:
        return "AMBAS"
    if h:
        return "HIPEREMIA"
    if g:
        return "GRANULAÇÃO"
    return "NÃO CLASSIFICÁVEL"


def m2_label(values: list[int]) -> str:
    e, n = bool(values[2]), bool(values[3])
    if e and n:
        return "AMBAS"
    if n:
        return "NECROSE SECA"
    if e:
        return "ESFACELO"
    return "NÃO CLASSIFICÁVEL"


def sl6_label(values: list[int]) -> str:
    positives = [name for name, value in zip(TISSUE5[:4], values[:4]) if value]
    if len(positives) >= 2:
        return "MULTI"
    return positives[0] if positives else "NÃO CLASSIFICÁVEL"


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]
