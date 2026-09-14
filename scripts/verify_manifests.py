#!/usr/bin/env python3
"""Verify a local image collection against the published manifests.

The script does three independent things:

1. **Integrity.** Every image under ``--data-dir`` is hashed with SHA-256 and
   matched against the manifest records. It reports which records are covered,
   which are missing, and which local files belong to no record.
2. **Leakage.** It re-checks the frozen partition: no case-proxy group and no
   SHA-256 digest may appear in more than one split, and no pair of retained
   images may cross a split with perceptual distance <= 8. The perceptual test
   runs only when the images are available locally and ``--phash`` is given.
3. **Binding.** It writes ``manifests/local_index.json``, mapping each digest to
   a file on this machine. Every other script reads images through that index,
   so no absolute path is ever stored in the repository.

Usage:
    python scripts/verify_manifests.py --data-dir /path/to/images
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_INDEX,
    MANIFESTS,
    ImageIndex,
    iter_images,
    load_manifest,
    sha256_file,
    write_json,
)

POOLS = ("stage", "tissue", "multitask", "detector")


def digests_required(pool: str) -> dict[str, list[dict[str, Any]]]:
    """Every digest a pool needs, mapped to the records that need it."""
    required: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in load_manifest(pool)["items"]:
        required[item["content_sha256"]].append({"pool": pool, "id": item.get("id"), "role": "image"})
        if pool == "multitask":
            required[item["stage_content_sha256"]].append(
                {"pool": pool, "id": item.get("id"), "role": "stage_image"}
            )
    return required


def scan(data_dir: Path, workers: int) -> dict[str, str]:
    """Hash every image below ``data_dir``; first path wins on duplicates."""
    found: dict[str, str] = {}
    paths = list(iter_images(data_dir))
    if not paths:
        raise SystemExit(f"no image files found under {data_dir}")
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(pool.map(sha256_file, paths))
    else:
        digests = [sha256_file(path) for path in paths]
    for path, digest in zip(paths, digests):
        found.setdefault(digest, str(path))
    print(f"hashed {len(paths)} files, {len(found)} distinct digests", flush=True)
    return found


def check_partition(pool: str) -> dict[str, Any]:
    """Group and exact-content leakage, recomputed from the manifest alone."""
    payload = load_manifest(pool)
    group_key = "case_proxy_group" if pool == "detector" else "group_id"
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for item in payload["items"]:
        groups[item[group_key]].add(item["split"])
        hashes[item["content_sha256"]].add(item["split"])
    group_leaks = sorted(key for key, value in groups.items() if len(value) > 1)
    hash_leaks = sorted(key for key, value in hashes.items() if len(value) > 1)
    return {
        "pool": pool,
        "n_items": len(payload["items"]),
        "n_by_split": payload["n_by_split"],
        "n_groups": len(groups),
        "group_leaks": group_leaks,
        "exact_content_leaks": hash_leaks,
    }


def check_cross_pool_partition() -> dict[str, Any]:
    """Stage, tissue and multitask share one frozen group assignment."""
    assignment = load_manifest("group_assignment.json")
    groups: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    mismatched = []
    for pool in ("stage", "tissue", "multitask"):
        for item in load_manifest(pool)["items"]:
            groups[item["group_id"]].add(item["split"])
            hashes[item["content_sha256"]].add(item["split"])
            expected = assignment.get(item["group_id"])
            if expected is not None and expected != item["split"]:
                mismatched.append({"pool": pool, "id": item.get("id"), "group_id": item["group_id"]})
    return {
        "group_leaks": sorted(key for key, value in groups.items() if len(value) > 1),
        "exact_content_leaks": sorted(key for key, value in hashes.items() if len(value) > 1),
        "records_disagreeing_with_group_assignment": mismatched,
    }


def check_perceptual(index: ImageIndex, threshold: int) -> dict[str, Any]:
    """Recompute pHash from the local files and look for near-duplicate leaks."""
    from build_data import perceptual_hash  # local import: needs Pillow and SciPy

    leaks = []
    recomputed_mismatch = []
    for pool in ("stage", "tissue"):
        items = []
        for item in load_manifest(pool)["items"]:
            try:
                path = index.path(item)
            except KeyError:
                continue
            value = perceptual_hash(str(path))
            if f"{value:016x}" != item["perceptual_hash"]:
                recomputed_mismatch.append({"pool": pool, "id": item.get("id")})
            items.append((item, value))
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                distance = (items[left][1] ^ items[right][1]).bit_count()
                if distance <= threshold and items[left][0]["split"] != items[right][0]["split"]:
                    leaks.append(
                        {
                            "pool": pool,
                            "left": items[left][0].get("id"),
                            "right": items[right][0].get("id"),
                            "hamming": distance,
                            "splits": [items[left][0]["split"], items[right][0]["split"]],
                        }
                    )
    return {
        "threshold": threshold,
        "perceptual_leaks": leaks,
        "records_whose_phash_changed": recomputed_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        help="directory holding your copy of the images; repeat for several roots",
    )
    parser.add_argument("--index-out", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report-out", type=Path, default=MANIFESTS / "verification_report.json")
    parser.add_argument("--workers", type=int, default=8, help="threads used for hashing")
    parser.add_argument(
        "--phash",
        action="store_true",
        help="also recompute perceptual hashes and re-test the <=8 bit criterion (slow)",
    )
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write the index and exit 0 even when some records have no local file",
    )
    args = parser.parse_args()

    partitions = {pool: check_partition(pool) for pool in POOLS}
    cross = check_cross_pool_partition()
    report: dict[str, Any] = {"partitions": partitions, "cross_pool": cross}

    leak_failures = sum(
        len(value["group_leaks"]) + len(value["exact_content_leaks"])
        for pool, value in partitions.items()
        if pool != "detector"
    )
    leak_failures += len(cross["group_leaks"]) + len(cross["exact_content_leaks"])
    leak_failures += len(cross["records_disagreeing_with_group_assignment"])

    print("== partition ==")
    for pool, value in partitions.items():
        marker = "detector: task-specific split, reported only" if pool == "detector" else ""
        print(
            f"  {pool:<10} n={value['n_items']:<5} groups={value['n_groups']:<5} "
            f"group_leaks={len(value['group_leaks']):<4} hash_leaks={len(value['exact_content_leaks'])} {marker}"
        )
    print(
        f"  shared assignment: group_leaks={len(cross['group_leaks'])} "
        f"hash_leaks={len(cross['exact_content_leaks'])} "
        f"disagreements={len(cross['records_disagreeing_with_group_assignment'])}"
    )

    missing_total = 0
    if args.data_dir:
        found = {}
        for directory in args.data_dir:
            found.update({key: value for key, value in scan(directory, args.workers).items() if key not in found})
        coverage = {}
        for pool in POOLS:
            required = digests_required(pool)
            missing = sorted(digest for digest in required if digest not in found)
            coverage[pool] = {
                "n_required_digests": len(required),
                "n_found": len(required) - len(missing),
                "n_missing": len(missing),
                "missing_records": [row for digest in missing[:200] for row in required[digest]],
            }
            missing_total += len(missing)
            print(
                f"  {pool:<10} matched {coverage[pool]['n_found']}/{len(required)} digests"
                + (f"  MISSING {len(missing)}" if missing else "")
            )
        every_required = set()
        for pool in POOLS:
            every_required.update(digests_required(pool))
        unreferenced = sorted(digest for digest in found if digest not in every_required)
        report["coverage"] = coverage
        report["n_local_files_not_in_any_manifest"] = len(unreferenced)
        index = ImageIndex({digest: path for digest, path in found.items() if digest in every_required})
        write_json(args.index_out, {"n": len(index.mapping), "sha256_to_path": index.mapping})
        print(f"  wrote index -> {args.index_out} ({len(index.mapping)} files)")
        print(f"  local files matching no record: {len(unreferenced)}")
        if args.phash:
            perceptual = check_perceptual(index, args.phash_threshold)
            report["perceptual"] = perceptual
            print(
                f"  perceptual leaks (<= {args.phash_threshold} bits): "
                f"{len(perceptual['perceptual_leaks'])}, "
                f"changed hashes: {len(perceptual['records_whose_phash_changed'])}"
            )
            leak_failures += len(perceptual["perceptual_leaks"])
    else:
        print("  (no --data-dir given: manifest-only checks)")

    write_json(args.report_out, report)
    print(f"  wrote report -> {args.report_out}")

    if leak_failures:
        print(f"FAIL: {leak_failures} leakage findings in the audited pools", file=sys.stderr)
        return 1
    if missing_total and not args.allow_incomplete:
        print(f"FAIL: {missing_total} manifest records have no local file", file=sys.stderr)
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
