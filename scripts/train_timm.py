#!/usr/bin/env python3
"""Train the timm models of the paper: stage heads, monoliths, tissue, multitask.

Ten experiments per seed, three seeds (42, 43, 44) = 30 timm runs. Every run
reads images straight from the manifests through the local index, so the folder
datasets built for YOLO are not needed here.

Usage:
    python scripts/train_timm.py --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, hamming_loss
from torch.utils.data import DataLoader, Dataset

import timm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    M1,
    M2,
    STAGE5,
    STAGE12,
    STAGE34,
    TISSUE5,
    ImageIndex,
    add_index_argument,
    add_work_argument,
    items_by_split,
    parse_seeds,
    write_json,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_transform(image_size: int = 224, augment: bool = False) -> T.Compose:
    if augment:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.RandomAffine(0, translate=(0.1, 0.1)),
                T.ColorJitter(0.3, 0.3, 0.3, 0.1),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_backbone(model_name: str, outputs: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=outputs)


class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        features = self.backbone.num_features
        self.head_stage = nn.Linear(features, 5)
        self.head_tissue = nn.Linear(features, 5)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(value)
        return self.head_stage(features), self.head_tissue(features)


class ManifestDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        mode: str,
        augment: bool,
        image_size: int = 224,
        oversample_m2: bool = False,
        seed: int = 42,
    ) -> None:
        self.mode = mode
        self.items = list(items)
        self.transform = build_transform(image_size, augment)
        if mode == "m2" and oversample_m2 and augment:
            minority = [item for item in self.items if self.m2_index(item["tissue"]) in (0, 2)]
            if minority:
                majority = len(self.items) - len(minority)
                rng = random.Random(seed)
                self.items.extend(rng.choice(minority) for _ in range(max(0, majority - len(minority))))

    @staticmethod
    def m1_index(values: list[int]) -> int:
        h, g = bool(values[0]), bool(values[1])
        if h and g:
            return 2
        if h:
            return 0
        if g:
            return 1
        return 3

    @staticmethod
    def m2_index(values: list[int]) -> int:
        e, n = bool(values[2]), bool(values[3])
        if e and n:
            return 2
        if n:
            return 0
        if e:
            return 1
        return 3

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = self.transform(Image.open(item["path"]).convert("RGB"))
        if self.mode == "stage5":
            return image, STAGE5.index(item["label_5"])
        if self.mode == "stage12":
            return image, STAGE12.index(item["label_12"])
        if self.mode == "stage34":
            return image, STAGE34.index(item["label_34"])
        if self.mode == "m1":
            return image, self.m1_index(item["tissue"])
        if self.mode == "m2":
            return image, self.m2_index(item["tissue"])
        if self.mode == "tissue_ml":
            return image, torch.tensor(item["tissue"], dtype=torch.float32)
        if self.mode == "multitask":
            return (
                image,
                STAGE5.index(item["label_5"]),
                torch.tensor(item["tissue"], dtype=torch.float32),
            )
        raise ValueError(f"unsupported mode: {self.mode}")


def load_items(pool: str, index: ImageIndex) -> dict[str, list[dict[str, Any]]]:
    """Manifest records for one pool, grouped by split and bound to local files."""
    return {split: index.resolve(items) for split, items in items_by_split(pool).items()}


def class_weights(labels: list[int], classes: int) -> torch.Tensor:
    counts = Counter(labels)
    values = torch.tensor(
        [max(1, counts.get(index, 0)) for index in range(classes)], dtype=torch.float32
    )
    weights = 1.0 / values
    return weights / weights.sum() * classes


def pos_weights(items: list[dict[str, Any]]) -> torch.Tensor:
    values = np.asarray([item["tissue"] for item in items], dtype=np.float32)
    positives = values.sum(axis=0)
    return torch.tensor((len(values) - positives) / np.maximum(positives, 1.0), dtype=torch.float32)


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def predict_cls(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> dict[str, Any]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        losses += criterion(logits, labels).item() * len(images)
        truths.append(labels.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    true = np.concatenate(truths)
    pred = np.concatenate(predictions)
    return {
        "loss": losses / max(1, len(loader.dataset)),
        "accuracy": float(accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, labels=list(range(classes)), average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "f1_per_class": f1_score(true, pred, labels=list(range(classes)), average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(classes))).tolist(),
        "y_true": true,
        "y_pred": pred,
        "probabilities": np.concatenate(probabilities),
    }


@torch.inference_mode()
def predict_ml(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        losses += criterion(logits, labels).item() * len(images)
        truths.append(labels.cpu().numpy().astype(np.int32))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    true = np.concatenate(truths)
    probability = np.concatenate(probabilities)
    pred = (probability >= 0.5).astype(np.int32)
    return {
        "loss": losses / max(1, len(loader.dataset)),
        "exact_match": float(accuracy_score(true, pred)),
        "hamming": float(hamming_loss(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(true, pred, average="micro", zero_division=0)),
        "f1_per_class": f1_score(true, pred, average=None, zero_division=0).tolist(),
        "y_true": true,
        "y_pred": pred,
        "probabilities": probability,
    }


@torch.inference_mode()
def predict_multitask(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    stage_true: list[np.ndarray] = []
    stage_prob: list[np.ndarray] = []
    tissue_true: list[np.ndarray] = []
    tissue_prob: list[np.ndarray] = []
    losses = 0.0
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    for images, stages, tissues in loader:
        images = images.to(device, non_blocking=True)
        stages = stages.to(device, non_blocking=True)
        tissues = tissues.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            stage_logits, tissue_logits = model(images)
        losses += (ce(stage_logits, stages) + bce(tissue_logits, tissues)).item() * len(images)
        stage_true.append(stages.cpu().numpy())
        stage_prob.append(torch.softmax(stage_logits, dim=1).cpu().numpy())
        tissue_true.append(tissues.cpu().numpy().astype(np.int32))
        tissue_prob.append(torch.sigmoid(tissue_logits).cpu().numpy())
    ys = np.concatenate(stage_true)
    ps = np.concatenate(stage_prob)
    yts = np.concatenate(tissue_true)
    pts = np.concatenate(tissue_prob)
    stage_pred = ps.argmax(axis=1)
    tissue_pred = (pts >= 0.5).astype(np.int32)
    stage_f1 = float(f1_score(ys, stage_pred, labels=list(range(5)), average="macro", zero_division=0))
    tissue_f1 = float(f1_score(yts, tissue_pred, average="macro", zero_division=0))
    return {
        "loss": losses / max(1, len(loader.dataset)),
        "stage_accuracy": float(accuracy_score(ys, stage_pred)),
        "stage_f1_macro": stage_f1,
        "stage_f1_per_class": f1_score(ys, stage_pred, labels=list(range(5)), average=None, zero_division=0).tolist(),
        "tissue_exact_match": float(accuracy_score(yts, tissue_pred)),
        "tissue_hamming": float(hamming_loss(yts, tissue_pred)),
        "tissue_f1_macro": tissue_f1,
        "tissue_f1_per_class": f1_score(yts, tissue_pred, average=None, zero_division=0).tolist(),
        "avg_f1": (stage_f1 + tissue_f1) / 2.0,
        "joint_exact": float(np.mean((stage_pred == ys) & np.all(tissue_pred == yts, axis=1))),
        "stage_y_true": ys,
        "stage_y_pred": stage_pred,
        "stage_probabilities": ps,
        "tissue_y_true": yts,
        "tissue_y_pred": tissue_pred,
        "tissue_probabilities": pts,
    }


def clean_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}


def save_predictions(path: Path, ids: list[str], result: dict[str, Any]) -> None:
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["ids"] = np.asarray(ids)
    np.savez_compressed(path, **arrays)


def filtered_items(items: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, list[dict[str, Any]]]:
    if mode != "stage34":
        return {key: list(value) for key, value in items.items()}
    return {
        key: [item for item in value if item.get("label_34") is not None]
        for key, value in items.items()
    }


def train_classifier(
    runs: Path,
    exp_id: str,
    mode: str,
    model_name: str,
    names: list[str],
    source: dict[str, list[dict[str, Any]]],
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    oversample_m2: bool = False,
) -> dict[str, Any]:
    output = runs / f"seed_{seed}" / exp_id
    if (output / "metrics.json").exists() and not args.force:
        print(f"[{seed}:{exp_id}] already complete")
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    items = filtered_items(source, mode)
    if args.smoke:
        # Only the training split shrinks: val and test stay complete so the
        # identifiers still line up with the YOLO runs in evaluate_bootstrap.py.
        items = {**items, "train": items["train"][: min(len(items["train"]), 64)]}
    train_dataset = ManifestDataset(items["train"], mode, True, oversample_m2=oversample_m2, seed=seed)
    val_dataset = ManifestDataset(items["val"], mode, False, seed=seed)
    test_dataset = ManifestDataset(items["test"], mode, False, seed=seed)
    train_loader = make_loader(train_dataset, args.batch, args.workers, True, seed)
    val_loader = make_loader(val_dataset, args.batch, args.workers, False, seed)
    test_loader = make_loader(test_dataset, args.batch, args.workers, False, seed)

    is_multilabel = mode == "tissue_ml"
    model = build_backbone(model_name, len(names), pretrained=True).to(device)
    if is_multilabel:
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weights(items["train"]).to(device))
    else:
        raw_dataset = ManifestDataset(items["train"], mode, False, seed=seed)
        labels = [int(raw_dataset[index][1]) for index in range(len(raw_dataset))]
        criterion = nn.CrossEntropyLoss(weight=class_weights(labels, len(names)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, Any]] = []
    started = time.time()
    max_epochs = min(args.epochs, 3) if args.smoke else args.epochs
    patience = min(args.patience, 2) if args.smoke else args.patience
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_multilabel:
                labels = labels.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(images)
        scheduler.step()
        validation = (
            predict_ml(model, val_loader, device)
            if is_multilabel
            else predict_cls(model, val_loader, device, len(names))
        )
        score = float(validation["f1_macro"])
        history.append({"epoch": epoch, "train_loss": running / len(train_dataset), **clean_metrics(validation)})
        print(f"[{seed}:{exp_id}] epoch={epoch:03d} val_f1={score:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "round": "seed_2",
                    "seed": seed,
                    "epoch": epoch,
                    "exp_id": exp_id,
                    "mode": mode,
                    "model_name": model_name,
                    "class_names": names,
                    "model_state_dict": model.state_dict(),
                    "validation": clean_metrics(validation),
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = (
        predict_ml(model, test_loader, device)
        if is_multilabel
        else predict_cls(model, test_loader, device, len(names))
    )
    ids = [str(item.get("id") or item.get("name") or item["identity"]) for item in items["test"]]
    save_predictions(output / "test_predictions.npz", ids, test)
    result = {
        "round": "seed_2",
        "seed": seed,
        "exp_id": exp_id,
        "mode": mode,
        "model_name": model_name,
        "class_names": names,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_val_f1_macro": best_score,
        "n_train_unique": len(items["train"]),
        "n_train_effective": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "seconds": time.time() - started,
        "protocol": {
            "image_size": 224,
            "batch": args.batch,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "optimizer": "Adam",
            "lr": args.lr,
            "scheduler": "CosineAnnealingLR",
            "pretrained": "ImageNet",
            "mixed_precision": amp_enabled,
        },
        "test": clean_metrics(test),
        "history": history,
    }
    write_json(output / "metrics.json", result)
    print(f"[{seed}:{exp_id}] DONE best_epoch={best_epoch} test={test['f1_macro']:.4f}", flush=True)
    return result


def train_multitask(
    runs: Path,
    exp_id: str,
    model_name: str,
    source: dict[str, list[dict[str, Any]]],
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = runs / f"seed_{seed}" / exp_id
    if (output / "metrics.json").exists() and not args.force:
        print(f"[{seed}:{exp_id}] already complete")
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    items = {key: list(value) for key, value in source.items()}
    if args.smoke:
        # Only the training split shrinks: val and test stay complete so the
        # identifiers still line up with the YOLO runs in evaluate_bootstrap.py.
        items = {**items, "train": items["train"][: min(len(items["train"]), 64)]}
    datasets = {
        key: ManifestDataset(value, "multitask", key == "train", seed=seed) for key, value in items.items()
    }
    loaders = {
        key: make_loader(dataset, args.batch, args.workers, key == "train", seed)
        for key, dataset in datasets.items()
    }
    model = MultiTaskModel(model_name, pretrained=True).to(device)
    stage_labels = [STAGE5.index(item["label_5"]) for item in items["train"]]
    ce = nn.CrossEntropyLoss(weight=class_weights(stage_labels, 5).to(device))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights(items["train"]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, Any]] = []
    started = time.time()
    max_epochs = min(args.epochs, 3) if args.smoke else args.epochs
    patience = min(args.patience, 2) if args.smoke else args.patience
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for images, stages, tissues in loaders["train"]:
            images = images.to(device, non_blocking=True)
            stages = stages.to(device, non_blocking=True)
            tissues = tissues.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                stage_logits, tissue_logits = model(images)
                loss = ce(stage_logits, stages) + bce(tissue_logits, tissues)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(images)
        scheduler.step()
        validation = predict_multitask(model, loaders["val"], device)
        score = float(validation["avg_f1"])
        history.append({"epoch": epoch, "train_loss": running / len(datasets["train"]), **clean_metrics(validation)})
        print(
            f"[{seed}:{exp_id}] epoch={epoch:03d} val_avg={score:.4f} "
            f"stage={validation['stage_f1_macro']:.4f} tissue={validation['tissue_f1_macro']:.4f}",
            flush=True,
        )
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "round": "seed_2",
                    "seed": seed,
                    "epoch": epoch,
                    "exp_id": exp_id,
                    "mode": "multitask",
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "validation": clean_metrics(validation),
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = predict_multitask(model, loaders["test"], device)
    ids = [str(item.get("id") or item["identity"]) for item in items["test"]]
    save_predictions(output / "test_predictions.npz", ids, test)
    result = {
        "round": "seed_2",
        "seed": seed,
        "exp_id": exp_id,
        "mode": "multitask",
        "model_name": model_name,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_val_f1_macro": best_score,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "seconds": time.time() - started,
        "protocol": {
            "image_size": 224,
            "batch": args.batch,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "optimizer": "Adam",
            "lr": args.lr,
            "scheduler": "CosineAnnealingLR",
            "pretrained": "ImageNet",
            "loss": "weighted CE + pos-weighted BCE, equal task weights",
            "mixed_precision": amp_enabled,
        },
        "test": clean_metrics(test),
        "history": history,
    }
    write_json(output / "metrics.json", result)
    print(f"[{seed}:{exp_id}] DONE best_epoch={best_epoch} test_avg={test['avg_f1']:.4f}", flush=True)
    return result


PLAN = [
    ("stage_head12_effnet", "cls", "stage12", "tf_efficientnetv2_s", STAGE12, "stage", False),
    ("stage_head34_effnet", "cls", "stage34", "tf_efficientnetv2_s", STAGE34, "stage", False),
    ("stage_mono_effnet", "cls", "stage5", "tf_efficientnetv2_s", STAGE5, "stage", False),
    ("stage_mono_densenet121", "cls", "stage5", "densenet121", STAGE5, "stage", False),
    ("stage_mono_resnet18", "cls", "stage5", "resnet18", STAGE5, "stage", False),
    ("tissue_m1_effnet", "cls", "m1", "tf_efficientnetv2_s", M1, "tissue", False),
    ("tissue_m2_effnet_os", "cls", "m2", "tf_efficientnetv2_s", M2, "tissue", True),
    ("tissue_mono_effnet_ml5", "cls", "tissue_ml", "tf_efficientnetv2_s", TISSUE5, "tissue", False),
    ("multitask_effnet", "mt", "multitask", "tf_efficientnetv2_s", None, "multitask", False),
    ("multitask_densenet121", "mt", "multitask", "densenet121", None, "multitask", False),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--only", default="", help="comma-separated exp_id subset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3 epochs on 64 training images; val/test stay complete, CPU allowed",
    )
    parser.add_argument("--force", action="store_true", help="retrain runs that already finished")
    add_index_argument(parser)
    add_work_argument(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("CUDA is required for the full round; use --smoke for a CPU dry run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    index = ImageIndex.load(args.index)
    runs = args.work_dir / "runs/timm"
    sources = {
        "stage": load_items("stage", index),
        "tissue": load_items("tissue", index),
        "multitask": load_items("multitask", index),
    }
    only = {value.strip() for value in args.only.split(",") if value.strip()}
    summary: list[dict[str, Any]] = []
    for seed in parse_seeds(args.seeds):
        for exp_id, kind, mode, model_name, names, pool, oversample in PLAN:
            if only and exp_id not in only:
                continue
            if kind == "mt":
                result = train_multitask(runs, exp_id, model_name, sources[pool], seed, device, args)
            else:
                assert names is not None
                result = train_classifier(
                    runs, exp_id, mode, model_name, names, sources[pool], seed, device, args,
                    oversample_m2=oversample,
                )
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
