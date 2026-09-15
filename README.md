# Pressure Injury Staging and Tissue Classification — Split Manifests and Reproduction Code

Frozen split manifests and the complete reproduction pipeline for the study on
offline mobile support for pressure injury staging and tissue classification.

This repository contains **metadata, not images**. Every record carries the image
identifier, its SHA-256 digest, its 64-bit perceptual hash, its case-proxy group,
and its train/validation/test assignment. Clinical photographs are not
redistributed here: source licensing and publication permission must be verified
per image, independently of their use in model development. If you hold the
images, `scripts/verify_manifests.py` binds your copy to these records by content
hash and lets you re-run the entire pipeline.

---

## What is in here

| Path | Contents |
|---|---|
| `manifests/` | The frozen partition: five pools as JSON and CSV, the shared group assignment, and the full content audit |
| `scripts/` | Verification, manifest rebuild, dataset materialization, training, inference, calibration, bootstrap evaluation |
| `configs/` | Detector `data.yaml` and the frozen hyperparameters of every architecture |

Nothing in `scripts/` stores an absolute path. Every script resolves paths from
`Path(__file__).resolve()` and takes `argparse` options with relative defaults.

---

## The pipeline

```
photograph
    │
    ├─► YOLO26s detect ───────────────► wound bounding box (region of interest)
    │
    ├─► Stage branch
    │      Head 12 {E1, E2, R} ──► routing class R opens
    │      Head 34 {E3, E4, NC} ──► cascade output over five classes
    │      (compared against a five-class monolith)
    │
    ├─► Tissue branch
    │      M1 {hyperemia, granulation, both, NC}
    │      M2 {dry necrosis, slough, both, NC}
    │      fusion ──► ML5 vector [H, G, E, N, NC]
    │      (compared against an ML5 multi-label monolith and a SL6 single-label YOLO head)
    │
    └─► Multi-task model: one backbone, one stage head and one tissue head,
        evaluated on the 45 identifiers where both annotations exist
```

Stage is a five-class problem with exactly one label per image. Tissue is five
**independent binary attributes**, so its counts are positives per attribute, not
a partition of the pool. The two branches are never scored in the same space
unless they are evaluated on the same identifiers, which is what the multi-task
pool exists for.

---

## The data

All counts below come from the manifests in this repository and match the paper.

| Pool | Manifest | Total | Train (80%) | Val. (15%) | Test (5%) |
|---|---|---:|---:|---:|---:|
| Stage, five classes | `stage_split.json` | 1,581 | 1,271 | 229 | 81 |
| Head 34 subset (E3/E4/NC) | `stage_split.json`, `in_head_34` | 1,017 | 829 | 131 | 57 |
| Tissue, ML5 | `tissue_split.json` | 1,224 | 1,005 | 159 | 60 |
| Stage ∩ tissue (multi-task) | `multitask_split.json` | 905 | 730 | 130 | 45 |
| Detector, bounding boxes | `detector_split.json` | 1,514 | 1,059 | 227 | 228 |

Stage test support: E1 = 11, E2 = 13, E3 = 40, E4 = 12, NC = 5.
Tissue test positives: H = 14, G = 16, E = 9, N = 3, NC = 2.
Joint test stage support: E1 = 11, E2 = 10, E3 = 21, E4 = 3, NC = 0 — the joint
test contains no true NC case, and by fixed policy F1(NC) = 0 still enters the
five-class macro F1 rather than silently disappearing from the mean.

### Record fields

`stage_split.json` / `.csv`

| Field | Meaning |
|---|---|
| `id`, `identity` | Original catalogue identifier and its normalized form |
| `file_name` | Original file name; `source_relative_path` keeps the legacy location for traceability |
| `content_sha256` | **The join key.** Images are matched to records by content, never by path |
| `perceptual_hash` | 64-bit DCT pHash, hex |
| `group_id` | Case-level proxy, shared across all three classification pools |
| `split` | `train` / `val` / `test` |
| `label_5`, `label_12`, `label_34`, `in_head_34` | Five-class label and the two projected head labels |

`tissue_split.json` / `.csv` carries the binary vector `tissue` = [H, G, E, N, NC]
(also as one column per attribute) plus the derived `m1`, `m2` and `sl6` labels.
`multitask_split.json` carries both the tissue and the stage digest for each
identifier, plus `match_kind` (`identity` or `sha256`) recording how the two
annotations were linked.

`detector_split.json` additionally embeds the YOLO boxes
(`boxes_xywhn` = normalized `[class, x_center, y_center, width, height]`), so the
detection dataset can be rebuilt from the manifest alone once you have the images.

---

## Content audit and anti-leakage protocol

The audit ran **before** any partitioning, and every number below is reproduced by
`scripts/verify_manifests.py` from the manifests.

1. **SHA-256.** Every file was hashed. When images with identical content carried
   conflicting stage labels, the whole content group was discarded; when they
   agreed, one representative was kept.
2. **64-bit perceptual hash (DCT).** The same conflict/redundancy rules were then
   applied to recompressed or resized copies. Perceptual equivalence: Hamming
   distance ≤ 4, or ≤ 10 together with mean absolute grayscale error ≤ 10 on
   64×64 thumbnails. Borderline pairs were inspected visually.
3. **Conservative case grouping.** No clinical patient identifier was available.
   Files sharing a numeric prefix (`<digits>_`), a copy suffix (`(2)`, `__dup2`,
   `copy`), or retained visual similarity at pHash ≤ 8 were forced into one
   `group_id` (`clinical:<id>` or `name:<stem>`).
4. **One frozen assignment.** A single grouped 80/15/5 partition was searched with
   seed `20260812`, optimizing class balance under a rare-label-support penalty,
   and shared by stage, tissue and multi-task records.

| Audit | Result |
|---|---:|
| Stage exact conflict groups excluded | 5 groups / 10 files |
| Stage exact redundancies removed | 4 |
| Stage perceptual conflict groups excluded | 7 groups / 15 files |
| Stage perceptual redundancies removed | 19 |
| Tissue exact redundancies removed | 1 |
| Tissue perceptual redundancies removed | 2 |
| Tissue conflict groups (exact or perceptual) | 0 |
| **Group / SHA-256 / perceptual leaks after the split** | **0 / 0 / 0** |

> **The grouping rule is a case-level proxy, not verified patient metadata.** It
> controls *detectable* content leakage. It does not establish patient-level
> independence, and no claim in the paper depends on it doing so.

> **The detector is different, on purpose.** Its 1,059/227/228 split predates this
> protocol and was kept as-is; the detector was trained once, not across the three
> seeds, and its metrics never enter a classification claim. For transparency,
> `verify_manifests.py` reports what that split would look like under the same
> proxy: **121 case-proxy groups and 2 exact-content groups cross its splits**.
> Treat its numbers as localization performance under its own protocol only.

---

## Installation

Two environments are needed, because the `timm` and Ultralytics rounds ran on
different torch and numpy majors.

```bash
git clone <this repository>
cd pressure-wound-curascan

# Environment A -- manifests, timm models, calibration, evaluation (Python 3.11)
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Environment B -- Ultralytics YOLO26s (Python 3.12)
python3.12 -m venv .venv-yolo && . .venv-yolo/bin/activate
pip install -r requirements-yolo.txt
```

Install the `torch` wheel matching your CUDA runtime first, for example
`pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128`.

Reference environments: Python 3.11 / PyTorch 2.10 / timm 1.0.25, and
Python 3.12 / PyTorch 2.12 / Ultralytics 8.4.86. A CUDA GPU is required for the
full round; every training script accepts `--smoke`, which caps training at three
epochs on 64 images while keeping validation and test complete, so the whole
chain stays runnable end to end.

---

## Reproduction

### 1. Verify your images and bind them to the manifests

```bash
python scripts/verify_manifests.py \
    --data-dir /path/to/your/stage_images \
    --data-dir /path/to/your/tissue_images \
    --data-dir /path/to/your/detector_images
```

Pass `--data-dir` once per root; subdirectories are searched recursively. The
script hashes every image, reports coverage per pool, re-checks the partition for
group / content leakage, and writes `manifests/local_index.json` mapping each
digest to a file on your machine. **Every later script reads images through that
index**, which is why no absolute path ever enters the repository.

It exits non-zero if a leak is found (exit 1) or if records have no local file
(exit 2; use `--allow-incomplete` to proceed anyway). Add `--phash` to also
recompute the perceptual hashes and re-test the ≤ 8-bit criterion — slower, and
the strictest available check.

```
== partition ==
  stage      n=1581  groups=1224  group_leaks=0    hash_leaks=0
  tissue     n=1224  groups=661   group_leaks=0    hash_leaks=0
  multitask  n=905   groups=549   group_leaks=0    hash_leaks=0
  detector   n=1514  groups=950   group_leaks=121  hash_leaks=2   (own split, reported only)
  shared assignment: group_leaks=0 hash_leaks=0 disagreements=0
```

### 2. Materialize the YOLO folder datasets

```bash
python scripts/build_yolo_datasets.py --seeds 42,43,44
```

Writes `work/yolo_datasets/` — six classification datasets plus per-seed M2
oversampling and the rebuilt detector (`images/`, `labels/`, `data.yaml`). Files
are hard-linked, so this costs almost no disk space. The `timm` scripts do not
need this step; they read images directly through the index.

### 3. Train — three seeds, 48 classification runs

```bash
# Environment A: 10 experiments x 3 seeds = 30 runs
python scripts/train_timm.py --seeds 42,43,44

# Environment B: 6 experiments x 3 seeds = 18 runs
python scripts/train_yolo.py --seeds 42,43,44 --weights yolo26s-cls.pt
```

Both scripts are resumable: a run whose `metrics.json` exists is skipped unless
you pass `--force`. Use `--only <exp_id>,<exp_id>` to train a subset.

`seed_2` is the name of the experimental round, not a single random seed. The
split is frozen and identical for all three; only initialization and batch order
vary between 42, 43 and 44, which is exactly the variability the confidence
intervals absorb.

The detector is a separate, single run on its own partition:

```bash
python scripts/train_yolo.py --task detect --weights yolo26s.pt   # 300 epochs, 640px, seed 0
```

### 4. Auxiliary inference for the fusions

```bash
python scripts/infer.py --family timm --seeds 42,43,44    # Environment A
python scripts/infer.py --family yolo --seeds 42,43,44    # Environment B
```

The cascade and the tissue fusion combine two heads over the *same* identifiers,
so each head must also run over contexts that are not its own test folder. This
writes `<context>_aux.npz` for `stage_test`, `tissue_val`, `tissue_test` and
`joint_test`, each carrying ids, group ids, ground truth and full probabilities.

### 5. Calibrate on validation only

```bash
python scripts/calibrate.py --seeds 42,43,44
```

One threshold per tissue attribute, grid-searched over [0.05, 0.95] in 0.01 steps
to maximize **validation** F1, ties resolving closest to 0.5. The NC fusion policy
(AND / OR / M1 / M2) is chosen the same way, per family and seed, inside step 6;
AND is preserved on ties. The test set is never consulted by either choice.

### 6. Evaluate with the hierarchical bootstrap

```bash
python scripts/evaluate_bootstrap.py --seeds 42,43,44 --bootstrap 5000
```

Writes `work/results/results_master.json`: per-system point estimates, SD across
training seeds, 95% percentile CIs, per-class F1, and the eight paired deltas.
The bootstrap (B = 5,000, seed `20260812`) resamples **both** the training seeds
and the paired held-out identifiers, so seed noise and test-set noise enter the
same interval. Paired deltas use the same resampled seeds and identifiers.

A confidence interval that contains zero is **inconclusive evidence, not
equality** — the larger point estimate is not reported as an established win.

---

## Smoke test

Before committing a GPU to the full round, run the whole chain in a few minutes.
`--smoke` caps training at three epochs on 64 training images; validation and
test splits stay complete, so identifiers still line up and every downstream step
is exercised for real.

```bash
# Environment A
python scripts/verify_manifests.py --data-dir <your images> --phash
python scripts/build_yolo_datasets.py --seeds 42
python scripts/train_timm.py --smoke --seeds 42 --work-dir work-smoke

# Environment B
python scripts/train_yolo.py --smoke --seeds 42 --work-dir work-smoke \
    --weights yolo26s-cls.pt
python scripts/infer.py --family yolo --seeds 42 --work-dir work-smoke

# Environment A again
python scripts/infer.py --family timm --seeds 42 --work-dir work-smoke
python scripts/calibrate.py --seeds 42 --work-dir work-smoke
python scripts/evaluate_bootstrap.py --seeds 42 --bootstrap 500 --work-dir work-smoke
```

The metrics it produces are meaningless — three epochs on 64 images — but the run
proves your images, environments and both model families are wired correctly. It
should finish with `work-smoke/results/results_master.json` holding 24 systems and
8 paired deltas. The detector can be smoke-tested the same way with
`python scripts/train_yolo.py --task detect --smoke --work-dir work-smoke`.

---

## Expected results

Reference values from the published round (mean over seeds 42/43/44 ± SD,
[95% hierarchical CI]). Use these to check a reproduction.

**Five-class stage output** (n = 81)

| System | Accuracy | Macro F1 |
|---|---|---|
| EfficientNetV2-S cascade | 0.695 ± 0.029 [0.605; 0.782] | 0.697 ± 0.029 [0.575; 0.780] |
| YOLO26s-cls cascade | 0.741 ± 0.025 [0.654; 0.827] | 0.711 ± 0.012 [0.586; 0.800] |
| EfficientNetV2-S monolith | 0.671 ± 0.043 [0.572; 0.765] | 0.667 ± 0.033 [0.541; 0.762] |
| YOLO26s-cls monolith | 0.704 ± 0.033 [0.605; 0.794] | 0.689 ± 0.038 [0.558; 0.784] |
| ResNet18 monolith | 0.745 ± 0.014 [0.654; 0.827] | 0.749 ± 0.029 [0.631; 0.835] |
| DenseNet121 monolith | 0.728 ± 0.012 [0.638; 0.811] | 0.725 ± 0.012 [0.605; 0.810] |

**Tissue output** (n = 60)

| System | Output | EM / Acc | Macro F1 |
|---|---|---|---|
| EfficientNetV2-S M1+M2 fusion | ML5 | 0.206 ± 0.139 [0.050; 0.367] | 0.471 ± 0.009 [0.289; 0.567] |
| YOLO26s-cls M1+M2 fusion | ML5 | 0.306 ± 0.025 [0.194; 0.417] | 0.550 ± 0.030 [0.384; 0.650] |
| EfficientNetV2-S monolith | ML5 | 0.539 ± 0.082 [0.400; 0.672] | 0.576 ± 0.052 [0.393; 0.706] |
| YOLO26s-cls monolith | SL6 | 0.628 ± 0.025 [0.511; 0.733] | 0.591 ± 0.029 [0.410; 0.698] |

**Joint system on the same 45 identifiers**

| System | Stage F1 | Tissue F1 | Joint exact |
|---|---|---|---|
| Multi-task EfficientNetV2-S | 0.589 ± 0.075 [0.427; 0.701] | 0.536 ± 0.008 [0.398; 0.636] | 0.496 ± 0.013 [0.363; 0.622] |
| Multi-task DenseNet121 | 0.558 ± 0.028 [0.415; 0.657] | 0.533 ± 0.032 [0.424; 0.624] | 0.467 ± 0.097 [0.311; 0.622] |
| Modular YOLO26s-cls | 0.445 ± 0.024 [0.357; 0.514] | 0.569 ± 0.034 [0.393; 0.676] | 0.289 ± 0.022 [0.178; 0.415] |
| Modular EfficientNetV2-S | 0.467 ± 0.087 [0.342; 0.577] | 0.469 ± 0.012 [0.291; 0.586] | 0.178 ± 0.139 [0.022; 0.348] |

`joint exact` requires the stage label **and** the complete tissue vector to be
correct for the same image.

**Detector**, on its own held-out partition (n = 228): precision 0.871, recall
0.835, mAP@50 0.897, mAP@50–95 0.525. Not comparable with, and never combined
with, the classification numbers above.

Training is stochastic below the level these intervals resolve: expect your point
estimates to land inside the reported CIs rather than to match digit for digit.
Steps 4–6 are deterministic given fixed checkpoints — re-running them on the
published checkpoints reproduces `results_master.json` exactly.

---

## Rebuilding the manifests from scratch (optional)

```bash
python scripts/build_data.py \
    --stage-catalogue  /path/to/stage_catalogue.json \
    --tissue-catalogue /path/to/tissue_catalogue.json \
    --out-dir manifests/rebuilt --trials 100000
```

Catalogue format: `{"items": [{"id": ..., "path": ..., "label_5": ...}, ...]}` for
stage, and `{"items": [{"name": ..., "path": ..., "tissue": [H,G,E,N,NC]}, ...]}`
for tissue. Given the same catalogue the search is deterministic and rebuilds the
published partition exactly. Given a *different* catalogue it produces a
different, equally valid partition — and a different test set — so results
computed that way are not comparable with the published numbers. The output
defaults to `manifests/rebuilt/` so the frozen manifests are never overwritten by
accident.

---

## Known limitations

These are properties of the study, not of this repository, and are stated here so
a reproduction is not read as stronger than it is.

- **No confirmed patient/case identifier.** The filename proxy reduces leakage
  risk but does not prove clinical independence.
- **Small test sets and rare classes.** NC, dry necrosis and E4 are especially
  thin in the joint pool; several intervals are correspondingly wide.
- **No external cohort**, and no end-to-end evaluation from an unconstrained
  photograph through detection to staging.
- **The detector was not re-run** under this protocol, and its split carries the
  leakage counters reported above.
- **On-device cost is unmeasured**: latency, memory, energy and export-equivalence
  are not part of these results.

---

## Citation

```bibtex
@inproceedings{pressure_wound_staging,
  title     = {Pressure Injury Staging and Tissue Classification: A Content-Audited Evaluation with a Prospective Clinical Pilot},
  author    = {Andreza Falcao, Lenon Anthony, Barbara Angelo, Filipe Rolim, Maria Da Conceição Cavalcanti, Rafaela Andrade, Mateus Silva, Taciana Pontual and André Câmara},
  year      = {2026}
}
```

Please also cite Ultralytics for YOLO26 and `timm` for the ImageNet backbones.

## License

Code and manifests: MIT (see `LICENSE`). The manifests describe images that are
**not** distributed here; rights to the underlying photographs are held by their
respective sources and are not granted by this license.
