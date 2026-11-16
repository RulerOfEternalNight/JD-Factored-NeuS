# Reproduction and exact artifact restoration

For an immutable source checkout, use tag `softmax-10landmark-working-code-v1`.
Model artifacts use release tag `softmax-10landmark-working-v1`. Both retain the
same 64-file training/configuration snapshot; additional source-only-clone test
handling and CSV normalization do not change model behavior.

## Two different meanings of “the same result”

1. **Exact archived state:** download the release and verify its SHA-256 manifest.
   Checkpoints, RNG/optimizer/cache state, meshes, coordinates and numeric results
   are restored byte for byte. This is the reliable way to revisit the exact run.
2. **Repeat training:** run the same frozen code, original input bytes, baseline
   checkpoint, random seed, schedule and environment. This reproduces the
   experimental recipe, but cross-GPU/software bitwise identity is not promised.
   The original code seeds RNGs; it did not establish deterministic CUDA execution
   across machines. Compare coverage and errors, not an assumed identical hash.

The original run was recorded at `runs/20261116T004348Z_staged10_corrected`.
The original resolved launch configuration contains historical absolute paths;
it is evidence, not the portable launch file. Use the unchanged relative HOCON
configuration or the wrapper documented below.

## Environment

Original runtime: Python 3.10.21, PyTorch 1.13.1+cu117, CUDA 11.7 runtime,
NVIDIA RTX 4070 Laptop 8 GiB, Linux/WSL environment. The host driver must support
the documented CUDA wheel. On the original machine the Conda-compatible manager
is Micromamba at `/workspace/.local/bin/micromamba`, with environment
`/workspace/.micromamba/envs/factored-neus`. A Conda installation is not required.

For a new Linux environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip check
```

Optional documentation generation uses `requirements-docs.txt`. The full observed
package inventory is `reproducibility/environment.freeze.txt`, and Conda-package
metadata is `reproducibility/conda-packages.json`. A recorded `packaging @ file:...`
line is a build-origin inventory entry, not a portable install command; use the
curated reproduction requirements for installation. Upstream requirements are
preserved for attribution and are not the working installation recipe.

For comparable execution settings:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
```

## Restore public model artifacts

Release: [softmax-10landmark-working-v1](https://github.com/RulerOfEternalNight/JD-Factored-NeuS/releases/tag/softmax-10landmark-working-v1).
The release is a prerelease research snapshot, so use the exact tag rather than
requesting the repository's default/latest release.

Download `staged10-working-artifacts-v1.tar.gz` and `SHA256SUMS` into `downloads/`.
Run the restore commands in the root README. The helper checks member names,
types and all hashes before copying results, rejects path traversal/symlink
escapes, and refuses to overwrite different existing files. Identical existing
files are left intact. The artifact manifest is tracked in the branch.

Included:

- Baseline `ckpt_010000.pth` and baseline-10k visibility reference mesh.
- Baseline-30k comparison mesh and metadata.
- All eight original corrected checkpoints, from 12,500 through 30,000.
- Four intermediate geometry meshes and final normalized/world/marker meshes.
- Original scalar TensorBoard logs, final landmark JSON and float heatmap caches.
- All 80 periodic prediction maps, 80 targets and 80 comparison maps, plus stats.

Excluded by design:

- Native source photos, photo annotations and photographic reprojection overlays.
- RGB validation composites and other image diagnostics that may contain photos.
- Private dataset, source-photo review images and local machine credentials.

The original full output folder is not wholly public because those excluded
images contain private source photos. The exact **public model results** are
fully checksummed; private photographic diagnostics can be regenerated locally
when the original dataset is available.

## Private original inputs

Restore or copy the original training inputs into:

```text
public_data/122_1_masked/
  image/000.png ... 047.png
  mask/000.png ... 047.png
  cameras_sphere.npz
```

Use the full 1728 × 2592 images, not the separate historical quarter-resolution
test dataset. Image ordering, camera matrices, masks and original annotation
bytes must match `reproducibility/inputs_manifest.json`.

```bash
python tools/verify_reproduction.py --profile inputs
python tools/verify_reproduction.py --profile all
python tools/check_saved_model.py
```

The golden read-only check loads final iteration 30,000 in a temporary output
directory, renders retained camera-6 centers and compares the median true-class
response with **0.8673762083**, using a small floating-point tolerance. It does
not step an optimizer or modify archived checkpoints.

## Do not regenerate the archived visibility sidecar when resuming

`annotations/122_1_masked/staged10_visibility_baseline10k.json` is part of the
original checkpoint objective signature. Its metadata includes a historical
reference-mesh path. Keep its original bytes; the loader separately validates
the reference mesh using the current config path and its hash. Regenerating the
sidecar may change its semantic signature even if the validity decisions look
similar, causing a deliberate resume rejection. For a new dataset/objective,
initialize geometry as a new experiment instead of resuming this checkpoint.

## Repeat the original staged training recipe safely

```bash
python tools/run_staged10.py train --dry-run
python tools/run_staged10.py train
```

The wrapper changes only path locations and the new output destination. All
model/loss/schedule/sampling/validation hyperparameters remain original. Training
starts from baseline-10k reconstruction weights with a newly initialized head
and optimizer, rather than the final successful head. Fresh output is
`exp/122_1_masked/staged_10_reproduction/geometry`.

```bash
python tools/run_staged10.py resume
python tools/run_staged10.py validate
python tools/run_staged10.py export
```

`--max_steps 5` bounds an invocation without changing the 30k schedule. Fresh
training refuses an occupied output directory. Choose a new `--output` or resume
explicitly. Direct original launch remains available:

```bash
python exp_runner.py --conf confs/staged_10_corrected.conf \
  --case 122_1_masked --type dtu --gpu 0 --mode train
```

Use the wrapper for new training if archived artifacts already occupy the
original path. Reusing the original direct output directory would mix results.

## Evaluate or inspect restored results without retraining

The exact final numbers are already in `results/landmarks.json`. To recompute
mesh difference against baseline using restored model artifacts:

```bash
python scripts/evaluate_unified.py --exports \
  baseline=exp/122_1_masked/baseline/geometry/exports/00030000 \
  staged10=exp/122_1_masked/staged_10_corrected/geometry/exports/00030000 \
  --output verification/restored_comparison.json
```

This does not establish ground-truth geometry accuracy. Any export/rendering
regeneration from a checkpoint requires the private calibrated dataset and may
have floating-point differences; preserve the downloaded archive as the exact
historical reference.

## Local private backup

The original machine also retains a complete private reproduction backup under
`/workspace/private_reproduction_archives/staged10_corrected/`, containing the
source photos and photographic diagnostics as well as the original experiment.
This directory is outside the Git worktrees and is **not uploaded to GitHub**.
Its checksum and location are recorded locally. Keep a separate private copy of
that backup; a public Git branch cannot replace the private dataset.
