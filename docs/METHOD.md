# Method and original training setup

## Shared geometry and landmark field

The existing SDF produces 257 outputs: one SDF value plus 256 features. A separate
MLP maps those features through widths 128 and 64 to **11 softmax classes**:
class zero is non-landmark, and the remaining classes are the ten configured IDs.
No independent geometry network, explicit feature grid or second ray renderer
is introduced.

```text
(s, f) = F_SDF(x)
q = softmax(F_landmark(f))
H_k(ray) = sum_i weight_i(ray) * q_i,k
L_joint = L_original_reconstruction + lambda * (L_heatmap + 0.1 L_nonlandmark)
```

The renderer uses its existing 64 uniform plus 64 importance samples. Landmark
contributions are restricted to the unit sphere, independently of outside-NeRF
sampling. RGB behavior is preserved. Main heatmap gradients flow through both
shared features and SDF-derived rendering weights.

The auxiliary non-landmark loss uses stable log-softmax on sufficiently opaque
known-negative foreground rays. Its normalized visibility weights are detached
to avoid rewarding removal of surface opacity. It does not detach the main
heatmap objective. Missing/excluded observations are never turned into negative
labels. The hard-negative cache is saved with optimizer/RNG state.

## Exact selected landmarks and class order

| Head class | MediaPipe ID | Name |
|---:|---:|---|
| 0 | — | Non-landmark |
| 1 | 1 | Nose tip |
| 2 | 33 | Outer eye corner |
| 3 | 291 | Mouth corner |
| 4 | 263 | Outer eye corner |
| 5 | 133 | Inner eye corner |
| 6 | 362 | Inner eye corner |
| 7 | 61 | Mouth corner |
| 8 | 10 | Forehead center |
| 9 | 168 | Nose bridge |
| 10 | 152 | Chin |

## Schedule and hyperparameters

| Setting | Original value |
|---|---|
| Initialization | Baseline geometry checkpoint `ckpt_010000.pth` |
| Frozen reconstruction / head-only training | 10k–15k |
| Joint geometry, appearance and landmark training | 15k–30k |
| Random seed | 42 |
| Batch | 512 rays: 384 general, 96 landmark-centered, 32 extra negatives |
| Hard-negative selection | Up to 24 of 32 extra negatives; cache capacity 1,024 per camera |
| Gaussian sigma | 20 **native** image pixels |
| Heatmap loss | Balanced weighted L1, positive:negative 1:1 |
| Auxiliary non-landmark weight / minimum opacity | 0.1 / 0.5 |
| Initial/final outer landmark lambda | 0.01 / 0.1; ramp in first 1k joint iterations |
| Nominal learning rate | 0.0005 with original scheduling |
| Training image size | 1728 × 2592; no image resizing for this run |
| Landmark validation | Every 2,500 iterations, camera index 1, 432 × 648 |
| Mesh validation | Every 5,000 iterations |
| Final mesh grid / field query chunk | 256 / 8,192 |
| Multi-view extraction | 12 cameras; at least 3 supporting views |

All exact settings are in the frozen configs and original launch configuration.
The staged run has 25k total geometry updates (10k baseline plus 15k joint) and
20k head updates. The iteration counter alone is not a geometry-update budget.

## Annotation validity and coordinates

Original ten-ID detections: 350 observations across 35 of 48 images. A fixed
baseline-10k geometry visibility review retains **284** observations. Coordinates
are not edited. This review uses calibrated triangulation, baseline mesh
visibility, masks and residual checks; its flags are inferred, not ground truth.

The SDF and landmark head are queried in normalized scene coordinates. Exported
world coordinates use `scale_mat_0`. Projection of normalized points uses
`world_mat_v @ scale_mat_v`. Never query world-coordinate mesh vertices directly
without the inverse normalization transform.

## Multi-view agreement: scope

Training supervises a shared 3D field from different cameras, but has **no explicit
training-time cross-view consistency loss**. Final extraction samples the learned
field on the mesh and uses rendered heatmaps from 12 cameras, visibility tests,
support fraction, angular spread and peak evidence. Scores average responses
over visible cameras. This is an extraction rule, not independently calibrated
anatomical confidence. A self-consistent incorrect blob is still possible.
