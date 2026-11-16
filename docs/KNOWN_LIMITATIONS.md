# Known limitations of this preserved version

- **Eight of ten landmarks**, not all ten, were recovered. Forehead and chin
  remain unresolved; their errors/coordinates must remain null.
- Headline error is agreement with original MediaPipe detections in training
  views. There is no held-out landmark protocol, independent 3D landmark ground
  truth, or reference surface mesh.
- A mesh change from baseline does not establish improved geometry accuracy.
  World units are input units and are not asserted to be millimeters.
- Periodic landmark maps use camera 1 only; RGB validation selects random views.
  Final extraction uses 12 cameras.
- Multi-view agreement is implemented during extraction, not as an explicit
  training loss. Confidence is not guaranteed anatomical correctness.
- The class-zero correction, domain restriction, new ranking and visibility
  review were enabled together; their individual causal effects were not isolated.
- Joint-stage sampling introduces cameras with no landmark labels, lowering
  all-batch landmark loss. Compare supervised-batch curves and fixed validation.
- The visibility sidecar depends on baseline geometry and inferred occlusion.
  Regenerating it in a new checkout can change checkpoint compatibility.
- One subject and one seed were tested. This branch does not claim the same
  recipe works for dense 478 landmarks. It intentionally preserves the sparse
  successful snapshot rather than later dense experiments or proposed sigmoid work.
- Seed 42 is set, but cross-platform bitwise deterministic retraining was not
  established. Exact archived outputs can be restored using published hashes.
- Source photos and photographic overlays are kept private. Data manifests do
  not contain the actual images. Supply the original private inputs to retrain.
