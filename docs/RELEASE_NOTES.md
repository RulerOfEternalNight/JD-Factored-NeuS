# Corrected staged 10-landmark softmax research snapshot

Branch: `softmax_10landmark_working`.

Preserves the 64-file source/config snapshot used by `staged_10_corrected`.
Result: 8/10 selected, mean 19.3065 native pixels and median 12.3729 pixels on
selected landmarks. Forehead and chin are unresolved. Results use MediaPipe
training-view detections and are not independent 3D accuracy measurements.

The public archive includes baseline initialization/reference geometry, all
eight corrected checkpoints, intermediate and final meshes, original scalar
logs, landmark JSON, float heatmaps and periodic prediction/target heatmaps.
Every file has a tracked SHA-256 entry. Source photos and photographic overlays
are excluded at the user's request.

Restore using `tools/restore_artifacts.py`, then verify the artifacts. Supply
the private original dataset to run the golden inference check or repeat
training. Original model/training files remain byte-identical; added wrappers
use a separate reproduction directory and reject unsafe overwrites.

This research prerelease intentionally preserves partial sparse success. It
does not contain the later dense sigma experiments or a sigmoid implementation.
