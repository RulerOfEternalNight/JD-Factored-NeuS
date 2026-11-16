"""Feature-based landmark field, pixel targets, annotations, and training schedule."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn


class LandmarkHead(nn.Module):
    def __init__(self, num_landmarks, feature_dim=256, hidden_dims=(128, 64)):
        super().__init__()
        if num_landmarks < 1:
            raise ValueError('num_landmarks must be positive')
        dims = [feature_dim, *hidden_dims, num_landmarks + 1]
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def aggregate_landmarks(probabilities, weights):
    """Weights already aligned to SDF samples; channel zero is non-landmark."""
    if probabilities.shape[:-1] != weights.shape:
        raise ValueError('SDF feature/weight sample alignment mismatch')
    return (weights[..., None] * probabilities[..., 1:]).sum(dim=-2)


def heatmap_targets(pixels, xy, valid, sigma):
    sigma = torch.as_tensor(sigma, dtype=pixels.dtype, device=pixels.device)
    if (sigma <= 0).any():
        raise ValueError('Gaussian sigma must be positive')
    distance2 = (pixels[:, None] - xy[None]).square().sum(-1)
    return torch.exp(-distance2 / (2 * sigma.square())) * valid[None]


def heatmap_loss(prediction, target, valid, positive_weight=10., negative_weight=1., balance_positive_negative=True):
    """Soft positive weighting avoids a hard target threshold; missing labels ignored."""
    if positive_weight <= 0 or negative_weight <= 0:
        raise ValueError('Both heatmap weights must be positive')
    if balance_positive_negative:
        # Give scarce positive mass a controlled share of each class's loss.
        # Without balancing, weighted L1 can still prefer an almost-zero field
        # when uniform negatives vastly outnumber narrow Gaussian peaks.
        positive = target / target.sum(0, keepdim=True).clamp_min(1e-8)
        negative = (1 - target) / (1 - target).sum(0, keepdim=True).clamp_min(1e-8)
        weights = (positive_weight * positive + negative_weight * negative) * valid[None]
    else:
        weights = (negative_weight + (positive_weight - negative_weight) * target) * valid[None]
    numerator = (weights * (prediction - target).abs()).sum(0)
    denominator = weights.sum(0)
    per_class = numerator / denominator.clamp_min(1e-8)
    return numerator.sum() / denominator.sum().clamp_min(1e-8), per_class


class LandmarkAnnotations:
    """Filename keyed labels in calibrated, undistorted training-image coordinates.

    Existing MediaPipe v1 JSON is accepted. Missing images/points are invalid,
    not negative detections. Optional pixel_transform maps source pixels to the
    training frame (resize/crop only), with matching calibrated image dimensions.
    """
    def __init__(self, path, image_paths, width, height, ids, names=None):
        self.ids = list(ids)
        self.names = list(names or [str(i) for i in ids])
        if len(self.names) != len(ids) or len(set(map(str, ids))) != len(ids) or len(set(self.names)) != len(ids):
            raise ValueError('Landmark names/IDs must have matching, unique entries')
        raw = json.loads(Path(path).read_text())
        if raw.get('distorted', False):
            raise ValueError('Undistort annotations and images using the camera calibration before training')
        coordinates = raw.get('coordinate_system', 'pixels')
        if coordinates not in ('pixels', 'normalized'):
            raise ValueError('coordinate_system must be pixels or normalized')
        self.xy = np.zeros((len(image_paths), len(ids), 2), dtype=np.float32)
        self.valid = np.zeros((len(image_paths), len(ids)), dtype=bool)
        self.source = str(path)
        self.image_names = [Path(p).name for p in image_paths]
        self.width, self.height = width, height
        images = raw['images']
        for j, filename in enumerate(image_paths):
            row = images.get(Path(filename).name)
            if row is None:
                continue
            if 'camera_index' in row and int(row['camera_index']) != j:
                raise ValueError('Annotation camera_index does not match sorted image/calibration index')
            sw, sh = int(row['width']), int(row['height'])
            transform = np.asarray(row.get('pixel_transform', np.eye(3)), dtype=np.float64)
            if transform.shape != (3, 3) or not np.isfinite(transform).all() or not np.allclose(transform[2], [0, 0, 1]):
                raise ValueError('pixel_transform must be a finite affine 3x3 matrix')
            if 'pixel_transform' in row:
                if row.get('training_width') != width or row.get('training_height') != height:
                    raise ValueError('Transformed annotation training dimensions mismatch')
            elif (sw, sh) != (width, height):
                raise ValueError(f'Annotation dimensions mismatch for {filename}; supply explicit pixel_transform')
            points = {}
            for point in row.get('landmarks', []):
                for key in ('id', 'name'):
                    if key in point:
                        points[str(point[key])] = point
            for k, (landmark_id, name) in enumerate(zip(ids, self.names)):
                point = points.get(str(landmark_id), points.get(name))
                if point is None or not point.get('valid', True) or (point.get('visibility') is not None and float(point['visibility']) <= 0) or point.get('xy') is None:
                    continue
                xy = np.asarray(point['xy'], dtype=float)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    continue
                if coordinates == 'normalized':
                    xy *= [sw, sh]
                xy = (transform @ np.r_[xy, 1.])[:2]
                if 0 <= xy[0] < width and 0 <= xy[1] < height:
                    self.xy[j, k] = xy
                    self.valid[j, k] = True

        # Preserve detector support separately from visibility-filtered training
        # validity. Excluded/missing landmarks must not become negative labels.
        self.source_valid = self.valid.copy()
        self.visibility_hash = None

    def apply_visibility_filter(self, path):
        payload = Path(path).read_bytes()
        cache = json.loads(payload)
        if (cache['ids'] != self.ids or cache['image_names'] != self.image_names or
                cache['native_size'] != [self.width, self.height]):
            raise ValueError('Visibility cache does not match landmark IDs/images/dimensions')
        if cache['annotation_sha256'] != hashlib.sha256(Path(self.source).read_bytes()).hexdigest():
            raise ValueError('Visibility cache annotations changed; regenerate the cache')
        allowed = np.asarray(cache['supervision_valid'], dtype=bool)
        if allowed.shape != self.valid.shape:
            raise ValueError('Visibility cache has incorrect shape')
        self.valid = self.source_valid & allowed
        semantic = {k: v for k, v in cache.items() if k != 'created_utc'}
        self.visibility_hash = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        return cache

    def targets(self, image_index, pixels, sigma):
        xy = torch.as_tensor(self.xy[image_index], device=pixels.device)
        valid = torch.as_tensor(self.valid[image_index], device=pixels.device)
        return heatmap_targets(pixels, xy, valid, sigma), valid


def stage_at(mode, step, geometry_iters=0, head_iters=0):
    if mode == 'baseline':
        return 'geometry'
    if mode == 'joint_from_zero':
        return 'joint'
    if mode != 'staged':
        raise ValueError('Unknown training strategy: ' + mode)
    if step < geometry_iters:
        return 'geometry'
    if step < geometry_iters + head_iters:
        return 'head'
    return 'joint'


def sample_pixels(width, height, batch_size, xy, valid, sigma, center_fraction=.1875, negative_fraction=.0625, center_sigma=1.,
                  foreground_mask=None, negative_pool=None, hard_fraction=0.,
                  negative_xy=None, negative_valid=None):
    """Integer pixels: uniform + Gaussian-centered + rejection-sampled negatives.

    Fall back to uniform where labels or negative regions are unavailable.
    Sampling intentionally changes the supervised ray distribution; no claim of
    an unbiased full-image heatmap objective is made.
    """
    if min(center_fraction, negative_fraction) < 0 or center_fraction + negative_fraction > 1:
        raise ValueError('Sampling fractions must be nonnegative and sum to <= 1')
    if not 0 <= hard_fraction <= 1:
        raise ValueError('Hard-negative fraction must be in [0,1]')
    if foreground_mask is not None and foreground_mask.shape != (height, width):
        raise ValueError('Foreground sampling mask dimensions mismatch')
    valid_indices = np.flatnonzero(valid)
    pixels = np.column_stack([np.random.randint(width, size=batch_size), np.random.randint(height, size=batch_size)])
    if not len(valid_indices):
        return pixels
    sigmas = np.broadcast_to(np.asarray(sigma), (len(valid),))
    n_center = round(batch_size * center_fraction)
    n_negative = min(batch_size - n_center, round(batch_size * negative_fraction))
    chosen = np.random.choice(valid_indices, n_center)
    centered = xy[chosen] + np.random.normal(size=(n_center, 2)) * sigmas[chosen, None] * center_sigma
    pixels[:n_center] = np.rint(centered).astype(int)
    avoid_xy = xy if negative_xy is None else negative_xy
    avoid_valid = valid if negative_valid is None else negative_valid
    def acceptable(candidate):
        x, y = map(int, candidate)
        return (0 <= x < width and 0 <= y < height and
                (foreground_mask is None or foreground_mask[y, x] > .5) and
                (((np.asarray([x, y]) - avoid_xy[avoid_valid]) ** 2).sum(-1) >
                 (3 * sigmas[avoid_valid]) ** 2).all())
    hard = []
    if negative_pool is not None and len(negative_pool) and hard_fraction:
        pool = np.asarray([row for row in negative_pool if acceptable(row[:2])], dtype=float).reshape(-1, 3)
        if len(pool):
            priorities = np.maximum(pool[:, 2], 1e-6)
            chosen = np.random.choice(len(pool), round(n_negative * hard_fraction),
                                      replace=True, p=priorities/priorities.sum())
            hard = pool[chosen, :2].astype(int)
    # Require negative samples to be at least 3 sigma from every valid point.
    for i in range(n_center, n_center + n_negative):
        if i - n_center < len(hard):
            pixels[i] = hard[i - n_center]
            continue
        for _ in range(64):
            candidate = [np.random.randint(width), np.random.randint(height)]
            if acceptable(candidate):
                pixels[i] = candidate
                break
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    np.random.shuffle(pixels)
    return pixels
