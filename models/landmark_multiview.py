"""Visibility-aware surface selection from predicted heatmaps only.

This module deliberately has no annotation or ground-truth input. Proposals come
from predicted 2D peaks intersected with the mesh and (optionally) raw field
peaks. A proposal must explain several independently rendered camera views.
"""
import logging
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


class MeshRayQueries:
    """Exact first intersections, accelerated when EmbreeX is available.

    Trimesh 3.21 only discovers the older pyembree package automatically;
    the current environment already supplies the maintained embreex wrapper.
    """
    def __init__(self, mesh, backend='auto'):
        if backend not in ('auto', 'embreex', 'trimesh'):
            raise ValueError('Unknown mesh ray backend')
        self.mesh, self.backend = mesh, 'trimesh'
        if backend != 'trimesh':
            try:
                from embreex import rtcore_scene
                from embreex.mesh_construction import TriangleMesh
            except ImportError:
                if backend == 'embreex':
                    raise
            else:
                self.scene = rtcore_scene.EmbreeScene()
                self.geometry = TriangleMesh(self.scene, np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                                             np.ascontiguousarray(mesh.faces, dtype=np.int32))
                self.backend = 'embreex'

    def intersects_location(self, origins, directions, multiple_hits=False):
        if multiple_hits:
            raise ValueError('Only first intersections are supported')
        if self.backend == 'trimesh':
            return self.mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        origins, directions = np.asarray(origins), np.asarray(directions)
        output = self.scene.run(np.ascontiguousarray(origins, dtype=np.float32),
                                np.ascontiguousarray(directions, dtype=np.float32), output=1)
        hit = (output['primID'] >= 0) & np.isfinite(output['tfar']) & (output['tfar'] >= 0)
        indices = np.flatnonzero(hit)
        locations = origins[hit] + directions[hit] * output['tfar'][hit, None]
        return locations, indices, output['primID'][hit]


def project_vertices(vertices, projection):
    projected = np.c_[vertices, np.ones(len(vertices))] @ np.asarray(projection)[:3].T
    depth = projected[:, 2]
    xy = projected[:, :2] / np.where(depth[:, None] > 1e-8, depth[:, None], np.nan)
    return xy, depth


def sample_heatmap(heatmap, xy, image_width, image_height):
    """Bilinear lookup matching Dataset.gen_rays_at's endpoint-inclusive grid."""
    h, w = heatmap.shape[:2]
    xy = np.asarray(xy, dtype=float)
    inside = (np.isfinite(xy).all(1) & (xy >= 0).all(1) &
              (xy[:, 0] <= image_width - 1) & (xy[:, 1] <= image_height - 1))
    safe = np.where(inside[:, None], xy, 0.)
    x = safe[:, 0] * (w - 1) / (image_width - 1)
    y = safe[:, 1] * (h - 1) / (image_height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    values = ((1-dx)*(1-dy)*heatmap[y0, x0] + dx*(1-dy)*heatmap[y0, x1] +
              (1-dx)*dy*heatmap[y1, x0] + dx*dy*heatmap[y1, x1])
    return np.where(inside, values, 0.), inside


def heatmap_peaks(heatmap, count, radius, threshold):
    """Greedy nonmaximum suppression; no annotation-centered search window."""
    work = np.array(heatmap, dtype=float, copy=True)
    peaks = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if work[y, x] < threshold or work[y, x] <= 0:
            break
        peaks.append((x, y))
        work[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = 0.
    return np.asarray(peaks, dtype=float).reshape(-1, 2)


def visible_vertices(mesh, vertices, camera_center, tolerance, chunk_size=16, ray_queries=None):
    """A vertex is visible only if it agrees with the first mesh intersection."""
    origins = np.broadcast_to(np.asarray(camera_center), vertices.shape)
    delta = vertices - origins
    distances = np.linalg.norm(delta, axis=1)
    directions = delta / np.maximum(distances[:, None], 1e-12)
    first = np.full(len(vertices), np.inf)
    ray_queries = MeshRayQueries(mesh) if ray_queries is None else ray_queries
    for start in range(0, len(vertices), chunk_size):
        end = min(start + chunk_size, len(vertices))
        loc, rays, _ = ray_queries.intersects_location(origins[start:end], directions[start:end], multiple_hits=False)
        if len(loc):
            first[start + rays] = np.linalg.norm(loc - origins[start + rays], axis=1)
    return np.isfinite(first) & (distances > 1e-8) & (np.abs(first - distances) <= tolerance)


@threadpool_limits.wrap(limits=1, user_api='blas')
def select_multiview_landmarks(mesh, probabilities, heatmaps, projections, camera_centers,
                               image_width, image_height, sigma, options=None, search_bounds=None):
    """Return mesh vertex selections or explicit abstentions, one per class.

    Corrected visible_mean scores only visible usable cameras; legacy all_views
    counts invisible cameras as zero. Raw field probability is not the score.
    Both an absolute and a per-view-relative threshold define supporting views.
    """
    o = dict(options or {})
    minimum = int(o.get('min_views', 3))
    min_response = float(o.get('min_response', .05))
    min_relative = float(o.get('min_relative_response', .5))
    min_fraction = float(o.get('min_support_fraction', .5))
    min_spread = float(o.get('min_camera_spread_degrees', 10.))
    contrast = float(o.get('min_peak_contrast', 2.))
    tolerance = float(o.get('visibility_tolerance', .02))
    n_peaks = int(o.get('peaks_per_view', 8))
    neighbors = int(o.get('proposal_neighbors', 32))
    radius = float(o.get('proposal_radius', .03))
    field_count = int(o.get('field_candidates', 32))
    visibility_chunk = int(o.get('visibility_chunk_size', 16))
    score_mode = str(o.get('score_mode', 'all_views'))
    domain = str(o.get('domain', 'unrestricted'))
    if score_mode not in ('all_views', 'visible_mean') or domain not in ('unrestricted', 'unit_sphere'):
        raise ValueError('Unknown multiview score mode or candidate domain')
    if minimum < 2 or not 0 < min_response <= 1 or not 0 < min_relative <= 1 or not 0 <= min_fraction <= 1:
        raise ValueError('Invalid multiview support thresholds')
    if tolerance <= 0 or radius <= 0 or n_peaks < 1 or neighbors < 1 or field_count < 0 or visibility_chunk < 1 or contrast < 1 or not 0 <= min_spread <= 180:
        raise ValueError('Invalid multiview proposal/visibility settings')
    vertices = np.asarray(mesh.vertices)
    probabilities = np.asarray(probabilities)
    heatmaps = [np.asarray(h) for h in heatmaps]
    k_count = probabilities.shape[1]
    v_count = len(heatmaps)
    if len(vertices) == 0 or probabilities.shape[0] != len(vertices) or v_count < minimum:
        raise ValueError('Mesh/probability shape mismatch or insufficient configured cameras')
    if len(projections) != v_count or len(camera_centers) != v_count or min(image_width, image_height) <= 1:
        raise ValueError('Camera dimensions/count mismatch')
    if any(h.ndim != 3 or h.shape[2] != k_count or min(h.shape[:2]) < 2 or
           not np.isfinite(h).all() or h.min() < 0 or h.max() > 1.0001 for h in heatmaps):
        raise ValueError('Heatmaps must be finite [H,W,K] probability responses')
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite field probabilities')
    sigma = np.broadcast_to(np.asarray(sigma), (k_count,))
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Sigma must be positive')
    maxima = np.array([h.max(axis=(0, 1)) for h in heatmaps])
    means = np.array([h.mean(axis=(0, 1)) for h in heatmaps])
    active = (maxima >= min_response) & (maxima >= contrast * np.maximum(means, 1e-8))
    tree = cKDTree(vertices)
    ray_queries = MeshRayQueries(mesh, str(o.get('ray_backend', 'auto')))
    pools = []
    for k in range(k_count):
        candidates = set()
        for vi, (heatmap, projection, center) in enumerate(zip(heatmaps, projections, camera_centers)):
            if not active[vi, k]:
                continue
            h, w = heatmap.shape[:2]
            suppression = max(1, int(np.ceil(.5*sigma[k]*min((w-1)/(image_width-1), (h-1)/(image_height-1)))))
            peaks = heatmap_peaks(heatmap[:, :, k], n_peaks, suppression, min_response)
            if not len(peaks):
                continue
            pixels = peaks * [(image_width-1)/(w-1), (image_height-1)/(h-1)]
            directions = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(np.asarray(projection)[:3, :3]).T
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            origins = np.broadcast_to(center, directions.shape)
            hits, _, _ = ray_queries.intersects_location(origins, directions, multiple_hits=False)
            if len(hits):
                distances, indices = tree.query(hits, k=min(neighbors, len(vertices)))
                indices, distances = np.asarray(indices).reshape(len(hits), -1), np.asarray(distances).reshape(len(hits), -1)
                candidates.update(indices[distances <= radius].tolist())
        if field_count:
            n = min(field_count, len(vertices))
            candidates.update(np.argpartition(probabilities[:, k], len(vertices)-n)[-n:].tolist())
        candidates = np.array(sorted(candidates), dtype=int)
        if domain == 'unit_sphere' and len(candidates):
            candidates = candidates[np.linalg.norm(vertices[candidates], axis=1) < 1.]
        if search_bounds is not None and search_bounds[k] is not None and len(candidates):
            low, high = np.asarray(search_bounds[k])
            candidates = candidates[((vertices[candidates] >= low) & (vertices[candidates] <= high)).all(1)]
        # An occlusion test can only remove supporting views. Reject proposals
        # that cannot meet the minimum even before visibility; this preserves
        # the selection rule while avoiding costly queries for isolated peaks.
        if len(candidates):
            potential = np.zeros(len(candidates), dtype=int)
            for vi, (heatmap, projection) in enumerate(zip(heatmaps, projections)):
                if not active[vi, k]:
                    continue
                xy, depth = project_vertices(vertices[candidates], projection)
                response, inside = sample_heatmap(heatmap[:, :, k], xy, image_width, image_height)
                potential += inside & (depth > 1e-8) & (response >= min_response) & (response >= min_relative * maxima[vi, k])
            candidates = candidates[potential >= minimum]
        pools.append(candidates)
    union = np.unique(np.concatenate(pools))
    logging.getLogger(__name__).info('Multiview surface proposals per landmark: %s; %d unique vertices',
                                    [len(p) for p in pools], len(union))
    if len(union):
        positions = vertices[union]
        visible = []
        for i, center in enumerate(camera_centers):
            logging.getLogger(__name__).info('Checking mesh visibility in camera %d/%d', i+1, v_count)
            visible.append(visible_vertices(mesh, positions, center, tolerance, visibility_chunk, ray_queries))
        visibility = np.array(visible)
        projections_xy = [project_vertices(positions, p) for p in projections]
    selections = []
    for k, candidates in enumerate(pools):
        if not len(candidates):
            selections.append(dict(valid=False, reason='No surface proposals with sufficient heatmap support in search region', active_views=int(active[:, k].sum())))
            continue
        columns = np.searchsorted(union, candidates)
        values, seen = [], []
        for vi, heatmap in enumerate(heatmaps):
            xy, depth = projections_xy[vi]
            response, inside = sample_heatmap(heatmap[:, :, k], xy[columns], image_width, image_height)
            values.append(response)
            seen.append(visibility[vi, columns] & inside & (depth[columns] > 1e-8) & active[vi, k])
        values, seen = np.array(values), np.array(seen)
        relative = values / np.maximum(maxima[:, k, None], 1e-8)
        support = seen & (values >= min_response) & (relative >= min_relative)
        counts, visible_counts = support.sum(0), seen.sum(0)
        fractions = counts / np.maximum(visible_counts, 1)
        denominator = np.maximum(visible_counts, 1) if score_mode == 'visible_mean' else max(int(active[:, k].sum()), 1)
        scores = (values * seen).sum(0) / denominator
        # Several nearly coincident cameras are not independent angular support.
        toward_camera = np.asarray(camera_centers)[:, None] - vertices[candidates][None]
        toward_camera /= np.maximum(np.linalg.norm(toward_camera, axis=2, keepdims=True), 1e-12)
        cosines = np.einsum('vic,wic->vwi', toward_camera, toward_camera)
        pairs = support[:, None] & support[None]
        spread = np.degrees(np.arccos(np.clip(np.where(pairs, cosines, 1.).min(axis=(0, 1)), -1., 1.)))
        acceptable = (counts >= minimum) & (fractions >= min_fraction) & (spread >= min_spread)
        if not acceptable.any():
            selections.append(dict(valid=False, reason='Insufficient visible multiview heatmap agreement',
                                   active_views=int(active[:, k].sum()), maximum_supporting_views=int(counts.max()),
                                   maximum_camera_spread_degrees=float(spread.max())))
            continue
        winner = int(np.argmax(np.where(acceptable, scores, -np.inf)))
        selections.append(dict(valid=True, vertex_index=int(candidates[winner]),
                               confidence=float(scores[winner]),
                               confidence_type='mean_response_over_visible_cameras' if score_mode == 'visible_mean' else 'mean_predicted_response_with_visibility',
                               raw_probability=float(probabilities[candidates[winner], k]),
                               supporting_views=np.flatnonzero(support[:, winner]).tolist(),
                               visible_views=np.flatnonzero(seen[:, winner]).tolist(),
                               active_views=int(active[:, k].sum()),
                               camera_spread_degrees=float(spread[winner]),
                               support_fraction=float(fractions[winner])))
    for selection in selections:
        selection['visibility_backend'] = ray_queries.backend
    return selections
