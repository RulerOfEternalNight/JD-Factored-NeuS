"""Training-only visibility review using calibrated labels and a fixed reference mesh.

No detector coordinates are edited. The returned validity mask can exclude
uncertain supervision; it must not be used to fabricate negative detections.
"""
import numpy as np
import hashlib
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from models.landmark_multiview import project_vertices, visible_vertices, MeshRayQueries


def mask_fingerprint(data_dir, image_names):
    digest = hashlib.sha256()
    for name in image_names:
        path = Path(data_dir) / 'mask' / name
        digest.update(name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest() if path.exists() else b'missing-mask')
    return digest.hexdigest()


@threadpool_limits.wrap(limits=1, user_api='blas')
def review_visibility(labels, projections, camera_centers, mesh, masks=None,
                      tolerance=.02, maximum_residual=60., maximum_surface_distance=.05,
                      minimum_views=3, robust_scale=20.):
    if tolerance <= 0 or maximum_residual <= 0 or maximum_surface_distance <= 0 or minimum_views < 2:
        raise ValueError('Invalid annotation visibility thresholds')
    original = labels.source_valid
    allowed = original.copy()
    reasons = [[[] for _ in labels.ids] for _ in labels.image_names]
    vertices = np.asarray(mesh.vertices)
    if not len(vertices):
        raise ValueError('Visibility reference mesh is empty')
    tree, queries = cKDTree(vertices), MeshRayQueries(mesh)
    anchors = []
    for k, landmark_id in enumerate(labels.ids):
        ix = np.flatnonzero(original[:, k])
        point = surface = None
        residual = np.full(len(labels.image_names), np.nan)
        visible = np.zeros(len(labels.image_names), dtype=bool)
        distance = None
        if len(ix) >= minimum_views:
            P = np.asarray(projections)[ix]; xy = labels.xy[ix, k]
            A = np.concatenate([xy[:, 0, None]*P[:, 2]-P[:, 0], xy[:, 1, None]*P[:, 2]-P[:, 1]])
            _, _, vt = np.linalg.svd(A)
            if abs(vt[-1, 3]) > 1e-10:
                initial = vt[-1, :3] / vt[-1, 3]
                def errors(x):
                    p = np.einsum('vij,j->vi', P, np.r_[x, 1.])
                    safe = np.where(np.abs(p[:, 2:3]) > 1e-8, p[:, 2:3], 1e-8)
                    return (p[:, :2]/safe - xy).ravel()
                fit = least_squares(errors, initial, loss='soft_l1', f_scale=robust_scale)
                point = fit.x
                distance, nearest = tree.query(point)
                surface = vertices[int(nearest)]
                for i, projection in enumerate(projections):
                    projected, depth = project_vertices(surface[None], projection)
                    residual[i] = np.linalg.norm(projected[0]-labels.xy[i,k]) if depth[0] > 0 else np.inf
                    visible[i] = visible_vertices(mesh, surface[None], camera_centers[i], tolerance,
                                                  ray_queries=queries)[0]
        for i in range(len(labels.image_names)):
            why = reasons[i][k]
            if not original[i, k]:
                why.append('missing_or_invalid_detection')
            elif surface is None:
                why.append('insufficient_calibrated_support')
            else:
                if distance > maximum_surface_distance:
                    why.append('reference_surface_too_far')
                if np.linalg.norm(surface) >= 1.:
                    why.append('anchor_outside_landmark_domain')
                if not np.isfinite(residual[i]) or residual[i] > maximum_residual:
                    why.append('multiview_reprojection_outlier')
                if not visible[i]:
                    why.append('mesh_inferred_occlusion')
                if masks is not None and masks[i] is not None:
                    x, y = np.rint(labels.xy[i, k]).astype(int)
                    if not masks[i][y, x]:
                        why.append('foreground_mask_conflict')
            allowed[i, k] = original[i, k] and not why
        anchors.append(dict(id=landmark_id, name=labels.names[k],
                            triangulated_normalized=None if point is None else point.tolist(),
                            surface_normalized=None if surface is None else surface.tolist(),
                            surface_distance_normalized=None if distance is None else float(distance),
                            original_views=int(original[:,k].sum()), retained_views=int(allowed[:,k].sum()),
                            reprojection_errors_pixels=[float(x) if np.isfinite(x) else None for x in residual]))
    return dict(supervision_valid=allowed.tolist(), reasons=reasons, anchors=anchors,
                parameters=dict(visibility_tolerance=tolerance, maximum_reprojection_error=maximum_residual,
                                maximum_surface_distance=maximum_surface_distance,
                                minimum_views=minimum_views, robust_scale=robust_scale),
                caveat='Mesh-inferred review flags, not verified anatomical visibility. Original coordinates are retained.')
