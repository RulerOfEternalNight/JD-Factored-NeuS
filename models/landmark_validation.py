"""Heatmap diagnostics and normalized/world-coordinate surface landmark export."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
import logging
from models.landmarks import heatmap_targets
from models.landmark_multiview import select_multiview_landmarks


def transform_points(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :3] / homogeneous[:, 3:4]


class LandmarkValidationMixin:
    def render_landmark_response_map(self, idx, level):
        """Render existing landmark-weight aggregation, collecting maps on CPU."""
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=level)
        h, w = rays_o.shape[:2]
        responses = []
        # Spatial derivatives are needed by NeuS even during inference.
        for origins, directions in zip(rays_o.reshape(-1, 3).split(self.batch_size),
                                       rays_d.reshape(-1, 3).split(self.batch_size)):
            near, far = self.dataset.near_far_from_sphere(origins, directions)
            out = self.renderer.render(origins, directions, near, far, perturb_overwrite=0,
                                       cos_anneal_ratio=self.get_cos_anneal_ratio())
            responses.append(out['landmark_response'].detach().cpu())
        return torch.cat(responses).reshape(h, w, -1)

    def validate_landmarks(self, idx=0):
        if self.landmark_head is None:
            return
        c = self.landmark_conf
        level = int(c.get('validation_resolution_level', 16))
        if level < 1 or min(self.dataset.H, self.dataset.W) // level < 1:
            raise ValueError('Invalid landmark validation resolution level')
        prediction_map = self.render_landmark_response_map(idx, level)
        h, w = prediction_map.shape[:2]
        # Match gen_rays_at's exact linspace sampling, rather than resize assumptions.
        y, x = torch.meshgrid(torch.linspace(0, self.dataset.H - 1, h, device=self.device),
                              torch.linspace(0, self.dataset.W - 1, w, device=self.device), indexing='ij')
        pixels = torch.stack([x, y], -1).reshape(-1, 2)
        prediction = prediction_map.reshape(-1, prediction_map.shape[-1])
        target, valid = self.annotations.targets(idx, pixels, c.get('sigma', 12.))
        directory = Path(self.base_exp_dir) / 'landmark_validation' / f'{self.iter_step:08d}_{idx:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        photo = cv2.imread(self.dataset.images_lis[idx])
        stats = []
        for k, name in enumerate(self.landmark_names):
            # Filenames use integer class indices; semantic names may contain slashes.
            pred = prediction[:, k].reshape(h, w).cpu().numpy()
            gt = target[:, k].reshape(h, w).cpu().numpy()
            pred_image = cv2.applyColorMap(np.uint8(np.clip(pred, 0, 1) * 255), cv2.COLORMAP_JET)
            gt_image = cv2.applyColorMap(np.uint8(np.clip(gt, 0, 1) * 255), cv2.COLORMAP_JET)
            cv2.imwrite(str(directory / f'{k}_prediction.png'), pred_image)
            cv2.imwrite(str(directory / f'{k}_target.png'), gt_image)
            cv2.imwrite(str(directory / f'{k}_comparison.png'), np.concatenate([pred_image, gt_image], 1))
            if valid[k]:
                point = tuple(np.rint(self.annotations.xy[idx, k]).astype(int))
                cv2.circle(photo, point, 9, (0, 255, 0), 2)
                cv2.putText(photo, name, (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            elif getattr(self.annotations, 'source_valid', self.annotations.valid)[idx, k]:
                point = tuple(np.rint(self.annotations.xy[idx, k]).astype(int))
                cv2.circle(photo, point, 9, (0, 165, 255), 2)
                cv2.putText(photo, name + ' (excluded)', (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 165, 255), 2)
            stats.append(dict(id=self.landmark_ids[k], name=name, valid=bool(valid[k]),
                              source_detection_valid=bool(getattr(self.annotations, 'source_valid', self.annotations.valid)[idx, k]),
                              target_supervised=bool(valid[k]),
                              mean=float(pred.mean()), maximum=float(pred.max()), target_maximum=float(gt.max())))
        cv2.imwrite(str(directory / 'annotations.png'), photo)
        (directory / 'statistics.json').write_text(json.dumps(stats, indent=2))

    def select_multiview_surface_landmarks(self, vertices, triangles, probabilities, directory):
        """Select views without consulting annotation coordinates or validity."""
        c = self.landmark_conf
        options = c.get('multiview', {})
        n = len(self.dataset.images_lis)
        if 'view_indices' in options:
            views = [int(i) for i in options['view_indices']]
            if len(set(views)) != len(views) or any(i < 0 or i >= n for i in views):
                raise ValueError('Invalid/duplicate multiview camera indices')
        else:
            maximum = int(options.get('max_views', 12))
            if maximum < 2:
                raise ValueError('Multiview extraction requires at least two cameras')
            views = np.linspace(0, n-1, min(n, maximum), dtype=int).tolist()
        minimum = int(options.get('min_views', 3))
        if len(views) < minimum:
            raise ValueError('Fewer selected cameras than multiview.min_views')
        level = int(options.get('resolution_level', c.get('validation_resolution_level', 4)))
        if level < 1 or min(self.dataset.H, self.dataset.W) // level < 2:
            raise ValueError('Invalid multiview resolution level')
        maps, projections, centers = [], [], []
        for i in views:
            logging.info('Rendering multiview landmark heatmap for camera %d (%d/%d), resolution level %d',
                         i, len(maps)+1, len(views), level)
            (directory / 'multiview_progress.json').write_text(json.dumps(
                dict(phase='rendering',camera=i,completed_views=len(maps),total_views=len(views)),indent=2))
            prediction = self.render_landmark_response_map(i, level).numpy()
            maps.append(prediction)
            projections.append((self.dataset.world_mats_np[i] @ self.dataset.scale_mats_np[i])[:3])
            centers.append(self.dataset.pose_all[i, :3, 3].detach().cpu().numpy())
            if bool(options.get('save_heatmaps', False)):
                target = directory / 'multiview_heatmaps'
                target.mkdir(exist_ok=True)
                np.savez_compressed(target / f'{i:03d}.npz', prediction=prediction,
                                    native_size=np.array([self.dataset.W, self.dataset.H]),
                                    landmark_ids=np.array(self.landmark_ids), camera_index=i,
                                    projection=np.asarray(projections[-1]), iteration=self.iter_step,
                                    render_domain=str(c.get('render_domain', 'legacy')))
        boxes = c.get('search_bounds', {})
        bounds = [boxes.get(str(i), boxes.get(name)) for i, name in zip(self.landmark_ids, self.landmark_names)]
        selections = select_multiview_landmarks(
            trimesh.Trimesh(vertices, triangles, process=False), probabilities, maps,
            projections, centers, self.dataset.W, self.dataset.H,
            c.get('sigma', 12.), options=options, search_bounds=bounds)
        (directory / 'multiview_progress.json').write_text(json.dumps(
            dict(phase='selected',completed_views=len(maps),total_views=len(views)),indent=2))
        for selection in selections:
            for key in ('supporting_views', 'visible_views'):
                if key in selection:
                    selection[key] = [views[i] for i in selection[key]]
        metadata = dict(method='multiview', view_indices=views, resolution_level=level,
                        heatmap_size=[maps[0].shape[1], maps[0].shape[0]],
                        visibility_backend=selections[0]['visibility_backend'] if selections else None,
                        uses_annotation_coordinates=False, uses_annotation_visibility=False,
                        options=dict(options),
                        confidence_caveat='Model agreement score, not calibrated anatomical confidence')
        return selections, metadata

    def export_surface_landmarks(self, resolution=None, output_dir=None, reproject=True):
        c = self.landmark_conf
        resolution = resolution or int(c.get('export_resolution', 256))
        directory = Path(output_dir or Path(self.base_exp_dir) / 'exports' / f'{self.iter_step:08d}')
        directory.mkdir(parents=True, exist_ok=True)
        vertices, triangles = self.renderer.extract_geometry(
            torch.as_tensor(self.dataset.object_bbox_min, dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dataset.object_bbox_max, dtype=torch.float32, device=self.device), resolution, 0.)
        if len(vertices) == 0:
            raise ValueError('No zero-level surface exists; train geometry before exporting')
        scale = self.dataset.scale_mats_np[0]
        world_vertices = transform_points(vertices, scale)
        trimesh.Trimesh(world_vertices, triangles, process=False).export(directory / 'mesh_world.ply')
        trimesh.Trimesh(vertices, triangles, process=False).export(directory / 'mesh_normalized.ply')
        result = dict(iteration=self.iter_step, strategy=self.training_strategy,
                      coordinates='world', field_query_coordinates='normalized',
                      normalized_to_world=scale.tolist(), mesh_resolution=resolution,
                      mesh='mesh_world.ply', landmarks=[],
                      geometry_metrics=dict(chamfer_distance=None, reason='No reference mesh supplied'))
        if self.landmark_head is not None:
            probabilities = []
            with torch.no_grad():
                for chunk in torch.as_tensor(vertices, dtype=torch.float32, device=self.device).split(int(c.get('export_chunk_size', 8192))):
                    probabilities.append(self.landmark_head(self.sdf_network(chunk)[:, 1:]).softmax(-1)[:, 1:].cpu().numpy())
            probabilities = np.concatenate(probabilities)
            method = str(c.get('extraction_method', 'raw_argmax'))
            if method == 'multiview':
                selections, metadata = self.select_multiview_surface_landmarks(vertices, triangles, probabilities, directory)
                result['selection'] = metadata
            elif method != 'raw_argmax':
                raise ValueError('Unknown landmark extraction method: ' + method)
            else:
                result['selection'] = dict(method='raw_argmax')
            boxes = c.get('search_bounds', {})
            for k, name in enumerate(self.landmark_names):
                if method == 'multiview':
                    row = dict(id=self.landmark_ids[k], name=name, **selections[k])
                    if row['valid']:
                        index = row['vertex_index']
                        row.update(normalized=vertices[index].tolist(), world=world_vertices[index].tolist(), surface_vertex_distance=0.)
                    else:
                        row.update(vertex_index=None, normalized=None, world=None, confidence=0., surface_vertex_distance=None)
                    result['landmarks'].append(row)
                    continue
                candidates = np.arange(len(vertices))
                box = boxes.get(str(self.landmark_ids[k]), boxes.get(name))
                if box is not None:
                    low, high = np.asarray(box)
                    candidates = candidates[((vertices >= low) & (vertices <= high)).all(-1)]
                if not len(candidates):
                    raise ValueError('No mesh vertices in landmark search bounds for ' + name)
                index = int(candidates[np.argmax(probabilities[candidates, k])])
                result['landmarks'].append(dict(id=self.landmark_ids[k], name=name, vertex_index=index,
                                               normalized=vertices[index].tolist(), world=world_vertices[index].tolist(),
                                               confidence=float(probabilities[index, k]), surface_vertex_distance=0.))
            valid_rows = [row for row in result['landmarks'] if row.get('valid', True)]
            points = np.asarray([row['world'] for row in valid_rows], dtype=float).reshape(-1, 3)
            trimesh.points.PointCloud(points).export(directory / 'landmarks_world.ply')
            radius = float(c.get('marker_radius', .01))
            marked_meshes = [trimesh.Trimesh(world_vertices, triangles, process=False)]
            # Radius is specified in normalized coordinates and transformed with mesh.
            for row in valid_rows:
                marker = trimesh.creation.icosphere(subdivisions=1, radius=radius)
                marker.vertices = transform_points(marker.vertices + row['normalized'], scale)
                marker.visual.vertex_colors = [255, 50, 50, 255]
                marked_meshes.append(marker)
            trimesh.util.concatenate(marked_meshes).export(directory / 'mesh_with_landmarks_world.ply')
            if reproject:
                result['reprojection'] = self.reproject_landmarks(result['landmarks'], directory)
        (directory / 'landmarks.json').write_text(json.dumps(result, indent=2))
        return result

    def reproject_landmarks(self, landmarks, directory):
        selected = np.asarray([row.get('valid', True) for row in landmarks], dtype=bool)
        vertices = np.asarray([row['normalized'] if row.get('valid', True) else [0., 0., 0.] for row in landmarks])
        homogeneous = np.c_[vertices, np.ones(len(vertices))]
        rows, errors = [], []
        (directory / 'reprojections').mkdir(exist_ok=True)
        per_class = [[] for _ in landmarks]
        reviewed_class = [[] for _ in landmarks]
        for i, path in enumerate(self.dataset.images_lis):
            projection = self.dataset.world_mats_np[i] @ self.dataset.scale_mats_np[i]
            projected = homogeneous @ projection[:3].T
            positive_depth = (projected[:, 2] > 1e-8) & selected
            pixels = projected[:, :2] / np.where(positive_depth[:, None], projected[:, 2:3], np.nan)
            # Keep the historical detector-label evaluation population. Visibility
            # filtering changes training supervision, not which errors are hidden.
            source_valid = getattr(self.annotations, 'source_valid', self.annotations.valid)[i]
            valid = source_valid & positive_depth
            reviewed = self.annotations.valid[i] & positive_depth
            distances = np.linalg.norm(pixels - self.annotations.xy[i], axis=-1)
            photo = cv2.imread(path)
            for k in range(len(landmarks)):
                if valid[k]:
                    errors.append(float(distances[k])); per_class[k].append(float(distances[k]))
                    gt = tuple(np.rint(self.annotations.xy[i, k]).astype(int))
                    cv2.circle(photo, gt, 8, (0, 255, 0), 2)
                if reviewed[k]:
                    reviewed_class[k].append(float(distances[k]))
                if positive_depth[k] and np.isfinite(pixels[k]).all() and (pixels[k] >= 0).all() and (pixels[k] < [self.dataset.W, self.dataset.H]).all():
                    p = tuple(np.rint(pixels[k]).astype(int)); cv2.circle(photo, p, 8, (0, 0, 255), 2)
                rows.append(dict(image=Path(path).name, id=self.landmark_ids[k],
                                 predicted_xy=pixels[k].tolist() if positive_depth[k] else None,
                                 valid=bool(valid[k]), error_pixels=float(distances[k]) if valid[k] else None))
                rows[-1]['training_supervision_valid'] = bool(reviewed[k])
            cv2.imwrite(str(directory / 'reprojections' / Path(path).name), photo)
        return dict(unit='native_image_pixels', valid_observations=len(errors),
                    selected_landmarks=int(selected.sum()), total_landmarks=len(landmarks),
                    selection_coverage=float(selected.mean()) if len(selected) else 0.,
                    coverage_caveat='Errors exclude unresolved landmarks; assess error together with coverage.',
                    mean_error=float(np.mean(errors)) if errors else None,
                    median_error=float(np.median(errors)) if errors else None,
                    per_landmark=[dict(id=self.landmark_ids[k], count=len(values), mean_error=float(np.mean(values)) if values else None) for k, values in enumerate(per_class)],
                    visibility_reviewed_subset=dict(
                        count=sum(len(values) for values in reviewed_class),
                        mean_error=float(np.mean([v for values in reviewed_class for v in values])) if any(reviewed_class) else None,
                        per_landmark=[dict(id=self.landmark_ids[k],count=len(values),mean_error=float(np.mean(values)) if values else None) for k,values in enumerate(reviewed_class)]),
                    observations=rows, caveat='Detected 2D labels are pseudo-ground-truth; visibility/occlusion only as annotated.')
