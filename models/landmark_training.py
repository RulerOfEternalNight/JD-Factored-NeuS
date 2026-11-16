"""Optional Stage 1 landmark training; upstream reconstruction losses stay in Runner."""
import logging
import random
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from models.landmarks import LandmarkAnnotations, LandmarkHead, heatmap_loss, sample_pixels, stage_at
from models.landmark_corrections import nonlandmark_surface_loss, known_negative_rays, HardNegativeCache


class LandmarkTrainingMixin:
    def setup_landmarks(self, reconstruction_parameters):
        self.landmark_conf = self.conf.get('landmarks', {})
        c = self.landmark_conf
        self.training_strategy = c.get('training_mode', 'baseline')
        self.landmarks_enabled = bool(c.get('enabled', False)) and self.training_strategy != 'baseline'
        if self.training_strategy not in ('baseline', 'joint_from_zero', 'staged'):
            raise ValueError('Invalid landmarks.training_mode')
        if self.training_strategy != 'baseline' and not self.landmarks_enabled:
            raise ValueError('Landmark training modes require landmarks.enabled=true')
        self.reconstruction_parameters = reconstruction_parameters
        self.landmark_head = None
        self.annotations = None
        self.current_stage = None
        self.image_perm = None
        self.initialization_source = None
        self.hard_negative_cache = None
        self.annotation_hash = None
        self.geometry_iters = int(c.get('geometry_warmup_iters', 10000))
        self.head_iters = int(c.get('landmark_warmup_iters', 5000))
        self.joint_iters = int(c.get('joint_iters', 15000))
        if self.training_strategy == 'staged':
            if min(self.geometry_iters, self.head_iters, self.joint_iters) < 0 or self.joint_iters < 1:
                raise ValueError('Invalid staged iteration budget')
            if self.end_iter != self.geometry_iters + self.head_iters + self.joint_iters:
                raise ValueError('train.end_iter must equal sum of staged iteration budgets')
        self.optimizer = torch.optim.Adam(reconstruction_parameters, lr=self.learning_rate)
        if self.landmarks_enabled:
            if self.type != 'dtu':
                raise ValueError('Landmark supervision currently supports the calibrated DTU-style loader')
            k = int(c['num_landmarks'])
            self.landmark_ids = list(c['ids'])
            self.landmark_names = list(c.get('names', [str(i) for i in self.landmark_ids]))
            if len(self.landmark_ids) != k:
                raise ValueError('num_landmarks must match ids')
            self.landmark_head = LandmarkHead(k, self.conf['model.sdf_network.d_out'] - 1,
                                              c.get('hidden_dims', [128, 64])).to(self.device)
            self.optimizer.add_param_group({'params': list(self.landmark_head.parameters()),
                                            'lr': self.learning_rate})
            self.annotations = LandmarkAnnotations(c['annotation_path'], self.dataset.images_lis,
                                                  self.dataset.W, self.dataset.H,
                                                  self.landmark_ids, self.landmark_names)
            self.annotation_hash = hashlib.sha256(Path(c['annotation_path']).read_bytes()).hexdigest()
            visibility = c.get('visibility', {})
            if visibility.get('enabled', False):
                from models.landmark_visibility import mask_fingerprint
                path = Path(visibility['cache_path'])
                if not path.exists():
                    raise ValueError('Prepare annotation visibility first: python scripts/prepare_landmark_visibility.py --conf ' + str(self.conf_path) + ' --case ' + str(self.case))
                reviewed = self.annotations.apply_visibility_filter(path)
                camera_path = Path(self.dataset.data_dir) / self.conf['dataset.render_cameras_name']
                if reviewed['cameras_sha256'] != hashlib.sha256(camera_path.read_bytes()).hexdigest():
                    raise ValueError('Visibility cache calibration changed; regenerate the cache')
                reference = Path(visibility['reference_mesh'])
                if reviewed['reference_mesh_sha256'] != hashlib.sha256(reference.read_bytes()).hexdigest():
                    raise ValueError('Visibility reference mesh changed; regenerate the cache')
                if reviewed.get('masks_sha256') != mask_fingerprint(self.dataset.data_dir, self.annotations.image_names):
                    raise ValueError('Visibility cache foreground masks changed; regenerate the cache')
                expected = dict(visibility_tolerance=float(visibility.get('tolerance', .02)),
                                maximum_reprojection_error=float(visibility.get('maximum_reprojection_error', 60.)),
                                maximum_surface_distance=float(visibility.get('maximum_surface_distance', .05)),
                                minimum_views=int(visibility.get('minimum_views', 3)),
                                robust_scale=float(visibility.get('robust_scale', np.median(c.get('sigma', 12.)))))
                if reviewed['parameters'] != expected or reviewed['reference_mesh_coordinates'] != visibility.get('mesh_coordinates', 'world'):
                    raise ValueError('Visibility review settings changed; regenerate the cache')
            hard = c.get('hard_negatives', {})
            if hard.get('enabled', False):
                self.hard_negative_cache = HardNegativeCache(int(hard.get('capacity', 1024)))
            aux = c.get('nonlandmark', {})
            if aux.get('enabled', False) and (float(aux.get('weight', .1)) <= 0 or
                                            not 0 < float(aux.get('minimum_opacity', .5)) <= 1 or
                                            not 0 < float(aux.get('maximum_target', .01)) < 1):
                raise ValueError('Invalid non-landmark classification settings')
            self.valid_images = np.flatnonzero(self.annotations.valid.any(-1))
            if not len(self.valid_images):
                raise ValueError('No valid landmark labels match this dataset/configuration')
            sigma = np.broadcast_to(np.asarray(c.get('sigma', 12.)), (k,))
            if (sigma <= 0).any() or float(c.get('loss_weight', .1)) <= 0 or not 0 < float(c.get('initial_weight_ratio', .1)) <= 1:
                raise ValueError('Sigma, landmark weight and initial weight ratio must be positive')
            center, negative = float(c.get('center_fraction', .1875)), float(c.get('negative_fraction', .0625))
            if min(center, negative) < 0 or center + negative > 1 or not 0 <= float(c.get('annotated_image_fraction', .5)) <= 1:
                raise ValueError('Invalid sampling proportions')
            if not np.isfinite(sigma).all():
                raise ValueError('Sigma must be finite')
            logging.info('Landmark labels: %s valid image/class pairs in %s views', self.annotations.valid.sum(), len(self.valid_images))

    def training_stage(self):
        return stage_at(self.training_strategy, self.iter_step, self.geometry_iters, self.head_iters)

    def set_training_stage(self):
        stage = self.training_stage()
        if stage != self.current_stage:
            if self.current_stage is not None:
                self.save_checkpoint()  # persist a boundary before changing trainability
            for p in self.reconstruction_parameters:
                p.requires_grad = stage != 'head'
                p.grad = None
            if self.landmark_head is not None:
                for p in self.landmark_head.parameters():
                    p.requires_grad = stage != 'geometry'
                    p.grad = None
            logging.info('Training stage %s at iteration %d', stage, self.iter_step)
            self.current_stage = stage
        return stage

    def landmark_weight(self):
        c = self.landmark_conf
        start = 0 if self.training_strategy == 'joint_from_zero' else self.geometry_iters + self.head_iters
        ramp = max(1, int(c.get('weight_ramp_iters', 1000)))
        ratio = float(c.get('initial_weight_ratio', .1))
        progress = np.clip((self.iter_step - start) / ramp, 0, 1)
        return float(c.get('loss_weight', .1)) * (ratio + (1 - ratio) * progress)

    def sample_training_data(self, image_index, active):
        if not active:
            return self.dataset.gen_random_rays_at(image_index, self.batch_size), None, int(image_index)
        c = self.landmark_conf
        image_index = int(image_index)
        # Start landmark optimization with actual valid labels. Later mixed view
        # selection retains unannotated views for RGB/geometry supervision.
        if self.iter_step == 0 or self.training_stage() == 'head' or np.random.rand() < float(c.get('annotated_image_fraction', .5)):
            image_index = int(np.random.choice(self.valid_images))
        extra = {}
        if self.hard_negative_cache is not None:
            extra = dict(foreground_mask=self.dataset.masks[image_index, :, :, 0].numpy(),
                         negative_pool=self.hard_negative_cache.get(image_index),
                         hard_fraction=float(c['hard_negatives'].get('sample_fraction', .75)),
                         negative_xy=self.annotations.xy[image_index],
                         negative_valid=self.annotations.source_valid[image_index])
        pixels = sample_pixels(self.dataset.W, self.dataset.H, self.batch_size,
                               self.annotations.xy[image_index], self.annotations.valid[image_index],
                               c.get('sigma', 12.), float(c.get('center_fraction', .1875)),
                               float(c.get('negative_fraction', .0625)), float(c.get('center_sigma', 1.)), **extra)
        return self.dataset.rays_at_pixels(image_index, pixels), torch.as_tensor(pixels, device=self.device, dtype=torch.float32), image_index

    def landmark_objective(self, prediction, image_index, pixels, render_out=None, foreground=None):
        c = self.landmark_conf
        target, valid = self.annotations.targets(image_index, pixels, c.get('sigma', 12.))
        loss, per_class = heatmap_loss(prediction, target, valid,
                                      float(c.get('positive_weight', 10.)), float(c.get('negative_weight', 1.)),
                                      bool(c.get('balance_positive_negative', True)))
        self.writer.add_scalar('Loss/landmark_heatmap', loss, self.iter_step)
        aux = c.get('nonlandmark', {})
        if aux.get('enabled', False) or self.hard_negative_cache is not None:
            if render_out is None or foreground is None:
                raise ValueError('Corrected landmark objective needs rendered samples and foreground masks')
            source_xy = torch.as_tensor(self.annotations.xy[image_index], device=self.device)
            source_valid = torch.as_tensor(self.annotations.source_valid[image_index], device=self.device)
            negative = known_negative_rays(pixels, source_xy, source_valid, c.get('sigma', 12.),
                                           foreground > .5, float(aux.get('maximum_target', .01)))
            self.writer.add_scalar('Landmarks/known_negative_rays', negative.sum(), self.iter_step)
            if aux.get('enabled', False):
                if render_out.get('landmark_log_probs') is None:
                    raise ValueError('Stable non-landmark loss needs renderer log_softmax outputs')
                classification, usable = nonlandmark_surface_loss(
                    render_out['landmark_log_probs'], render_out['landmark_weights'], negative,
                    float(aux.get('minimum_opacity', .5)), bool(aux.get('detach_visibility', True)))
                loss = loss + float(aux.get('weight', .1)) * classification
                self.writer.add_scalar('Loss/nonlandmark_surface_ce', classification, self.iter_step)
                self.writer.add_scalar('Landmarks/classification_rays', usable.sum(), self.iter_step)
                if usable.any():
                    w = render_out['landmark_weights'].detach()
                    bg = (w * render_out['landmark_log_probs'][..., 0].detach().exp()).sum(-1) / w.sum(-1).clamp_min(1e-8)
                    self.writer.add_scalar('Landmarks/nonlandmark_probability_on_negatives', bg[usable].mean(), self.iter_step)
                    self.writer.add_scalar('Landmarks/false_response_on_negatives', prediction.detach()[usable].sum(-1).mean(), self.iter_step)
            if self.hard_negative_cache is not None:
                self.hard_negative_cache.update(image_index, pixels.detach().cpu().numpy(),
                                                prediction.detach().sum(-1).cpu().numpy(), negative.cpu().numpy())
                self.writer.add_scalar('Landmarks/hard_negative_cache_size', len(self.hard_negative_cache.get(image_index)), self.iter_step)
        self.writer.add_scalar('Landmarks/valid_classes', valid.sum(), self.iter_step)
        self.writer.add_scalar('Loss/landmark', loss, self.iter_step)
        self.writer.add_scalar('Landmarks/lambda', self.landmark_weight(), self.iter_step)
        for k, name in enumerate(self.landmark_names):
            if valid[k]:
                self.writer.add_scalar('Landmarks/loss/' + name, per_class[k], self.iter_step)
                self.writer.add_scalar('Landmarks/response_max/' + name, prediction[:, k].max(), self.iter_step)
        return loss

    def log_landmark_gradients(self, loss, render_out):
        frequency = int(self.landmark_conf.get('gradient_log_freq', 100))
        if frequency <= 0 or self.iter_step % frequency:
            return
        groups = {'head': list(self.landmark_head.parameters())}
        if self.training_stage() == 'joint':
            groups['sdf'] = list(self.sdf_network.parameters())
            groups['render_weights'] = [render_out['landmark_weights']]
        for name, parameters in groups.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            norm = sum(g.square().sum() for g in gradients if g is not None)
            if isinstance(norm, int):
                norm = torch.tensor(float(norm), device=self.device)
            self.writer.add_scalar('LandmarkGradients/' + name, norm.sqrt(), self.iter_step)

    def checkpoint_metadata(self):
        result = dict(training_strategy=self.training_strategy, stage=self.training_stage(),
                      schedule=dict(geometry_iters=self.geometry_iters, head_iters=self.head_iters,
                                    joint_iters=self.joint_iters, end_iter=self.end_iter),
                      configuration=self.conf.as_plain_ordered_dict(),
                      rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                      image_perm=self.image_perm, initialization_source=self.initialization_source)
        if self.landmark_head is not None:
            result.update(landmark_head=self.landmark_head.state_dict(), landmark_ids=self.landmark_ids,
                          landmark_names=self.landmark_names)
        signature = self.correction_signature()
        if signature is not None:
            result['correction_signature'] = signature
        if self.hard_negative_cache is not None:
            result['hard_negative_cache'] = self.hard_negative_cache.state_dict()
        return result

    def correction_signature(self):
        c = self.landmark_conf
        corrected = (c.get('render_domain', 'legacy') != 'legacy' or c.get('nonlandmark', {}).get('enabled', False)
                     or c.get('hard_negatives', {}).get('enabled', False) or c.get('visibility', {}).get('enabled', False))
        if not self.landmarks_enabled or not corrected:
            return None
        return json.loads(json.dumps(dict(render_domain=c.get('render_domain', 'legacy'),
                                         version=1, annotation_sha256=self.annotation_hash,
                                         nonlandmark=dict(c.get('nonlandmark', {})),
                                         hard_negatives=dict(c.get('hard_negatives', {})),
                                         visibility_hash=self.annotations.visibility_hash,
                                         positive_weight=c.get('positive_weight', 10.),
                                         negative_weight=c.get('negative_weight', 1.), sigma=c.get('sigma', 12.),
                                         heatmap_balance=c.get('balance_positive_negative', True),
                                         sampling={k: c.get(k, default) for k, default in
                                                   [('center_fraction', .1875), ('negative_fraction', .0625),
                                                    ('center_sigma', 1.), ('annotated_image_fraction', .5)]})))

    def restore_extended_checkpoint(self, checkpoint):
        if checkpoint.get('correction_signature') != self.correction_signature():
            raise ValueError('Landmark corrections/supervision differ; use geometry initialization for a new experiment, not resume')
        if 'landmark_head' in checkpoint and self.landmark_head is not None:
            if checkpoint.get('landmark_ids') != self.landmark_ids:
                raise ValueError('Checkpoint landmark IDs/order mismatch')
            self.landmark_head.load_state_dict(checkpoint['landmark_head'])
        if checkpoint.get('training_strategy', 'baseline') != self.training_strategy:
            raise ValueError('Strategy differs; use --init_checkpoint for geometry initialization')
        if checkpoint.get('schedule') and checkpoint['schedule'] != self.checkpoint_metadata()['schedule']:
            raise ValueError('Resume schedule mismatch; use original configuration')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.hard_negative_cache is not None:
            if 'hard_negative_cache' not in checkpoint:
                raise ValueError('Corrected checkpoint is missing its hard-negative cache')
            self.hard_negative_cache.load_state_dict(checkpoint['hard_negative_cache'])
        self.iter_step = checkpoint['iter_step']
        self.image_perm = checkpoint.get('image_perm')
        self.initialization_source = checkpoint.get('initialization_source')
        rng = checkpoint.get('rng')
        if rng:
            random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])

    def initialize_geometry(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_reconstruction_state(checkpoint)
        self.initialization_source = dict(path=str(path), iteration=int(checkpoint.get('iter_step', 0)))
        if self.training_strategy == 'staged' and checkpoint.get('iter_step') != self.geometry_iters:
            logging.warning('Imported pretraining has %s iterations; configured warmup budget is %s. Account for actual pretraining in comparisons.', checkpoint.get('iter_step'), self.geometry_iters)
        self.iter_step = self.geometry_iters if self.training_strategy == 'staged' else 0
        logging.info('Loaded reconstruction weights from %s; local iteration %d', path, self.iter_step)
