"""Checks for the staged-only, ten-landmark sigma-20 experiment; no GPU run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pyhocon import ConfigFactory
from models.dataset import Dataset, load_K_Rt_from_P
from models.landmarks import LandmarkAnnotations, heatmap_targets, heatmap_loss
from models.landmark_validation import LandmarkValidationMixin
from test_landmarks import renderer, render_scene


class StagedTenTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)
        torch.manual_seed(8)
        self.conf = ConfigFactory.parse_file('confs/staged_10_sigma20.conf')
        c = self.conf['landmarks']
        self.paths = sorted(Path('public_data/122_1_masked/image').glob('*.png'))
        self.labels = LandmarkAnnotations(
            c['annotation_path'].replace('CASE_NAME', '122_1_masked'),
            self.paths, 1728, 2592, c['ids'], c['names'])

    def test_schedule_labels_and_isolated_output(self):
        c = self.conf['landmarks']
        self.assertEqual(c['training_mode'], 'staged')
        self.assertEqual(c['num_landmarks'], 10)
        self.assertEqual(c['ids'][:3], [1, 33, 291])
        self.assertEqual(len(set(c['ids'])), 10)
        self.assertEqual(c['ids'], [1, 33, 291, 263, 133, 362, 61, 10, 168, 152])
        self.assertEqual(c['extraction_method'], 'multiview')
        self.assertEqual(c['multiview.min_views'], 3)
        self.assertEqual(c['sigma'], 20.)
        self.assertEqual(c['validation_resolution_level'], 4)
        self.assertEqual((c['geometry_warmup_iters'], c['landmark_warmup_iters'], c['joint_iters']), (10000, 5000, 15000))
        checkpoint = self.conf['train.init_checkpoint'].replace('CASE_NAME', '122_1_masked')
        self.assertTrue(Path(checkpoint).exists())
        self.assertIn('/staged_10_sigma20/', self.conf['general.base_exp_dir_geo'])
        np.testing.assert_array_equal(self.labels.valid.sum(0), np.full(10, 35))

    def test_sigma_is_measured_in_native_pixels(self):
        pixels = torch.tensor([[100., 100.], [120., 100.]])
        target = heatmap_targets(pixels, pixels[:1], torch.tensor([True]), 20.)
        torch.testing.assert_close(target[:, 0], torch.tensor([1., np.exp(-.5)], dtype=torch.float32))

    def test_ten_landmark_loss_reaches_head_and_sdf(self):
        r = renderer(k=10)
        out = render_scene(r)
        self.assertEqual(tuple(out['landmark_response'].shape), (3, 10))
        target = torch.zeros_like(out['landmark_response']); target[0] = 1.
        loss, _ = heatmap_loss(out['landmark_response'], target, torch.ones(10, dtype=torch.bool))
        for network in [r.landmark_head, r.sdf_network]:
            gradients = torch.autograd.grad(loss, list(network.parameters()), retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_validation_generates_432_by_648_maps(self):
        # Exercise the actual ray grid, coordinate tracking, targets and image
        # writer with deterministic synthetic responses, not trained predictions.
        cams = np.load('public_data/122_1_masked/cameras_sphere.npz')
        intrinsics, pose = load_K_Rt_from_P(None, (cams['world_mat_1'] @ cams['scale_mat_1'])[:3])
        dataset = Dataset.__new__(Dataset)
        dataset.W, dataset.H = 1728, 2592
        dataset.images_lis = list(map(str, self.paths))
        dataset.intrinsics_all_inv = torch.tensor(np.linalg.inv(intrinsics)[None], dtype=torch.float32).repeat(48, 1, 1)
        dataset.pose_all = torch.tensor(pose[None]).repeat(48, 1, 1)
        class SyntheticRenderer:
            def render(self, origins, directions, near, far, **kwargs):
                return {'landmark_response': torch.zeros(len(origins), 10)}
        v = LandmarkValidationMixin()
        v.landmark_head = object()
        v.landmark_conf = self.conf['landmarks']
        v.dataset, v.annotations = dataset, self.labels
        v.device, v.batch_size, v.iter_step = torch.device('cpu'), 512, 10000
        v.landmark_names, v.landmark_ids = self.labels.names, self.labels.ids
        v.renderer = SyntheticRenderer()
        v.get_cos_anneal_ratio = lambda: 1.
        with tempfile.TemporaryDirectory() as tmp, patch('models.landmark_validation.cv2.imwrite') as write:
            v.base_exp_dir = tmp
            v.validate_landmarks(idx=1)
            images = {Path(call.args[0]).name: call.args[1].shape for call in write.call_args_list}
            self.assertEqual(images['0_prediction.png'], (648, 432, 3))
            self.assertEqual(images['9_target.png'], (648, 432, 3))
            self.assertEqual(images['9_comparison.png'], (648, 864, 3))
            self.assertEqual(len(images), 31)


if __name__ == '__main__':
    unittest.main()
