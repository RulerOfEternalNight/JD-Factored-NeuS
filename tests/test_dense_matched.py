"""478-landmark tests using the unchanged 3-landmark training implementation."""
from pathlib import Path
import json
import unittest
import numpy as np
import torch
from pyhocon import ConfigFactory
from models.landmarks import LandmarkHead, LandmarkAnnotations, heatmap_loss
from test_landmarks import renderer, render_scene


class DenseMatchedTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_478_head_shape_and_softmax(self):
        head=LandmarkHead(478)
        probabilities=head(torch.randn(2,16,256)).softmax(-1)
        self.assertEqual(tuple(probabilities.shape),(2,16,479))
        torch.testing.assert_close(probabilities.sum(-1),torch.ones(2,16))

    def test_478_rendered_heatmap_loss_reaches_sdf(self):
        r=renderer(k=478)
        out=render_scene(r)
        prediction=out['landmark_response']
        self.assertEqual(tuple(prediction.shape),(3,478))
        self.assertTrue(torch.isfinite(prediction).all())
        target=torch.zeros_like(prediction);target[0,:]=1.
        loss,_=heatmap_loss(prediction,target,torch.ones(478,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        groups=[list(r.landmark_head.parameters()),list(r.sdf_network.parameters())]
        for parameters in groups:
            gradients=torch.autograd.grad(loss,parameters,retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)

    def test_full_resolution_dense_annotation_alignment(self):
        ids=list(range(478));paths=sorted(Path('public_data/122_1_masked/image').glob('*.png'))
        if not paths:self.skipTest('Original source photos are private; restore the dataset to run alignment checks')
        labels=LandmarkAnnotations('annotations/122_1_masked/mediapipe_478_v3.json',paths,1728,2592,ids)
        self.assertEqual(labels.xy.shape,(48,478,2))
        np.testing.assert_array_equal(labels.valid.sum(0),np.full(478,35))
        small=LandmarkAnnotations('annotations/122_1_masked/mediapipe_reliable_v2.json',paths,1728,2592,[1,33,291])
        for k,landmark_id in enumerate([1,33,291]):
            np.testing.assert_array_equal(labels.valid[:,landmark_id],small.valid[:,k])
            np.testing.assert_array_equal(labels.xy[:,landmark_id],small.xy[:,k])

    def test_config_diff_is_only_landmark_definitions_and_output(self):
        small=ConfigFactory.parse_file('confs/joint_from_zero.conf').as_plain_ordered_dict()
        dense=ConfigFactory.parse_file('confs/joint_from_zero_478.conf').as_plain_ordered_dict()
        allowed={('general','base_exp_dir_geo'),('landmarks','num_landmarks'),('landmarks','ids'),('landmarks','names'),('landmarks','annotation_path')}
        def compare(a,b,prefix=()):
            self.assertEqual(set(a),set(b))
            for key in a:
                path=prefix+(key,)
                if path in allowed:continue
                if isinstance(a[key],dict):compare(a[key],b[key],path)
                else:self.assertEqual(a[key],b[key],str(path))
        compare(small,dense)
        self.assertEqual(dense['train']['batch_size'],512)
        self.assertEqual(dense['train']['end_iter'],30000)
        self.assertNotIn('init_checkpoint',dense['train'])

if __name__=='__main__':unittest.main()
