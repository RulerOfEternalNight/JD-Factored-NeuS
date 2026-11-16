"""Behavioral tests for precision, visibility, domain and unknown-label handling."""
import copy,json,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
import trimesh
from pyhocon import ConfigFactory
from models.landmark_corrections import nonlandmark_surface_loss,known_negative_rays,HardNegativeCache
from models.landmark_multiview import select_multiview_landmarks,project_vertices
from models.landmark_visibility import review_visibility
from models.landmarks import LandmarkAnnotations,sample_pixels
from exp_runner import Runner
from test_landmarks import renderer,render_scene


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_tensor_type('torch.FloatTensor');torch.set_num_threads(2)
        torch.manual_seed(17);np.random.seed(17)

    def test_saturated_nonlandmark_gradient_is_strong_and_finite(self):
        z=torch.tensor([[[-100.,100.],[-100.,100.]]],requires_grad=True)
        w=torch.tensor([[.8,.1]],requires_grad=True)
        old=z.softmax(-1)[...,1:].sum(1).mean()
        old_gradient=torch.autograd.grad(old,z,retain_graph=True)[0]
        loss,used=nonlandmark_surface_loss(z.log_softmax(-1),w,torch.tensor([True]))
        gradient=torch.autograd.grad(loss,z,retain_graph=True)[0]
        self.assertTrue(torch.isfinite(loss));self.assertTrue(used[0])
        self.assertLess(float(old_gradient[...,0].abs().sum()),1e-10)
        self.assertGreater(float(gradient[...,0].abs().sum()),.99)
        self.assertIsNone(torch.autograd.grad(loss,w,allow_unused=True)[0])
        self.assertLess(float(gradient[...,0].sum()),0.)

    def test_empty_or_low_opacity_negatives_do_not_create_nan(self):
        logits=torch.randn(3,4,3,requires_grad=True)
        loss,used=nonlandmark_surface_loss(logits.log_softmax(-1),torch.zeros(3,4),torch.ones(3,dtype=torch.bool))
        self.assertEqual(float(loss),0.);self.assertFalse(used.any());loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unit_domain_excludes_missed_rays_without_changing_rgb(self):
        r=renderer();legacy=render_scene(r)
        r.landmark_domain='unit_sphere';r.return_landmark_log_probs=True
        fixed=render_scene(r)
        torch.testing.assert_close(fixed['color_fine'],legacy['color_fine'],rtol=0,atol=0)
        torch.testing.assert_close(fixed['weights'],legacy['weights'],rtol=0,atol=0)
        self.assertEqual(fixed['landmark_log_probs'].shape,(3,16,4))
        torch.testing.assert_close(fixed['landmark_response'][2],torch.zeros(3))
        torch.testing.assert_close(fixed['landmark_weights'],legacy['landmark_weights']*fixed['inside_sphere'])

    def test_corrected_loss_reaches_shared_features_and_preserves_main_weight_path(self):
        r=renderer();r.landmark_domain='unit_sphere';r.return_landmark_log_probs=True
        out=render_scene(r)
        aux,used=nonlandmark_surface_loss(out['landmark_log_probs'],out['landmark_weights'],torch.ones(3,dtype=torch.bool),min_opacity=.01)
        self.assertTrue(used.any())
        main=out['landmark_response'].square().sum()
        gradients=torch.autograd.grad(main+.1*aux,list(r.sdf_network.parameters()),retain_graph=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)
        weight_gradient=torch.autograd.grad(main,out['landmark_weights'],retain_graph=True)[0]
        self.assertGreater(float(weight_gradient.abs().sum()),0.)
        head_gradient=torch.autograd.grad(aux,list(r.landmark_head.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in head_gradient),0.)

    def test_auxiliary_training_keeps_frozen_geometry_unchanged(self):
        r=renderer();r.landmark_domain='unit_sphere';r.return_landmark_log_probs=True
        before=copy.deepcopy(r.sdf_network.state_dict())
        for p in r.sdf_network.parameters():p.requires_grad_(False)
        opt=torch.optim.SGD(r.landmark_head.parameters(),lr=.01)
        out=render_scene(r);loss,_=nonlandmark_surface_loss(out['landmark_log_probs'],out['landmark_weights'],torch.ones(3,dtype=torch.bool),min_opacity=.01)
        opt.zero_grad();loss.backward();opt.step()
        for name,value in r.sdf_network.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in r.sdf_network.parameters()))

    def test_missing_and_excluded_labels_are_not_silent_negatives(self):
        xy=torch.tensor([[10.,10.],[80.,80.]])
        pixels=torch.tensor([[10.,10.],[80.,80.],[50.,50.]])
        foreground=torch.tensor([True,True,True])
        self.assertFalse(known_negative_rays(pixels,xy,torch.tensor([True,False]),2.,foreground).any())
        # Visibility may exclude class two from heatmap loss, but its original
        # location remains protected in the known-negative test.
        mask=known_negative_rays(pixels,xy,torch.tensor([True,True]),2.,foreground)
        torch.testing.assert_close(mask,torch.tensor([False,False,True]))
        self.assertFalse(known_negative_rays(pixels,xy,torch.ones(2,dtype=torch.bool),2.,torch.zeros(3,dtype=torch.bool)).any())

    def test_hard_cache_and_sampling_protect_positive_and_unknown_regions(self):
        cache=HardNegativeCache(2)
        cache.update(0,np.array([[90,90],[10,10],[60,60]]),np.array([.9,1.,.2]),np.array([True,False,True]))
        self.assertEqual(len(cache.get(0)),2)
        clone=HardNegativeCache(2);clone.load_state_dict(json.loads(json.dumps(cache.state_dict())))
        np.testing.assert_array_equal(cache.get(0),clone.get(0))
        mask=np.zeros((100,100));mask[90,90]=1
        pixels=sample_pixels(100,100,16,np.array([[10.,10.]]),np.array([True]),2.,center_fraction=.25,negative_fraction=.5,
                             foreground_mask=mask,negative_pool=np.array([[90,90,.9],[10,10,1.]]),hard_fraction=1.)
        self.assertGreaterEqual(int((pixels==[90,90]).all(1).sum()),8)
        # No eligible foreground should fall back promptly, not hang.
        p=sample_pixels(20,20,16,np.array([[10.,10.]]),np.array([True]),2.,foreground_mask=np.zeros((20,20)))
        self.assertEqual(p.shape,(16,2))

    def test_visible_mean_does_not_reward_extra_visibility(self):
        mesh=trimesh.creation.icosphere(subdivisions=2);v=np.asarray(mesh.vertices)
        good=int(np.argmin(np.linalg.norm(v-[0,0,-1],axis=1)))
        bad=int(np.argmin(np.linalg.norm(v-[.7,0,-.7],axis=1)))
        centers=np.array([[-.3,0,-3],[0,0,-3],[.3,0,-3.]])
        K=np.array([[60,0,50],[0,60,50],[0,0,1.]])
        Ps=[K@np.c_[np.eye(3),-c] for c in centers];grid=np.stack(np.meshgrid(np.arange(101),np.arange(101)),-1)
        maps=[]
        for P in Ps:
            xy,_=project_vertices(v[[good,bad]],P)
            values=[level*np.exp(-((grid-point)**2).sum(-1)/2) for level,point in zip([.9,.8],xy)]
            maps.append(np.maximum(*values)[...,None])
        q=np.zeros((len(v),1));q[[good,bad]]=1.
        def visible(mesh,vertices,center,*args):
            mask=np.ones(len(vertices),bool)
            if np.allclose(center,centers[1]):mask[np.linalg.norm(vertices-v[good],axis=1)<1e-7]=False
            return mask
        options=dict(min_views=2,field_candidates=2,proposal_neighbors=1,proposal_radius=1e-6,peaks_per_view=1)
        with patch('models.landmark_multiview.visible_vertices',side_effect=visible):
            legacy=select_multiview_landmarks(mesh,q,maps,Ps,centers,101,101,1.,options)[0]
            fixed=select_multiview_landmarks(mesh,q,maps,Ps,centers,101,101,1.,dict(options,score_mode='visible_mean'))[0]
        self.assertEqual(legacy['vertex_index'],bad)
        self.assertEqual(fixed['vertex_index'],good)

    def test_visibility_flags_occlusion_and_mask_conflict_without_editing_coordinates(self):
        mesh=trimesh.creation.icosphere(subdivisions=2,radius=.5)
        centers=np.array([[-.3,0,-3],[0,0,-3],[.3,0,-3.]])
        K=np.array([[60,0,50],[0,60,50],[0,0,1.]])
        Ps=[K@np.c_[np.eye(3),-c] for c in centers]
        points=np.array([[0,0,-.5],[0,0,.5]])
        xy=np.array([project_vertices(points,P)[0] for P in Ps]);labels=SimpleNamespace(source_valid=np.ones((3,2),bool),xy=xy.copy(),ids=[1,33],names=['front','back'],image_names=['a','b','c'])
        masks=[np.ones((101,101),bool) for _ in range(3)];x,y=np.rint(xy[1,0]).astype(int);masks[1][y,x]=False
        review=review_visibility(labels,Ps,centers,mesh,masks=masks)
        valid=np.array(review['supervision_valid'])
        self.assertEqual(int(valid[:,0].sum()),2);self.assertEqual(int(valid[:,1].sum()),0)
        self.assertIn('mesh_inferred_occlusion',review['reasons'][0][1])
        self.assertIn('foreground_mask_conflict',review['reasons'][1][0])
        np.testing.assert_array_equal(labels.xy,xy)

    def test_real_sidecar_and_corrected_configuration_are_consistent(self):
        c=ConfigFactory.parse_file('confs/staged_10_corrected.conf');lc=c['landmarks']
        path=Path(lc['visibility.cache_path'].replace('CASE_NAME','122_1_masked'))
        if not path.exists():self.skipTest('Prepare real visibility sidecar first')
        paths=sorted(Path('public_data/122_1_masked/image').glob('*.png'))
        if not paths:self.skipTest('Original source photos are private; restore the dataset for sidecar alignment checks')
        labels=LandmarkAnnotations(lc['annotation_path'].replace('CASE_NAME','122_1_masked'),paths,1728,2592,lc['ids'],lc['names']);original=labels.xy.copy();review=labels.apply_visibility_filter(path)
        self.assertTrue((labels.valid.sum(0)>0).all());self.assertLess(int(labels.valid.sum()),int(labels.source_valid.sum()))
        np.testing.assert_array_equal(labels.xy,original)
        np.testing.assert_array_equal(labels.source_valid.sum(0),np.full(10,35))
        self.assertEqual(lc['multiview.score_mode'],'visible_mean');self.assertEqual(lc['render_domain'],'unit_sphere')
        self.assertIn('staged_10_corrected',c['general.base_exp_dir_geo'])
        with tempfile.TemporaryDirectory() as tmp:
            initial_hash=labels.visibility_hash
            regenerated=dict(review,created_utc='later')
            same=Path(tmp)/'same.json';same.write_text(json.dumps(regenerated))
            labels.apply_visibility_filter(same)
            self.assertEqual(labels.visibility_hash,initial_hash)
            changed=dict(review,annotation_sha256='changed');bad=Path(tmp)/'bad.json';bad.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):labels.apply_visibility_filter(bad)

    def test_corrected_resume_restores_cache_and_rejects_objective_changes(self):
        checkpoint=Path('verification/unified_smoke/baseline/checkpoints/ckpt_000002.pth').resolve()
        if not checkpoint.exists():self.skipTest('Synthetic upstream checkpoint fixture is unavailable')
        root=Path.cwd()
        c=ConfigFactory.parse_file('verification/unified_smoke/staged.conf').as_plain_ordered_dict()
        c['dataset']['data_dir']=str(root/'verification/unified_smoke/data')
        lc=c['landmarks'];lc.update(annotation_path=str(root/'verification/unified_smoke/annotations.json'),sigma=1.5,visualization_freq=0,
                                   render_domain='unit_sphere',nonlandmark=dict(enabled=True,weight=.1,minimum_opacity=.01,maximum_target=.01,detach_visibility=True),
                                   hard_negatives=dict(enabled=True,capacity=8,sample_fraction=.75))
        c['train']['init_checkpoint']=str(checkpoint)
        with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.is_available',return_value=False):
            c['general']['base_exp_dir_geo']=str(Path(tmp)/'run');path=Path(tmp)/'config.json';path.write_text(json.dumps(c))
            runner=Runner(str(path),case='fixture');runner.train(max_steps=1)
            state=runner.hard_negative_cache.state_dict();signature=runner.correction_signature()
            resumed=Runner(str(path),case='fixture',is_continue=True)
            self.assertEqual(resumed.iter_step,3);self.assertEqual(resumed.correction_signature(),signature)
            self.assertEqual(resumed.hard_negative_cache.state_dict(),state)
            c['landmarks']['nonlandmark']['weight']=.2;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'corrections/supervision differ'):Runner(str(path),case='fixture',is_continue=True)

    def test_filtered_training_labels_remain_in_historical_evaluation(self):
        from models.landmark_validation import LandmarkValidationMixin
        v=LandmarkValidationMixin();P=np.eye(4);P[2,3]=3.
        v.dataset=SimpleNamespace(images_lis=['fake.png'],world_mats_np=[P],scale_mats_np=[np.eye(4)],W=101,H=101)
        v.annotations=SimpleNamespace(xy=np.array([[[10.,10.]]]),valid=np.array([[False]]),source_valid=np.array([[True]]))
        v.landmark_ids=[1]
        with tempfile.TemporaryDirectory() as tmp,patch('models.landmark_validation.cv2.imread',return_value=np.zeros((101,101,3),np.uint8)),patch('models.landmark_validation.cv2.imwrite'):
            result=v.reproject_landmarks([dict(valid=True,normalized=[0.,0.,0.])],Path(tmp))
        self.assertEqual(result['valid_observations'],1)
        self.assertGreater(result['mean_error'],0.)
        self.assertEqual(result['visibility_reviewed_subset']['count'],0)

if __name__=='__main__':unittest.main()
