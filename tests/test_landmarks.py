import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from models.fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, RefColor, NeRF
from models.landmarks import LandmarkHead, LandmarkAnnotations, aggregate_landmarks, heatmap_targets, heatmap_loss, sample_pixels, stage_at
from models.renderer import NeuSRenderer


def renderer(k=3, outside=0):
    sdf = SDFNetwork(3, 257, 256, 2, skip_in=(), multires=0, weight_norm=False)
    head = LandmarkHead(k, hidden_dims=[32])
    return NeuSRenderer(8, 8, outside, 2, 0,
                        sdf_network=sdf, landmark_head=head,
                        deviation_network=SingleVarianceNetwork(.3),
                        color_network=RenderingNetwork(256, 'idr', 9, 3, 32, 2, weight_norm=False),
                        refColor_network=RefColor(),
                        nerf=NeRF(D=2,W=32,d_in=4,multires=1,multires_view=0,use_viewdirs=True,skips=[]))


def render_scene(r):
    origins = torch.tensor([[0.,0.,-2.], [.1,.1,-2.], [3.,3.,-2.]])
    directions = torch.tensor([[0.,0.,1.]]).expand_as(origins)
    near, far = torch.ones(3,1), torch.ones(3,1)*3
    return r.render(origins, directions, near, far, cos_anneal_ratio=1)


class LandmarkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        np.random.seed(8)
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)

    def test_architecture_k1_k3(self):
        sdf = SDFNetwork(3,257,256,2,skip_in=(),weight_norm=False)
        result = sdf(torch.randn(14,3))
        self.assertEqual(result.shape,(14,257))
        for k in (1,3):
            p=LandmarkHead(k)(result[:,1:].reshape(2,7,256)).softmax(-1)
            self.assertEqual(p.shape,(2,7,k+1))
            torch.testing.assert_close(p.sum(-1),torch.ones(2,7))

    def test_analytical_aggregation_zero_opacity_and_alignment(self):
        p=torch.tensor([[[.2,.3,.5],[.4,.5,.1]]],requires_grad=True)
        w=torch.tensor([[.2,.4]],requires_grad=True)
        out=aggregate_landmarks(p,w)
        torch.testing.assert_close(out,torch.tensor([[.26,.14]]))
        out.sum().backward(); self.assertGreater(float(w.grad.abs().sum()),0)
        torch.testing.assert_close(aggregate_landmarks(p,torch.zeros_like(w)),torch.zeros(1,2))
        with self.assertRaises(ValueError): aggregate_landmarks(p,torch.ones(1,3))

    def test_real_renderer_alignment_background_and_rgb_unchanged(self):
        for outside in (0,4):
            r=renderer(outside=outside); out=render_scene(r)
            self.assertEqual(out['weights'].shape,(3,16+outside))
            self.assertEqual(out['landmark_weights'].shape,(3,16))
            self.assertEqual(out['landmark_response'].shape,(3,3))
            self.assertTrue(torch.isfinite(out['landmark_response']).all())
            expected=out['weights'][:,:16]
            if outside: expected=expected*out['inside_sphere']
            torch.testing.assert_close(out['landmark_weights'],expected)
            r.landmark_head=None; baseline=render_scene(r)
            torch.testing.assert_close(out['color_fine'],baseline['color_fine'],atol=0,rtol=0)
            if outside: torch.testing.assert_close(out['landmark_response'][2],torch.zeros(3))

    def test_landmark_only_and_rgb_gradients(self):
        r=renderer(); out=render_scene(r)
        loss=out['landmark_response'].square().sum()
        head_grad=torch.autograd.grad(loss,list(r.landmark_head.parameters()),retain_graph=True)
        sdf_grad=torch.autograd.grad(loss,list(r.sdf_network.parameters()),retain_graph=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in head_grad),0)
        self.assertGreater(sum(float(g.abs().sum()) for g in sdf_grad),0)
        rgb_sdf_grad=torch.autograd.grad(out['color_fine'].sum(),list(r.sdf_network.parameters()),retain_graph=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in rgb_sdf_grad),0)
        rgb_grad=torch.autograd.grad(out['color_fine'].sum(),list(r.color_network.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in rgb_grad),0)

    def test_geometry_gradient_through_weights_with_constant_probabilities(self):
        r=renderer(k=1)
        # A constant head removes every feature-to-landmark gradient path.
        for p in r.landmark_head.parameters(): torch.nn.init.zeros_(p)
        out=render_scene(r)
        gradient=torch.autograd.grad(out['landmark_response'].sum(),list(r.sdf_network.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradient),1e-8)

    def test_geometry_feature_path_with_weights_held_constant(self):
        from unittest.mock import patch
        r=renderer()
        with patch('models.renderer.aggregate_landmarks',lambda probabilities,weights: aggregate_landmarks(probabilities,weights.detach())):
            out=render_scene(r)
        gradient=torch.autograd.grad(out['landmark_response'].sum(),list(r.sdf_network.parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradient),1e-8)

    def test_frozen_sdf_head_update(self):
        r=renderer(); before={k:v.clone() for k,v in r.sdf_network.state_dict().items()}
        for p in r.sdf_network.parameters(): p.requires_grad_(False)
        optimizer=torch.optim.Adam(r.landmark_head.parameters(),lr=.01)
        head_before=[p.clone() for p in r.landmark_head.parameters()]
        out=render_scene(r); out['landmark_response'].sum().backward(); optimizer.step()
        for k,v in r.sdf_network.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in r.sdf_network.parameters()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(head_before,r.landmark_head.parameters())))

    def test_heatmap_targets_invalid_and_weighted_loss(self):
        pixels=torch.tensor([[5.,5.],[6.,5.]])
        xy=torch.tensor([[5.,5.],[0.,0.]])
        valid=torch.tensor([True,False])
        target=heatmap_targets(pixels,xy,valid,1.)
        self.assertAlmostEqual(float(target[0,0]),1.)
        self.assertAlmostEqual(float(target[1,0]),np.exp(-.5),places=6)
        pred=torch.zeros_like(target,requires_grad=True)
        loss,classes=heatmap_loss(pred,target,valid)
        loss.backward(); self.assertLess(float(pred.grad[0,0]),0)
        self.assertEqual(float(classes[1]),0)
        empty,_=heatmap_loss(pred,target,torch.zeros_like(valid));self.assertEqual(float(empty),0)

    def test_annotations_matching_transform_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json'
            data={'images':{'x.png':{'width':100,'height':80,'landmarks':[{'name':'nose','xy':[40,30]}]}}}
            path.write_text(json.dumps(data))
            a=LandmarkAnnotations(path,['x.png','missing.png'],100,80,['nose'])
            self.assertTrue(a.valid[0,0]); self.assertFalse(a.valid[1,0])
            with self.assertRaises(ValueError): LandmarkAnnotations(path,['x.png'],50,40,['nose'])
            row=data['images']['x.png'];row.update(pixel_transform=[[.5,0,-.25],[0,.5,-.25],[0,0,1]],training_width=50,training_height=40)
            path.write_text(json.dumps(data));a=LandmarkAnnotations(path,['x.png'],50,40,['nose'])
            np.testing.assert_allclose(a.xy[0,0],[19.75,14.75])

    def test_positive_balance_rejects_all_zero_on_sparse_targets(self):
        target=torch.zeros(512,1);target[0]=1
        valid=torch.tensor([True])
        zero,_=heatmap_loss(torch.zeros_like(target),target,valid)
        fitted,_=heatmap_loss(target,target,valid)
        self.assertGreater(float(zero),.8)
        self.assertEqual(float(fitted),0.)
        # Adding an invalid class must not change the normalized loss.
        padded,_=heatmap_loss(torch.zeros(512,2),torch.cat([target,target],1),torch.tensor([True,False]))
        torch.testing.assert_close(zero,padded)

    def test_normalized_annotation_visibility_and_camera_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.json'
            row=dict(width=100,height=80,camera_index=0,landmarks=[dict(id=1,xy=[.4,.5],valid=True,visibility=0)])
            data=dict(coordinate_system='normalized',images={'x.png':row});path.write_text(json.dumps(data))
            labels=LandmarkAnnotations(path,['x.png'],100,80,[1]);self.assertFalse(labels.valid[0,0])
            row['landmarks'][0]['visibility']=1;path.write_text(json.dumps(data))
            labels=LandmarkAnnotations(path,['x.png'],100,80,[1]);np.testing.assert_allclose(labels.xy[0,0],[40,40])
            row['camera_index']=1;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):LandmarkAnnotations(path,['x.png'],100,80,[1])

    def test_export_coordinate_transform(self):
        from models.landmark_validation import transform_points
        points=np.array([[.1,.2,.3],[-.5,0,.4]])
        transform=np.array([[0,-2,0,5],[2,0,0,8],[0,0,2,-1],[0,0,0,1.]])
        world=transform_points(points,transform)
        np.testing.assert_allclose(transform_points(world,np.linalg.inv(transform)),points)
        np.testing.assert_allclose(world[0],[4.6,8.2,-.4])

    def test_sampling_and_schedule(self):
        xy=np.array([[30.,40.]],dtype=np.float32)
        pixels=sample_pixels(100,100,512,xy,np.array([True]),2.,.5,.25)
        self.assertEqual(pixels.shape,(512,2));self.assertTrue(((pixels>=0)&(pixels<100)).all())
        self.assertGreater(((pixels-xy)**2).sum(-1).__lt__(36).sum(),200)
        for step,expected in [(0,'geometry'),(1,'geometry'),(2,'head'),(3,'head'),(4,'joint')]:
            self.assertEqual(stage_at('staged',step,2,2),expected)
        self.assertEqual(stage_at('joint_from_zero',0),'joint')
        self.assertEqual(stage_at('baseline',100),'geometry')

if __name__=='__main__': unittest.main()
