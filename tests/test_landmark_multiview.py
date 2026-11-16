"""Analytical camera/heatmap tests; no training or detector labels used."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh
from models.landmark_multiview import MeshRayQueries, project_vertices, sample_heatmap, visible_vertices, select_multiview_landmarks
from models.landmark_validation import LandmarkValidationMixin


class MultiviewTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=2)
        vertices = np.asarray(self.mesh.vertices)
        self.target = int(np.argmin(np.linalg.norm(vertices - [0, 0, -1], axis=1)))
        self.hidden = int(np.argmin(np.linalg.norm(vertices - [0, 0, 1], axis=1)))
        self.centers = np.array([[-.3, 0, -3], [0, 0, -3], [.3, 0, -3]])
        K = np.array([[60, 0, 50], [0, 60, 50], [0, 0, 1.]])
        self.projections = [K @ np.c_[np.eye(3), -c] for c in self.centers]
        grid = np.stack(np.meshgrid(np.arange(101), np.arange(101)), -1)
        self.maps = []
        for projection in self.projections:
            xy, _ = project_vertices(vertices[self.target:self.target+1], projection)
            self.maps.append((.9 * np.exp(-((grid - xy[0])**2).sum(-1) / (2*5**2)))[..., None])
        self.q = np.full((len(vertices), 1), .01)
        self.q[self.target] = .8
        self.q[self.hidden] = .99999
        self.options = dict(min_views=3, min_response=.1, min_relative_response=.5,
                            field_candidates=2, proposal_neighbors=8, proposal_radius=.3,
                            visibility_tolerance=.02, min_peak_contrast=2.)

    def select(self, maps=None, **options):
        return select_multiview_landmarks(self.mesh, self.q, self.maps if maps is None else maps,
                                          self.projections, self.centers, 101, 101, 5.,
                                          dict(self.options, **options))

    def test_bilinear_uses_native_endpoint_coordinates(self):
        heatmap = np.array([[0., .4], [.6, 1.]])
        result, inside = sample_heatmap(heatmap, np.array([[0,0], [100,200], [50,100], [101,0]]), 101, 201)
        np.testing.assert_allclose(result, [0, 1, .5, 0])
        np.testing.assert_array_equal(inside, [True, True, True, False])

    def test_occlusion_rejects_hidden_raw_maximum(self):
        indices = np.array([self.target, self.hidden])
        vis = visible_vertices(self.mesh, np.asarray(self.mesh.vertices)[indices], self.centers[1], .02)
        np.testing.assert_array_equal(vis, [True, False])
        result = self.select()[0]
        self.assertTrue(result['valid'])
        self.assertEqual(result['vertex_index'], self.target)
        self.assertEqual(len(result['supporting_views']), 3)
        self.assertNotEqual(result['vertex_index'], int(self.q.argmax()))

    def test_accelerated_first_hits_match_trimesh_fallback(self):
        fast=MeshRayQueries(self.mesh,'auto')
        slow=MeshRayQueries(self.mesh,'trimesh')
        origins=np.array([[0.,0.,-3.],[.2,.1,-3.],[4.,4.,-3.]])
        directions=np.array([[.01,.02,1.],[.02,.03,1.],[0.,0.,1.]])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        a,ai,_=fast.intersects_location(origins,directions)
        b,bi,_=slow.intersects_location(origins,directions)
        np.testing.assert_array_equal(np.sort(ai),np.sort(bi))
        np.testing.assert_allclose(a[np.argsort(ai)],b[np.argsort(bi)],atol=1e-5)

    def test_weak_or_diffuse_maps_abstain_without_raw_fallback(self):
        for maps in [[np.zeros_like(m) for m in self.maps], [np.full_like(m,.8) for m in self.maps]]:
            result = self.select(maps)[0]
            self.assertFalse(result['valid'])
            self.assertNotIn('vertex_index', result)

    def test_single_view_peak_cannot_be_accepted(self):
        maps = [self.maps[0], np.zeros_like(self.maps[1]), np.zeros_like(self.maps[2])]
        self.assertFalse(self.select(maps)[0]['valid'])

    def test_coincident_cameras_do_not_supply_angular_agreement(self):
        result = select_multiview_landmarks(self.mesh,self.q,[self.maps[1]]*3,
                                           [self.projections[1]]*3,[self.centers[1]]*3,
                                           101,101,5.,self.options)
        self.assertFalse(result[0]['valid'])

    def test_two_classes_produce_independent_surface_selections(self):
        maps = [np.concatenate([m, np.zeros_like(m)], axis=2) for m in self.maps]
        result = select_multiview_landmarks(self.mesh, np.repeat(self.q,2,axis=1), maps,
                                          self.projections,self.centers,101,101,[5,5],self.options)
        self.assertTrue(result[0]['valid'])
        self.assertFalse(result[1]['valid'])

    def test_invalid_inputs_and_search_bounds(self):
        with self.assertRaises(ValueError): self.select(min_views=1)
        bad = [m.copy() for m in self.maps]; bad[0][0,0,0] = np.nan
        with self.assertRaises(ValueError): self.select(bad)
        result = select_multiview_landmarks(self.mesh,self.q,self.maps,self.projections,
                                           self.centers,101,101,5.,self.options,
                                           search_bounds=[[[10,10,10],[11,11,11]]])
        self.assertFalse(result[0]['valid'])

    def test_reprojection_reports_abstention_and_coverage(self):
        import cv2
        v = LandmarkValidationMixin()
        P = np.eye(4); P[:3] = self.projections[1]
        v.dataset = SimpleNamespace(images_lis=['fake.png'], world_mats_np=[P],
                                    scale_mats_np=[np.eye(4)], W=101, H=101)
        target = np.asarray(self.mesh.vertices)[self.target]
        xy, _ = project_vertices(target[None], self.projections[1])
        v.annotations = SimpleNamespace(valid=np.array([[True,True]]),xy=np.array([[xy[0],xy[0]]]))
        v.landmark_ids = [1,33]
        rows = [dict(valid=True,normalized=target.tolist()),dict(valid=False,normalized=None)]
        with tempfile.TemporaryDirectory() as tmp, patch('models.landmark_validation.cv2.imread',return_value=np.zeros((101,101,3),dtype=np.uint8)), patch('models.landmark_validation.cv2.imwrite'):
            result=v.reproject_landmarks(rows,Path(tmp))
        self.assertEqual(result['selection_coverage'],.5)
        self.assertEqual(result['valid_observations'],1)
        self.assertEqual(result['per_landmark'][1]['count'],0)
        self.assertIsNone(result['per_landmark'][1]['mean_error'])

    def test_runner_selects_views_without_annotation_access(self):
        v=LandmarkValidationMixin()
        v.landmark_conf={'multiview':dict(self.options,max_views=3,resolution_level=1),'sigma':5.}
        poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,:3,3]=self.centers
        import torch
        world=[]
        for P in self.projections:
            mat=np.eye(4);mat[:3]=P;world.append(mat)
        v.dataset=SimpleNamespace(images_lis=['a','b','c'],W=101,H=101,
                                  world_mats_np=world,scale_mats_np=[np.eye(4)]*3,
                                  pose_all=torch.tensor(poses))
        v.landmark_ids,v.landmark_names=[33],['eye']
        v.render_landmark_response_map=lambda i,level:torch.from_numpy(self.maps[i])
        # Deliberately no annotations attribute: selection must not need one.
        with tempfile.TemporaryDirectory() as tmp:
            result,metadata=v.select_multiview_surface_landmarks(np.asarray(self.mesh.vertices),np.asarray(self.mesh.faces),self.q,Path(tmp))
        self.assertTrue(result[0]['valid'])
        self.assertFalse(metadata['uses_annotation_coordinates'])
        self.assertEqual(metadata['view_indices'],[0,1,2])

    def test_export_preserves_world_transform_and_explicit_abstention(self):
        import torch,json
        v=LandmarkValidationMixin()
        v.landmark_conf={'extraction_method':'multiview'}
        v.landmark_head=torch.nn.Linear(256,3)
        v.sdf_network=torch.nn.Linear(3,257)
        scale=np.eye(4);scale[:3,:3]*=2;scale[:3,3]=[10,20,30]
        v.dataset=SimpleNamespace(object_bbox_min=[-1,-1,-1],object_bbox_max=[1,1,1],scale_mats_np=[scale])
        v.renderer=SimpleNamespace(extract_geometry=lambda *a: (np.asarray(self.mesh.vertices),np.asarray(self.mesh.faces)))
        v.landmark_ids,v.landmark_names=[1,33],['nose','eye']
        v.device,v.iter_step,v.training_strategy=torch.device('cpu'),30000,'staged'
        selections=[dict(valid=True,vertex_index=self.target,confidence=.8),dict(valid=False,reason='Insufficient support')]
        v.select_multiview_surface_landmarks=lambda *a:(selections,dict(method='multiview'))
        with tempfile.TemporaryDirectory() as tmp:
            v.base_exp_dir=tmp
            result=v.export_surface_landmarks(output_dir=tmp,reproject=False)
            np.testing.assert_allclose(result['landmarks'][0]['world'],np.asarray(self.mesh.vertices)[self.target]*2+[10,20,30])
            self.assertIsNone(result['landmarks'][1]['normalized'])
            cloud=trimesh.load(Path(tmp)/'landmarks_world.ply')
            self.assertEqual(len(cloud.vertices),1)
            # All landmarks unresolved must still produce a valid empty PLY.
            v.select_multiview_surface_landmarks=lambda *a:([dict(valid=False,reason='weak')]*2,dict(method='multiview'))
            result=v.export_surface_landmarks(output_dir=tmp,reproject=False)
            self.assertTrue((Path(tmp)/'landmarks_world.ply').exists())
            self.assertTrue(all(not row['valid'] for row in result['landmarks']))
            # Legacy configs keep raw argmax and never render extra cameras.
            v.landmark_conf={}
            v.select_multiview_surface_landmarks=lambda *a: self.fail('Legacy export called multiview')
            result=v.export_surface_landmarks(output_dir=tmp,reproject=False)
            self.assertEqual(result['selection']['method'],'raw_argmax')
            self.assertEqual(len(result['landmarks']),2)
            self.assertTrue(all(row['normalized'] is not None for row in result['landmarks']))


if __name__ == '__main__': unittest.main()
