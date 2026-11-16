"""Short CUDA integration checks on a calibrated synthetic sphere fixture.

Synthetic landmarks are test fixtures only, never anatomical measurements.
Run from the repository root: python tests/smoke_training.py
"""
import json
import gc
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import trimesh
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from exp_runner import Runner

ROOT=Path('verification/unified_smoke')

def fixture():
    data=ROOT/'data'; (data/'image').mkdir(parents=True,exist_ok=True); (data/'mask').mkdir(exist_ok=True)
    cameras={}; annotations={'coordinate_system':'pixels','images':{}}
    points=np.array([[0.,0.,-.5],[-.15,.1,-.46],[.15,-.1,-.46]])
    for i,angle in enumerate([-.2,0.,.2]):
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        k=np.array([[60,0,31.5],[0,60,31.5],[0,0,1.]])
        p=k@np.column_stack([rotation,[0,0,3.]])
        world=np.eye(4);world[:3]=p;cameras[f'world_mat_{i}']=world;cameras[f'scale_mat_{i}']=np.eye(4)
        image=np.full((64,64,3),30,dtype=np.uint8);cv2.circle(image,(32,32),11,(130,150,180),-1)
        mask=np.zeros_like(image);cv2.circle(mask,(32,32),11,(255,255,255),-1)
        name=f'{i:03d}.png';cv2.imwrite(str(data/'image'/name),image);cv2.imwrite(str(data/'mask'/name),mask)
        proj=np.c_[points,np.ones(3)]@p.T;xy=proj[:,:2]/proj[:,2:]
        annotations['images'][name]={'width':64,'height':64,'landmarks':[dict(id=j,xy=q.tolist(),valid=True) for j,q in enumerate(xy)]}
    np.savez(data/'cameras_sphere.npz',**cameras)
    (ROOT/'annotations.json').write_text(json.dumps(annotations))


def config(strategy, name):
    path=ROOT/(name+'.conf')
    path.write_text(f'''include "../../confs/wmask.conf"
general.base_exp_dir_geo = ./{ROOT}/{name}
dataset.data_dir = ./{ROOT}/data
train {{end_iter=6, batch_size=32, warm_up_end=0, seed=17, save_freq=1, report_freq=1, val_freq=1000, val_mesh_freq=1000}}
model.neus_renderer {{n_samples=16,n_importance=16,up_sample_steps=2,perturb=1.0}}
landmarks {{
 enabled={'false' if strategy=='baseline' else 'true'}
 training_mode={strategy}
 num_landmarks=3
 ids=[0,1,2]
 names=[fixture_a,fixture_b,fixture_c]
 annotation_path=./{ROOT}/annotations.json
 sigma=2.0
 center_fraction=.5
 negative_fraction=.125
 loss_weight=.1
 initial_weight_ratio=.1
 geometry_warmup_iters=2
 landmark_warmup_iters=2
 joint_iters=2
 gradient_log_freq=1
 visualization_freq=3
 validation_resolution_level=8
 export_freq=0
}}
''')
    return str(path)


def checkpoint(name, step):
    return torch.load(ROOT/name/'checkpoints'/f'ckpt_{step:06d}.pth',map_location='cpu')


def equal_state(a,b):
    return all(torch.equal(a[k],b[k]) for k in a)


def run(strategy,name,steps=None,resume=False,init=None):
    path=config(strategy,name)
    torch.cuda.reset_peak_memory_stats()
    r=Runner(path,case='fixture',is_continue=resume,init_checkpoint=init)
    r.train(max_steps=steps)
    r.validate_image(idx=0,resolution_level=8)
    assert list((ROOT/name/'validations_fine').glob('*.png'))
    exported=r.export_surface_landmarks(resolution=24)
    export_path=ROOT/name/'exports'/f'{r.iter_step:08d}'
    mesh=trimesh.load(export_path/'mesh_world.ply',process=False)
    for landmark in exported['landmarks']:
        np.testing.assert_allclose(mesh.vertices[landmark['vertex_index']],landmark['world'],atol=1e-7)
        assert 0 <= landmark['confidence'] <= 1
    if exported['landmarks']:
        assert exported['reprojection']['valid_observations']>0
    result={'steps':r.iter_step,'peak_cuda_memory_MiB':round(torch.cuda.max_memory_allocated()/2**20,2)}
    del r; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA integration checks require a GPU')
    torch.set_num_threads(2);torch.set_default_tensor_type('torch.cuda.FloatTensor')
    fixture(); results={}
    for strategy in ('baseline','joint_from_zero','staged'):
        results[strategy]=run(strategy,strategy)
        ck=checkpoint(strategy,6)
        for name,state in ck.items():
            if name in ('nerf','sdf_network_fine','variance_network_fine','color_network_fine','refColor_network','landmark_head'):
                assert all(torch.isfinite(v).all() for v in state.values())
        if strategy!='baseline':
            events=EventAccumulator(str(ROOT/strategy/'logs')).Reload()
            assert events.Scalars('LandmarkGradients/head')[0].value>0
            assert events.Scalars('LandmarkGradients/sdf')[0].value>0
            assert events.Scalars('LandmarkGradients/render_weights')[0].value>0
            assert list((ROOT/strategy/'landmark_validation').rglob('*_comparison.png'))
            if strategy=='joint_from_zero':
                assert events.Scalars('Loss/landmark')[0].step==0
    a,b,c=checkpoint('staged',2),checkpoint('staged',4),checkpoint('staged',6)
    for name in ('nerf','sdf_network_fine','variance_network_fine','color_network_fine','refColor_network'):
        assert equal_state(a[name],b[name]),'Frozen reconstruction changed: '+name
    assert not equal_state(a['landmark_head'],b['landmark_head'])
    assert not equal_state(b['sdf_network_fine'],c['sdf_network_fine'])
    results['frozen_geometry_unchanged']=True
    results['joint_unfreezes_sdf']=True
    for strategy,pause_at in [('baseline',2),('joint_from_zero',2),('staged',3)]:
        name='resume_'+strategy
        run(strategy,name,steps=pause_at)
        run(strategy,name,resume=True)
        resumed=checkpoint(name,6);reference=checkpoint(strategy,6)
        for component in ('nerf','sdf_network_fine','variance_network_fine','color_network_fine','refColor_network','landmark_head'):
            if component in reference:
                assert equal_state(reference[component],resumed[component]),'Resume differs: '+strategy+'/'+component
        results[strategy]['resume_exact_parameters']=True
    # Upstream geometry checkpoint: no extension metadata or landmark group.
    baseline=checkpoint('baseline',2)
    original={k:baseline[k] for k in ('nerf','sdf_network_fine','variance_network_fine','color_network_fine','refColor_network','optimizer','iter_step')}
    upstream=ROOT/'upstream.pth';torch.save(original,upstream)
    results['upstream_initialization']=run('staged','initialized',steps=1,init=str(upstream))
    # Explicit error rather than silently loading mismatched optimizer groups.
    r=Runner(config('joint_from_zero','mismatch'),case='fixture')
    try:
        r.restore_extended_checkpoint(original)
    except ValueError:
        results['optimizer_mismatch_rejected']=True
    else:
        raise AssertionError('Baseline checkpoint accepted as landmark resume')
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
