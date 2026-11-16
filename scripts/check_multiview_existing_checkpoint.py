"""Inference-only smoke check on the original three-landmark checkpoint.

Uses a coarse grid for a bounded check; the new experiment's configured maps
remain 432x648. No optimizer, training step, or annotation-based selection.
"""
import sys,json,time,argparse,logging
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,trimesh
from pyhocon import ConfigFactory
from models.fields import SDFNetwork,NeRF,SingleVarianceNetwork,RenderingNetwork,RefColor
from models.landmarks import LandmarkHead,LandmarkAnnotations
from models.dataset import load_K_Rt_from_P
from models.renderer import NeuSRenderer
from models.landmark_validation import LandmarkValidationMixin
from models.landmark_multiview import select_multiview_landmarks,project_vertices
logging.basicConfig(level=logging.INFO,format='%(message)s')

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resolution-level',type=int,default=16);parser.add_argument('--max-views',type=int,default=12);parser.add_argument('--cached',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'verification/staged10_multiview/existing_checkpoint';out.mkdir(parents=True,exist_ok=True)
start=time.monotonic();torch.set_num_threads(4)
mesh=trimesh.load(root/'exp/122_1_masked/joint_from_zero/geometry/exports/00030000/mesh_normalized.ply',process=False);vertices=np.asarray(mesh.vertices)
cam=np.load(root/'public_data/122_1_masked/cameras_sphere.npz')
if args.cached:
 cache=np.load(out/f'cache_level{args.resolution_level}_views{args.max_views}.npz')
 q=cache['probabilities'];maps=list(cache['heatmaps']);projections=cache['projections'];centers=cache['centers'];views=cache['views'].tolist();height,width=maps[0].shape[:2]
else:
 c=ConfigFactory.parse_file(str(root/'confs/joint_from_zero.conf'));m=c['model'];device='cuda';torch.set_default_tensor_type('torch.cuda.FloatTensor');torch.cuda.reset_peak_memory_stats()
 checkpoint=torch.load(root/'exp/122_1_masked/joint_from_zero/geometry/checkpoints/ckpt_030000.pth',map_location=device)
 sdf=SDFNetwork(**m['sdf_network']).cuda();head=LandmarkHead(3).cuda();nerf=NeRF(**m['nerf']).cuda();variance=SingleVarianceNetwork(**m['variance_network']).cuda();color=RenderingNetwork(**m['rendering_network']).cuda();ref=RefColor().cuda()
 for net,key in [(sdf,'sdf_network_fine'),(head,'landmark_head'),(nerf,'nerf'),(variance,'variance_network_fine'),(color,'color_network_fine'),(ref,'refColor_network')]:
  net.load_state_dict(checkpoint[key])
  for par in net.parameters():par.requires_grad_(False)
 renderer=NeuSRenderer(**m['neus_renderer'],nerf=nerf,sdf_network=sdf,landmark_head=head,deviation_network=variance,color_network=color,refColor_network=ref)
 with torch.no_grad():q=np.concatenate([head(sdf(torch.tensor(v,dtype=torch.float32,device=device))[:,1:]).softmax(-1)[:,1:].cpu().numpy() for v in np.array_split(vertices,max(1,int(np.ceil(len(vertices)/8192))))])
 cam=np.load(root/'public_data/122_1_masked/cameras_sphere.npz');views=np.linspace(0,47,args.max_views,dtype=int).tolist();maps=[];projections=[];centers=[]
 for vi in views:
  P=(cam[f'world_mat_{vi}']@cam[f'scale_mat_{vi}'])[:3];K,pose=load_K_Rt_from_P(None,P);width,height=1728//args.resolution_level,2592//args.resolution_level
  grid=np.stack(np.meshgrid(np.linspace(0,1727,width),np.linspace(0,2591,height)),-1).reshape(-1,2);dr=np.c_[grid,np.ones(len(grid))]@np.linalg.inv(K[:3,:3]).T;dr/=np.linalg.norm(dr,axis=1,keepdims=True);dr=dr@pose[:3,:3].T;rd=torch.tensor(dr,dtype=torch.float32,device=device);ro=torch.tensor(pose[:3,3],dtype=torch.float32,device=device).expand_as(rd);chunks=[]
  for oo,dd in zip(ro.split(512),rd.split(512)):
   mid=-(oo*dd).sum(-1,keepdim=True);render=renderer.render(oo,dd,mid-1,mid+1,perturb_overwrite=0,cos_anneal_ratio=1.)
   chunks.append(render['landmark_response'].detach().cpu().numpy())
  heatmap=np.concatenate(chunks).reshape(height,width,3);assert np.isfinite(heatmap).all();maps.append(heatmap);projections.append(P);centers.append(pose[:3,3]);print('Rendered camera',vi,'elapsed',round(time.monotonic()-start,1),flush=True)
 np.savez_compressed(out/f'cache_level{args.resolution_level}_views{args.max_views}.npz',probabilities=q,heatmaps=np.array(maps),projections=np.array(projections),centers=np.array(centers),views=np.array(views))
 print('Saved predicted maps; selecting surface points',flush=True)
options=ConfigFactory.parse_file(str(root/'confs/landmarks10_sigma20.conf'))['landmarks.multiview'].as_plain_ordered_dict();result=select_multiview_landmarks(mesh,q,maps,projections,centers,1728,2592,12.,options)
# Coordinates enter only now, after the selections have been fixed.
paths=sorted((root/'public_data/122_1_masked/image').glob('*.png'));labels=LandmarkAnnotations(root/'annotations/122_1_masked/mediapipe_reliable_v2.json',paths,1728,2592,[1,33,291])
for k,row in enumerate(result):
 row['id']=[1,33,291][k]
 if row['valid']:
  row['normalized']=vertices[row['vertex_index']].tolist();errors=[]
  for vi in range(48):
   if not labels.valid[vi,k]:continue
   xy,z=project_vertices(vertices[row['vertex_index']:row['vertex_index']+1],(cam[f'world_mat_{vi}']@cam[f'scale_mat_{vi}'])[:3])
   if z[0]>0:errors.append(float(np.linalg.norm(xy[0]-labels.xy[vi,k])))
  row['mean_reprojection_error_native_px']=float(np.mean(errors)) if errors else None
  row['supporting_views']=[views[i] for i in row['supporting_views']];row['visible_views']=[views[i] for i in row['visible_views']]
summary=dict(experiment='original_joint_3_checkpoint_inference_only',trained_new_model=False,heatmap_resolution_level=args.resolution_level,heatmap_size=[width,height],views=views,selections=result,elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=0 if args.cached else torch.cuda.max_memory_allocated(),note='Coarse smoke check, not the new ten-landmark model or a controlled evaluation.')
(out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
