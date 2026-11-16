"""Prepare a provenance-checked visibility sidecar; never alters detector JSON."""
import sys,argparse,json,hashlib,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2,trimesh
from pyhocon import ConfigFactory
from models.dataset import load_K_Rt_from_P
from models.landmarks import LandmarkAnnotations
from models.landmark_visibility import review_visibility,mask_fingerprint


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf',default='confs/staged_10_corrected.conf')
    parser.add_argument('--case',default='122_1_masked')
    args=parser.parse_args()
    c=ConfigFactory.parse_file(args.conf);lc=c['landmarks'];v=lc['visibility']
    def path(value):return Path(str(value).replace('CASE_NAME',args.case)).resolve()
    data=path(c['dataset.data_dir']);annotation=path(lc['annotation_path']);reference=path(v['reference_mesh']);output=path(v['cache_path'])
    if output==annotation:raise ValueError('Visibility sidecar must not overwrite detector annotations')
    paths=sorted((data/'image').glob('*.png'));first=cv2.imread(str(paths[0]));height,width=first.shape[:2]
    labels=LandmarkAnnotations(annotation,paths,width,height,lc['ids'],lc['names']);cams=np.load(data/c['dataset.render_cameras_name']);projections=[(cams[f'world_mat_{i}']@cams[f'scale_mat_{i}'])[:3] for i in range(len(paths))];centers=[load_K_Rt_from_P(None,P)[1][:3,3] for P in projections]
    mesh=trimesh.load(reference,process=False)
    if v.get('mesh_coordinates','world')=='world':
        xyz=np.c_[mesh.vertices,np.ones(len(mesh.vertices))]@np.linalg.inv(cams['scale_mat_0']).T;mesh.vertices=xyz[:,:3]/xyz[:,3:]
    elif v['mesh_coordinates']!='normalized':raise ValueError('Unknown visibility mesh coordinate system')
    # Boolean masks avoid constructing multi-gigabyte float image stacks.
    masks=[]
    for p in paths:
        m=cv2.imread(str(data/'mask'/p.name),0);masks.append(None if m is None else m>128)
    review=review_visibility(labels,projections,centers,mesh,masks=masks,
                             tolerance=float(v.get('tolerance',.02)),
                             maximum_residual=float(v.get('maximum_reprojection_error',60.)),
                             maximum_surface_distance=float(v.get('maximum_surface_distance',.05)),
                             minimum_views=int(v.get('minimum_views',3)),
                             robust_scale=float(v.get('robust_scale',np.median(lc.get('sigma',20.)))))
    review.update(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  ids=labels.ids,names=labels.names,image_names=labels.image_names,
                  native_size=[width,height],annotation_sha256=sha(annotation),
                  cameras_sha256=sha(data/c['dataset.render_cameras_name']),reference_mesh_sha256=sha(reference),
                  masks_sha256=mask_fingerprint(data,labels.image_names),
                  reference_mesh=str(reference),reference_mesh_coordinates=v.get('mesh_coordinates','world'))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(review,indent=2)+'\n')
    print(output)
    for anchor in review['anchors']:print(anchor['id'],anchor['original_views'],'->',anchor['retained_views'])

if __name__=='__main__':main()
