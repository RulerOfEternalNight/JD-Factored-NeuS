"""Compare exported geometry/landmarks. No reference means no accuracy claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def chamfer(a, b):
    if not len(a) or not len(b):
        return {'mean_symmetric_l2': None, 'reason': 'No surface samples in this region'}
    ab=cKDTree(b).query(a)[0]; ba=cKDTree(a).query(b)[0]
    return {'mean_symmetric_l2': float((ab.mean()+ba.mean())/2),
            'forward_mean_l2': float(ab.mean()),'backward_mean_l2':float(ba.mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', nargs='+', required=True, help='name=export_directory entries')
    parser.add_argument('--reference_mesh', type=Path, help='Reference surface in the SAME WORLD coordinates')
    parser.add_argument('--regions', type=Path, help='JSON {region: [[xmin,ymin,zmin],[xmax,ymax,zmax]]} in WORLD coordinates')
    parser.add_argument('--samples',type=int,default=50000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('verification/comparison.json'))
    args=parser.parse_args()
    if args.samples<1:parser.error('--samples must be positive')
    np.random.seed(args.seed)
    reference=None
    if args.reference_mesh:
        reference=trimesh.load(args.reference_mesh,force='mesh').sample(args.samples)
    regions=json.loads(args.regions.read_text()) if args.regions else {}
    result={'coordinates':'world','units':'input world units; not assumed millimeters',
            'chamfer_definition':'half the sum of two mean nearest-neighbor L2 distances on seeded surface samples',
            'reference_mesh':str(args.reference_mesh) if args.reference_mesh else None,
            'samples':args.samples,'seed':args.seed,'experiments':{}}
    clouds={}
    for entry in args.exports:
        name,path=entry.split('=',1);path=Path(path)
        metadata=json.loads((path/'landmarks.json').read_text())
        cloud=trimesh.load(path/'mesh_world.ply',force='mesh').sample(args.samples);clouds[name]=cloud
        row={'iteration':metadata['iteration'],'strategy':metadata['strategy'],
             'reprojection':metadata.get('reprojection'), 'landmarks':metadata.get('landmarks',[]),
             'geometry_accuracy':chamfer(cloud,reference) if reference is not None else
             {'mean_symmetric_l2':None,'reason':'No reference mesh supplied'},'regions':{}}
        if reference is not None:
            for region,(lo,hi) in regions.items():
                mask=lambda pts: ((pts>=lo)&(pts<=hi)).all(-1)
                row['regions'][region]=chamfer(cloud[mask(cloud)],reference[mask(reference)])
        result['experiments'][name]=row
    if 'baseline' in clouds:
        result['difference_from_baseline']={name:chamfer(cloud,clouds['baseline']) for name,cloud in clouds.items() if name!='baseline'}
        result['difference_caveat']='Difference from baseline measures change, not improved accuracy.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
