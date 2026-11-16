"""Package saved model results; exclude source-photo overlays and RGB images."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from verify_reproduction import ROOT, sha


def public_result(relative):
    parts=relative.parts;name=relative.name
    if relative.suffix in ['.pth','.ply','.json','.npz']:return True
    if name.startswith('events.out.tfevents.'):return True  # Scalar logs only.
    return ('landmark_validation' in parts and bool(re.fullmatch(r'\d+_(prediction|target|comparison)\.png',name)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'exp/122_1_masked/staged_10_corrected/geometry'
    files=[p for p in sorted(folder.rglob('*')) if p.is_file() and public_result(p.relative_to(folder))]
    files.extend(ROOT/p for p in [
        'exp/122_1_masked/baseline/geometry/checkpoints/ckpt_010000.pth',
        'exp/122_1_masked/baseline/geometry/meshes/00010000.ply',
        'exp/122_1_masked/baseline/geometry/exports/00030000/mesh_world.ply',
        'exp/122_1_masked/baseline/geometry/exports/00030000/landmarks.json'])
    manifest=dict(experiment='staged_10_corrected',files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
                  excludes='Source photos, annotations.png overlays, RGB validations and photographic reprojections are excluded.')
    (ROOT/'reproducibility/artifacts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=output/'staged10-working-artifacts-v1.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    (output/'SHA256SUMS').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(files=len(files),uncompressed_MiB=sum(p.stat().st_size for p in files)/2**20,archive_MiB=archive.stat().st_size/2**20),indent=2))


if __name__=='__main__':main()
