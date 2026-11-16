"""Create a complete private backup outside Git, including original source photos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from verify_reproduction import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.is_relative_to(ROOT.resolve()):raise SystemExit('Private backup must be outside the Git checkout.')
    out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive=out/'staged10_corrected_PRIVATE_FULL_BACKUP.tar.gz'
    if archive.exists():raise SystemExit('Refusing to overwrite an existing private backup.')
    source_run=json.loads((ROOT/'reproducibility/original_manifest.json').read_text())
    original_run=Path(source_run['source']).parent
    paths=[(ROOT/'public_data/122_1_masked','public_data/122_1_masked'),
           (ROOT/'exp/122_1_masked/staged_10_corrected/geometry','exp/122_1_masked/staged_10_corrected/geometry')]
    for relative in [
        'exp/122_1_masked/baseline/geometry/checkpoints/ckpt_010000.pth',
        'exp/122_1_masked/baseline/geometry/meshes/00010000.ply',
        'exp/122_1_masked/baseline/geometry/exports/00030000']:
        paths.append((ROOT/relative,relative))
    if original_run.exists():paths.append((original_run,'runs/original_staged10_corrected'))
    for path,_ in paths:
        if not path.exists():raise SystemExit('Required private input/output is missing: '+str(path))
    with tempfile.TemporaryDirectory(prefix='staged10_private_') as temp:
        source=Path(temp)/'source.tar'
        with source.open('wb') as stream:subprocess.run(['git','archive','--format=tar',commit],cwd=ROOT,stdout=stream,check=True)
        info=Path(temp)/'PRIVATE_BACKUP_INFO.json';info.write_text(json.dumps(dict(commit=commit,branch='softmax_10landmark_working',contains_private_source_photos=True,public_repository='RulerOfEternalNight/JD-Factored-NeuS'),indent=2)+'\n')
        with tarfile.open(archive,'w:gz',compresslevel=6) as package:
            package.dereference=True
            for path,name in paths:package.add(path,arcname=name)
            package.add(source,arcname='source/softmax_10landmark_working.tar')
            package.add(info,arcname=info.name)
    (out/'SHA256SUMS_PRIVATE').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(path=str(archive),MiB=archive.stat().st_size/2**20,commit=commit),indent=2))


if __name__=='__main__':main()
