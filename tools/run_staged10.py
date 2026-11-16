"""Launch the frozen experiment into a separate, portable reproduction directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from pyhocon import ConfigFactory
from verify_reproduction import ROOT, verify


def configuration(output):
    c = ConfigFactory.parse_file(str(ROOT/'confs/staged_10_corrected.conf')).as_plain_ordered_dict()
    def relocate(value):
        if isinstance(value, dict):return {k:relocate(v) for k,v in value.items()}
        if isinstance(value, list):return [relocate(v) for v in value]
        if isinstance(value, str):
            value = value.replace('CASE_NAME','122_1_masked')
            if value.startswith('./'):return str((ROOT/value).resolve())
        return value
    c = relocate(c)
    c['general']['base_exp_dir_geo'] = str(output)
    # Original backup code requires relative recording entries.
    c['general']['recording'] = ['./','./models']
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','resume','export','validate'])
    parser.add_argument('--output',type=Path,default=ROOT/'exp/122_1_masked/staged_10_reproduction/geometry')
    parser.add_argument('--max_steps',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args();output=args.output.resolve()
    frozen = json.loads((ROOT/'reproducibility/source_manifest.json').read_text())
    missing,changed = verify(ROOT,frozen['files'])
    if missing or changed:raise SystemExit('Frozen source verification failed: '+str(missing+changed))
    if args.action=='train' and output.exists() and any(output.iterdir()):
        raise SystemExit('Refusing to train over existing results. Choose a fresh --output or use resume.')
    if args.action!='train' and not (output/'checkpoints').is_dir():
        raise SystemExit('Requested output has no checkpoints. Train first or restore the archived artifacts.')
    c=configuration(output)
    if args.dry_run:
        print(json.dumps(c,indent=2));return
    inputs=json.loads((ROOT/'reproducibility/inputs_manifest.json').read_text())
    missing,changed=verify(ROOT,inputs['files'])
    if missing or changed:raise SystemExit('Private input or baseline verification failed; run tools/verify_reproduction.py --profile inputs')
    output.mkdir(parents=True,exist_ok=True);path=output/'reproduction_config.json';path.write_text(json.dumps(c,indent=2)+'\n')
    mode=dict(train='train',resume='train',export='export_landmarks',validate='validate_landmarks')[args.action]
    cmd=[sys.executable,'-u',str(ROOT/'exp_runner.py'),'--conf',str(path),'--case','122_1_masked','--type','dtu','--gpu','0','--mode',mode]
    if args.action!='train':cmd.append('--is_continue')
    if args.action=='validate':cmd.extend(['--idx','1'])
    if args.max_steps is not None:cmd.extend(['--max_steps',str(args.max_steps)])
    for name,value in [('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4'),('OPENBLAS_NUM_THREADS','1')]:os.environ.setdefault(name,value)
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))


if __name__=='__main__':main()
