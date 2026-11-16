"""Wait for a prior queue, then preflight/train/export one unchanged experiment."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text())
    directory=args.manifest.resolve().parent
    status_path=directory/'status.json'
    status=dict(state='waiting_for_current',phase='waiting',supervisor_pid=os.getpid(),child_pid=None,
                dependency_status=manifest['dependency_status'],started=now(),experiment=manifest['name'])
    def save():
        status['updated']=now();temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
    def execute(command,log):
        with open(directory/log,'a',buffering=1) as stream:
            stream.write(now()+' command '+json.dumps(command)+'\n')
            process=subprocess.Popen(command,cwd=manifest['source'],stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid;save()
            return process.wait()
    save()
    try:
        while True:
            dependency=json.loads(Path(manifest['dependency_status']).read_text())
            state=dependency['state']
            if state=='complete':break
            if state!='running':raise RuntimeError('Prior queue did not complete successfully: '+state)
            try:os.kill(dependency['supervisor_pid'],0)
            except ProcessLookupError:raise RuntimeError('Prior supervisor exited without a completed status')
            time.sleep(30)
        with open(Path(manifest['workspace'])/'exp/.experiment_queue.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            status.update(state='running',phase='gpu_preflight',dependency_completed=now());save()
            # The live pointer moves only after the previous queue has completed.
            (Path(manifest['workspace'])/'runs/latest.txt').write_text(str(directory)+'\n')
            program=Path(manifest['source'])/'exp_runner.py'
            config=manifest['config'];output=Path(manifest['output'])
            if output.exists() and any(output.iterdir()):raise RuntimeError('Refusing to overwrite an existing experiment: '+str(output))
            command=[sys.executable,'-u',str(Path(manifest['source'])/'scripts/preflight_dense.py'),
                     '--conf',config,'--output',str(directory/'gpu_preflight')]
            code=execute(command,'preflight.log');status['preflight_exit_code']=code
            if code:
                status.update(state='preflight_failed',phase=None,child_pid=None,
                              reason='Matched settings failed GPU preflight; no automatic batch/logging/mesh changes were made');save();return
            status.update(phase='training');save()
            command=[sys.executable,'-u',str(program),'--conf',config,'--case',manifest['case'],'--type','dtu','--gpu','0']
            code=execute(command,manifest['name']+'.train.log');status['training_exit_code']=code
            if code:
                status.update(state='training_failed',phase=None,child_pid=None);save();return
            status.update(phase='export');save()
            code=execute([*command,'--mode','export_landmarks','--is_continue','--resolution','256'],manifest['name']+'.export.log')
            status['export_exit_code']=code
            if code:
                status.update(state='export_failed',phase=None,child_pid=None);save();return
            status.update(phase='comparison');save()
            exports=[*manifest['comparison_exports'],manifest['name']+'='+str(output/'exports/00030000')]
            command=[sys.executable,str(Path(manifest['source'])/'scripts/evaluate_unified.py'),
                     '--exports',*exports,'--output',str(directory/'comparison.json')]
            code=execute(command,'comparison.log');status['comparison_exit_code']=code
            status.update(state='complete' if code==0 else 'comparison_failed',phase=None,child_pid=None,finished=now());save()
    except Exception as error:
        status.update(state='failed',phase=None,reason=str(error),child_pid=None,finished=now());save()
        raise

if __name__=='__main__':main()
