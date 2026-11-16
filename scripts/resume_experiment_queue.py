"""Resume the failed staged experiment without rerunning completed experiments."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());run=args.manifest.resolve().parent
    path=run/'status.json';status=json.loads(path.read_text());row=next(x for x in status['experiments'] if x['name']=='staged')
    experiment=next(x for x in manifest['experiments'] if x['name']=='staged')
    source=Path(manifest['source']);status.setdefault('failure_history',[]).append(dict(time=now(),experiment='staged',previous_status=row.copy()))
    def save():
        status['updated']=now();temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(path)
    def execute(command,log):
        with open(run/log,'a',buffering=1) as stream:
            stream.write('\n'+now()+' RECOVERY command '+json.dumps(command)+'\n')
            process=subprocess.Popen(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid;save();return process.wait()
    with open(Path(manifest['workspace'])/'exp/.experiment_queue.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        row.update(state='running',training_exit_code=None,export_exit_code=None,resumed=now(),resumed_from_iteration=22500)
        status.update(state='running',current_experiment='staged',phase='training',supervisor_pid=os.getpid(),child_pid=None)
        status.pop('finished',None);save()
        command=[sys.executable,'-u',str(source/'scripts/train_with_optimizer_diagnostics.py'),
                 '--diagnostic_path',str(run/'staged.optimizer_failure.json'),
                 '--conf',experiment['config'],'--case',manifest['case'],'--type','dtu','--gpu','0','--is_continue']
        code=execute(command,'staged.train.log');row['training_exit_code']=code
        if code:
            row.update(state='training_failed',finished=now());status.update(state='finished_with_errors',current_experiment=None,phase=None,child_pid=None,finished=now());save();return
        status['phase']='export';save()
        code=execute([sys.executable,'-u',str(source/'exp_runner.py'),'--conf',experiment['config'],
                      '--case',manifest['case'],'--type','dtu','--gpu','0','--is_continue','--mode','export_landmarks','--resolution','256'],'staged.export.log')
        row.update(export_exit_code=code,state='complete' if code==0 else 'export_failed',finished=now())
        if not code:
            status['phase']='comparison';save()
            exports=[x['name']+'='+str(Path(x['output'])/'exports/00030000') for x in manifest['experiments'] if next(y for y in status['experiments'] if y['name']==x['name'])['state']=='complete']
            status['comparison_exit_code']=execute([sys.executable,str(source/'scripts/evaluate_unified.py'),'--exports',*exports,'--output',str(run/'comparison.json')],'comparison.log')
        status.update(state='complete' if row['state']=='complete' and status.get('comparison_exit_code')==0 else 'finished_with_errors',current_experiment=None,phase=None,child_pid=None,finished=now());save()

if __name__=='__main__':main()
