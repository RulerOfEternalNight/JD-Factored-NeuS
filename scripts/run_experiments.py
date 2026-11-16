"""Run a prepared, immutable three-experiment manifest sequentially on one GPU."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    status_path = directory / 'status.json'
    lock = open(Path(manifest['workspace']) / 'exp' / '.experiment_queue.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    status = dict(started=now(), updated=now(), supervisor_pid=os.getpid(), state='running',
                  current_experiment=None, phase=None, experiments=[])

    def save():
        status['updated'] = now()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(status_path)

    def execute(command, log, cwd):
        with open(log, 'a', buffering=1) as stream:
            stream.write('\n' + now() + ' command: ' + json.dumps(command) + '\n')
            process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
            status['child_pid'] = process.pid; save()
            return process.wait()

    save()
    successful = []
    for experiment in manifest['experiments']:
        name = experiment['name']
        row = dict(name=name, state='running', started=now(), training_exit_code=None, export_exit_code=None)
        status['experiments'].append(row)
        status.update(current_experiment=name, phase='training'); save()
        command = [sys.executable, '-u', str(Path(manifest['source']) / 'exp_runner.py'),
                   '--conf', experiment['config'], '--case', manifest['case'], '--type', 'dtu', '--gpu', '0']
        code = execute(command, directory / (name + '.train.log'), manifest['source'])
        row['training_exit_code'] = code
        if code:
            row.update(state='training_failed', finished=now()); save()
            continue
        status['phase'] = 'export'; save()
        export = [*command, '--mode', 'export_landmarks', '--is_continue', '--resolution', '256']
        code = execute(export, directory / (name + '.export.log'), manifest['source'])
        row.update(export_exit_code=code, state='complete' if code == 0 else 'export_failed', finished=now()); save()
        if code == 0:
            successful.append(name + '=' + str(Path(experiment['output']) / 'exports' / '00030000'))
    if successful:
        status.update(current_experiment=None, phase='comparison'); save()
        command = [sys.executable, str(Path(manifest['source']) / 'scripts' / 'evaluate_unified.py'),
                   '--exports', *successful, '--output', str(directory / 'comparison.json')]
        status['comparison_exit_code'] = execute(command, directory / 'comparison.log', manifest['source'])
    status.update(state='complete' if all(row['state'] == 'complete' for row in status['experiments']) and status.get('comparison_exit_code') == 0 else 'finished_with_errors',
                  current_experiment=None, phase=None, child_pid=None, finished=now())
    save()


if __name__ == '__main__':
    main()
