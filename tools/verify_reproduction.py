"""Verify frozen source, private training inputs and/or archived public results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, files):
    missing, changed = [], []
    for relative, record in files.items():
        path = root / relative
        expected = record if isinstance(record, str) else record['sha256']
        if not path.is_file():
            missing.append(relative)
        elif sha(path) != expected:
            changed.append(relative)
    return missing, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['source', 'inputs', 'artifacts', 'all'], default='source')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    profiles = ['source', 'inputs', 'artifacts'] if args.profile == 'all' else [args.profile]
    failed = False
    for profile in profiles:
        path = ROOT/'reproducibility'/f'{profile}_manifest.json'
        manifest = json.loads(path.read_text())
        missing, changed = verify(args.root.resolve(), manifest['files'])
        print(f'{profile}: {len(manifest["files"])} files; {len(missing)} missing; {len(changed)} changed')
        for label, paths in [('MISSING', missing), ('CHANGED', changed)]:
            for item in paths[:15]:
                print(label, item)
        failed |= bool(missing or changed)
    if failed:
        raise SystemExit('Verification failed. See docs/REPRODUCIBILITY.md for required private inputs and public artifacts.')


if __name__ == '__main__':
    main()
