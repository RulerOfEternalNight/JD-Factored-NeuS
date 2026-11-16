"""Restore checksummed results without overwriting different existing files."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from verify_reproduction import ROOT, sha


def restore(archive, destination, manifest):
    expected = manifest['files']
    # Validate all member names and types before touching the destination.
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive entries differ from the tracked artifact manifest')
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError('Unsafe archive entry: '+member.name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive would escape destination through a local symlink')
            if target.exists() and (not target.is_file() or sha(target) != expected[member.name]['sha256']):
                raise FileExistsError('Refusing to overwrite a different existing result: '+str(target))
        with tempfile.TemporaryDirectory(prefix='staged10_restore_') as tmp:
            staged = Path(tmp)
            for member in members:
                path = staged/member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as src, path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                if sha(path) != expected[member.name]['sha256']:
                    raise ValueError('Artifact checksum mismatch: '+member.name)
            for member in members:
                target = destination/member.name
                if target.exists():
                    continue  # Identical existing files are preserved.
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged/member.name, target)
    return len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'reproducibility/artifacts_manifest.json').read_text())
    print('Verified/restored files:', restore(args.archive, args.destination.resolve(), manifest))


if __name__ == '__main__':
    main()
