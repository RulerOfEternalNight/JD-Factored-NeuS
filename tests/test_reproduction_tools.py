import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from restore_artifacts import restore
from verify_reproduction import verify,ROOT
from run_staged10 import configuration
from build_artifact_bundle import public_result


class ReproductionToolsTests(unittest.TestCase):
    def archive(self,path,name,content):
        with tarfile.open(path,'w:gz') as package:
            member=tarfile.TarInfo(name);member.size=len(content);package.addfile(member,io.BytesIO(content))

    def test_restore_is_checked_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);archive=p/'a.tar.gz';name='exp/test/value.json';content=b'{"test":1}'
            self.archive(archive,name,content);m=dict(files={name:dict(sha256=hashlib.sha256(content).hexdigest())})
            for _ in range(2):self.assertEqual(restore(archive,p/'restore',m),1)
            (p/'restore'/name).write_bytes(b'other')
            with self.assertRaises(FileExistsError):restore(archive,p/'restore',m)

    def test_reject_corruption_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);archive=p/'a.tar.gz';name='exp/test/value';self.archive(archive,name,b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                restore(archive,p/'restore',dict(files={name:dict(sha256='0'*64)}))
            self.assertFalse((p/'restore'/name).exists())

    def test_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);archive=p/'a.tar.gz';name='../escape';self.archive(archive,name,b'x')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                restore(archive,p/'restore',dict(files={name:dict(sha256=hashlib.sha256(b'x').hexdigest())}))

    def test_source_hashes_and_portable_config(self):
        m=json.loads((ROOT/'reproducibility/source_manifest.json').read_text())
        self.assertEqual(verify(ROOT,m['files']),( [], [] ))
        c=configuration(ROOT/'exp/test_reproduction')
        self.assertEqual(c['landmarks']['num_landmarks'],10);self.assertEqual(c['landmarks']['sigma'],20)
        self.assertEqual(c['train']['batch_size'],512);self.assertEqual(c['train']['end_iter'],30000)
        self.assertEqual(c['general']['recording'],['./','./models'])

    def test_public_archive_excludes_source_photos_and_overlays(self):
        for name in ['landmark_validation/00030000_001/annotations.png','exports/00030000/reprojections/006.png','validations_fine/30000.png']:
            self.assertFalse(public_result(Path(name)))
        self.assertTrue(public_result(Path('landmark_validation/00030000_001/0_prediction.png')))
        self.assertTrue(public_result(Path('checkpoints/ckpt_030000.pth')))
