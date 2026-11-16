import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class FollowupQueueTests(unittest.TestCase):
    def run_queue(self, fail_preflight=False):
        spec=importlib.util.spec_from_file_location('followup',Path('scripts/run_followup_experiment.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'exp').mkdir();(root/'runs').mkdir();directory=root/'runs/dense';directory.mkdir()
            dependency=root/'prior.json';dependency.write_text(json.dumps(dict(state='running',supervisor_pid=os.getpid())))
            config=root/'config.json';config.write_text(json.dumps(dict(train=dict(batch_size=512))))
            manifest=dict(workspace=str(root),source=str(root/'source'),name='joint_from_zero_478',case='fixture',
                          dependency_status=str(dependency),config=str(config),output=str(root/'exp/dense'),comparison_exports=[])
            path=directory/'manifest.json';path.write_text(json.dumps(manifest));calls=[]
            def finish_prior(seconds):
                self.assertEqual(calls,[])
                self.assertFalse((root/'runs/latest.txt').exists())
                dependency.write_text(json.dumps(dict(state='complete')))
            def spawn(command,**kwargs):
                self.assertEqual(json.loads(dependency.read_text())['state'],'complete')
                calls.append(command)
                return SimpleNamespace(pid=12345,wait=lambda:1 if fail_preflight and len(calls)==1 else 0)
            with patch('sys.argv',['followup',str(path)]),patch.object(module.time,'sleep',side_effect=finish_prior),patch.object(module.subprocess,'Popen',side_effect=spawn):
                module.main()
            status=json.loads((directory/'status.json').read_text())
            self.assertEqual(json.loads(config.read_text())['train']['batch_size'],512)
            if fail_preflight:
                self.assertEqual(len(calls),1);self.assertEqual(status['state'],'preflight_failed')
            else:
                self.assertEqual(len(calls),4);self.assertEqual(status['state'],'complete')
                self.assertTrue(calls[0][2].endswith('preflight_dense.py'))
                self.assertIn('--is_continue',calls[2])
    def test_wait_then_preflight_training_export(self):self.run_queue()
    def test_failed_preflight_blocks_training_without_changing_settings(self):self.run_queue(True)

if __name__=='__main__':unittest.main()
