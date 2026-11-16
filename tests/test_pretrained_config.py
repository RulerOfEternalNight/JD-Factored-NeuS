"""CPU integration checks for config-based baseline initialization and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from pyhocon import ConfigFactory
from exp_runner import Runner


class PretrainedConfigTests(unittest.TestCase):
    def test_config_initialization_freeze_and_resume(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)
        root=Path.cwd()
        checkpoint=root/'verification/unified_smoke/baseline/checkpoints/ckpt_000002.pth'
        fixture=root/'verification/unified_smoke/data'
        if not checkpoint.exists():
            self.skipTest('Run tests/smoke_training.py to generate the upstream-format fixture')
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available',return_value=False):
            config=ConfigFactory.parse_file('verification/unified_smoke/staged.conf').as_plain_ordered_dict()
            config['general']['base_exp_dir_geo']=str(Path(tmp)/'experiment')
            config['dataset']['data_dir']=str(fixture)
            config['landmarks']['annotation_path']=str(root/'verification/unified_smoke/annotations.json')
            config['landmarks']['visualization_freq']=0
            config['train']['init_checkpoint']=str(checkpoint)
            path=Path(tmp)/'config.json';path.write_text(json.dumps(config))
            runner=Runner(str(path),case='fixture')
            self.assertEqual(runner.iter_step,2)
            self.assertEqual(runner.current_stage,'head')
            self.assertEqual(len(runner.optimizer.state),0)
            original=torch.load(checkpoint,map_location='cpu')
            networks={'nerf':runner.nerf_outside,'sdf_network_fine':runner.sdf_network,
                      'variance_network_fine':runner.deviation_network,'color_network_fine':runner.color_network,
                      'refColor_network':runner.refColor_network}
            for key,network in networks.items():
                for name,value in network.state_dict().items():
                    torch.testing.assert_close(value,original[key][name],rtol=0,atol=0)
                self.assertTrue(all(not p.requires_grad for p in network.parameters()))
            head_before=copy.deepcopy(runner.landmark_head.state_dict())
            runner.train(max_steps=1)
            for key,network in networks.items():
                for name,value in network.state_dict().items():
                    torch.testing.assert_close(value,original[key][name],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(head_before[k],v) for k,v in runner.landmark_head.state_dict().items()))
            saved_head=copy.deepcopy(runner.landmark_head.state_dict())
            # The source checkpoint no longer needs to exist for staged resume.
            config['train']['init_checkpoint']=str(Path(tmp)/'missing_source.pth')
            path.write_text(json.dumps(config))
            resumed=Runner(str(path),case='fixture',is_continue=True)
            self.assertEqual(resumed.iter_step,3)
            for name,value in resumed.landmark_head.state_dict().items():
                torch.testing.assert_close(value,saved_head[name],rtol=0,atol=0)
            self.assertEqual(resumed.initialization_source['path'],str(checkpoint))

if __name__=='__main__':unittest.main()
