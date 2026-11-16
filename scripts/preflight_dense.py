"""Two matched-setting real-data GPU steps before an expensive dense run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from exp_runner import Runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    config=ConfigFactory.parse_file(str(args.conf)).as_plain_ordered_dict()
    assert config['train']['batch_size']==512 and config['train']['end_iter']==30000
    assert config['landmarks']['num_landmarks']==478
    assert config['landmarks']['training_mode']=='joint_from_zero'
    assert not config['train'].get('init_checkpoint')
    # Only the preflight output path differs; the full schedule and all training
    # settings remain identical. The long run starts fresh in a separate process.
    config['general']['base_exp_dir_geo']=str(args.output/'geometry')
    path=args.output/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    torch.set_default_tensor_type('torch.cuda.FloatTensor');torch.set_num_threads(2)
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    runner=Runner(str(path),case='122_1_masked')
    assert runner.initialization_source is None and runner.iter_step==0 and runner.current_stage=='joint'
    assert runner.annotations.valid.shape==(48,478)
    runner.train(max_steps=2)
    torch.cuda.synchronize()
    events=EventAccumulator(str(args.output/'geometry/logs')).Reload()
    values={name:events.Scalars('LandmarkGradients/'+name)[0].value for name in ['head','sdf','render_weights']}
    assert all(np.isfinite(x) and x>0 for x in values.values()),values
    assert events.Scalars('Loss/landmark')[0].step==0
    assert all(np.isfinite(row.value) for row in events.Scalars('Loss/loss'))
    result=dict(passed=True,steps=2,batch_size=512,num_landmarks=478,
                image_size=[runner.dataset.W,runner.dataset.H],
                initial_checkpoint=None,landmark_only_gradient_norms=values,
                peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_cuda_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                note='Two-step execution/memory check, not a guarantee of later peak memory or anatomical accuracy')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
