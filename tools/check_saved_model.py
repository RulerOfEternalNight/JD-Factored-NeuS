"""Read-only golden inference check from the original archived final checkpoint."""
import json
import sys
import tempfile
import shutil
import hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from exp_runner import Runner
from run_staged10 import configuration
from verify_reproduction import ROOT


def main():
    if not torch.cuda.is_available():raise SystemExit('Golden real-data inference requires the documented CUDA environment.')
    checkpoint=ROOT/'exp/122_1_masked/staged_10_corrected/geometry/checkpoints/ckpt_030000.pth'
    if not checkpoint.exists():raise SystemExit('Restore the public result artifacts first.')
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    golden=json.loads((ROOT/'results/golden_positive_probe.json').read_text())
    torch.set_default_tensor_type('torch.cuda.FloatTensor');torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='staged10_golden_') as directory:
        output=Path(directory);(output/'checkpoints').mkdir();shutil.copy2(checkpoint,output/'checkpoints/ckpt_030000.pth')
        path=output/'config.json';path.write_text(json.dumps(configuration(output)))
        runner=Runner(str(path),mode='analysis',case='122_1_masked',is_continue=True)
        assert runner.iter_step==30000 and runner.landmark_ids==[1,33,291,263,133,362,61,10,168,152]
        indices=np.flatnonzero(runner.annotations.valid[6]);pixels=np.rint(runner.annotations.xy[6,indices]).astype(np.int64)
        data=runner.dataset.rays_at_pixels(6,pixels);near,far=runner.dataset.near_far_from_sphere(data[:,:3],data[:,3:6])
        out=runner.renderer.render(data[:,:3],data[:,3:6],near,far,perturb_overwrite=0,cos_anneal_ratio=1.)
        values=out['landmark_response'][torch.arange(len(pixels)),torch.as_tensor(indices)].detach().cpu().numpy()
        median=float(np.median(values));expected=golden['rows'][0]['true_class_response']['median']
        np.testing.assert_allclose(median,expected,atol=1e-5,rtol=1e-5)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
        print(json.dumps(dict(iteration=runner.iter_step,head_outputs=11,camera=6,median_true_class_response=median,
                              golden_median=expected,checkpoint_unchanged=True),indent=2))


if __name__=='__main__':main()
