"""Explicit short real-data checks; uses all native images, never resizes."""
from pathlib import Path
import sys,gc,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from exp_runner import Runner
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root=Path('verification/full_resolution');root.mkdir(parents=True,exist_ok=True)
torch.set_default_tensor_type('torch.cuda.FloatTensor');torch.set_num_threads(2)
results={}
for strategy in ('baseline','joint_from_zero','staged'):
    config=root/(strategy+'.conf')
    config.write_text(f'''include "../../confs/{strategy}.conf"
general.base_exp_dir_geo=./{root}/{strategy}
train {{end_iter=3,batch_size=64,warm_up_end=0,save_freq=1,report_freq=1,val_freq=1000,val_mesh_freq=1000}}
landmarks {{geometry_warmup_iters=1,landmark_warmup_iters=1,joint_iters=1,gradient_log_freq=1,visualization_freq=0}}
''')
    torch.cuda.reset_peak_memory_stats()
    runner=Runner(str(config),case='122_1_masked')
    assert (runner.dataset.W,runner.dataset.H)==(1728,2592)
    assert runner.dataset.images_np.dtype==np.float32
    assert runner.dataset.images.data_ptr()==torch.from_numpy(runner.dataset.images_np).data_ptr()
    assert runner.dataset.masks.data_ptr()==torch.from_numpy(runner.dataset.masks_np).data_ptr()
    runner.train()
    events=EventAccumulator(str(root/strategy/'logs')).Reload()
    assert all(np.isfinite(x.value) for x in events.Scalars('Loss/loss'))
    result={'steps':runner.iter_step,'image_width':runner.dataset.W,'image_height':runner.dataset.H,
            'views':runner.dataset.n_images,'batch_size':runner.batch_size,
            'shared_float32_storage':True,'peak_cuda_memory_MiB':round(torch.cuda.max_memory_allocated()/2**20,1)}
    if strategy!='baseline':
        assert events.Scalars('LandmarkGradients/sdf')[0].value>0
        assert events.Scalars('LandmarkGradients/head')[0].value>0
        result['landmark_only_sdf_gradient_norm']=events.Scalars('LandmarkGradients/sdf')[0].value
        result['landmark_only_head_gradient_norm']=events.Scalars('LandmarkGradients/head')[0].value
    results[strategy]=result
    del runner;gc.collect();torch.cuda.empty_cache()
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
