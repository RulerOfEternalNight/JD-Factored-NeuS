"""Bounded real-data verification of corrected staged training, not a full run."""
import sys,json,gc,argparse,datetime,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from pyhocon import ConfigFactory
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from exp_runner import Runner


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=5);args=parser.parse_args()
    if not 1 <= args.steps <= 20:parser.error('This smoke check permits 1–20 steps only')
    root=Path(__file__).resolve().parents[1];folder=root/'verification'/('corrected_smoke_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir(parents=True)
    c=ConfigFactory.parse_file(str(root/'confs/staged_10_corrected.conf')).as_plain_ordered_dict()
    c['general']['base_exp_dir_geo']=str(folder/'geometry');c['landmarks']['visualization_freq']=0;c['landmarks']['gradient_log_freq']=1
    # Short invocation only: the 30k learning-rate/schedule configuration stays intact.
    path=folder/'config.json';path.write_text(json.dumps(c,indent=2)+'\n')
    torch.set_default_tensor_type('torch.cuda.FloatTensor');torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    runner=Runner(str(path),case='122_1_masked')
    assert (runner.dataset.W,runner.dataset.H,runner.batch_size)==(1728,2592,512)
    assert runner.iter_step==10000 and runner.current_stage=='head'
    before={name:{k:v.detach().cpu().clone() for k,v in network.state_dict().items()} for name,network in
            [('sdf',runner.sdf_network),('variance',runner.deviation_network),('color',runner.color_network),('ref',runner.refColor_network),('nerf',runner.nerf_outside)]}
    initial_head={k:v.detach().cpu().clone() for k,v in runner.landmark_head.state_dict().items()}
    runner.train(max_steps=args.steps)
    def unchanged(r):
        return all(torch.equal(value,network.state_dict()[k].cpu()) for name,network in
                   [('sdf',r.sdf_network),('variance',r.deviation_network),('color',r.color_network),('ref',r.refColor_network),('nerf',r.nerf_outside)]
                   for k,value in before[name].items())
    assert unchanged(runner)
    assert any(not torch.equal(v,runner.landmark_head.state_dict()[k].cpu()) for k,v in initial_head.items())
    cache=copy.deepcopy(runner.hard_negative_cache.state_dict())
    _,expected_pixels,expected_image=runner.sample_training_data(0,True);expected_pixels=expected_pixels.cpu().numpy()
    end=runner.iter_step;del runner;gc.collect();torch.cuda.empty_cache()
    runner=Runner(str(path),case='122_1_masked',is_continue=True)
    assert runner.iter_step==end and runner.hard_negative_cache.state_dict()==cache
    _,actual_pixels,actual_image=runner.sample_training_data(0,True)
    assert actual_image==expected_image;np.testing.assert_array_equal(actual_pixels.cpu().numpy(),expected_pixels)
    runner.train(max_steps=1);assert unchanged(runner)
    # Read-only joint-gradient probe; no geometry optimizer step is performed.
    for param in runner.reconstruction_parameters:param.requires_grad_(True)
    class NoWriter:
        def add_scalar(self,*args,**kwargs):pass
    runner.writer=NoWriter()
    data,pixels,image=runner.sample_training_data(1,True);ro,rd=data[:,:3],data[:,3:6];near,far=runner.dataset.near_far_from_sphere(ro,rd)
    output=runner.renderer.render(ro,rd,near,far,perturb_overwrite=0,cos_anneal_ratio=1.)
    loss=runner.landmark_objective(output['landmark_response'],image,pixels,render_out=output,foreground=data[:,9:10])
    sdf_grad=torch.autograd.grad(loss,list(runner.sdf_network.parameters()),retain_graph=True)
    weight_grad=torch.autograd.grad(loss,output['landmark_weights'])[0]
    sdf_norm=float(sum((g*g).sum() for g in sdf_grad).sqrt());assert sdf_norm>0 and torch.isfinite(weight_grad).all() and weight_grad.abs().sum()>0
    events=EventAccumulator(str(folder/'geometry/logs')).Reload()
    values={name:[float(v.value) for v in events.Scalars(name)] for name in ['Loss/loss','Loss/landmark_heatmap','Loss/nonlandmark_surface_ce','Landmarks/classification_rays','Landmarks/hard_negative_cache_size']}
    assert all(np.isfinite(v).all() for v in values.values());assert max(values['Landmarks/classification_rays'])>0
    result=dict(experiment='bounded_corrected_smoke_only',training_steps=args.steps+1,final_iteration=runner.iter_step,
                native_size=[runner.dataset.W,runner.dataset.H],batch_size=runner.batch_size,
                reconstruction_bitwise_unchanged=True,head_changed=True,resume_rng_and_hard_cache_verified=True,
                unfrozen_landmark_only_sdf_gradient_norm=sdf_norm,rendering_weight_gradient_norm=float(weight_grad.norm()),
                peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,scalars=values,
                caution='Operational smoke test, not evidence of improved landmark accuracy or a full training run.')
    (folder/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('OUTPUT',folder)

if __name__=='__main__':main()
