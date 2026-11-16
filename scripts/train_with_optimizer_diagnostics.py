"""Run the original entrypoint unchanged; capture Adam scalar state on failure."""
import argparse
import datetime
import json
from pathlib import Path
import runpy
import sys
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic_path',required=True,type=Path)
    args,runner_args=parser.parse_known_args()
    source=Path(__file__).resolve().parents[1]
    original_step=torch.optim.Adam.step
    def diagnosed_step(optimizer,*step_args,**step_kwargs):
        try:return original_step(optimizer,*step_args,**step_kwargs)
        except Exception as error:
            report=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),frames=[])
            traceback=error.__traceback__
            while traceback is not None:
                frame=traceback.tb_frame
                if frame.f_code.co_filename.endswith('/optim/adam.py'):
                    row=dict(function=frame.f_code.co_name,line=traceback.tb_lineno)
                    for name in ['step','lr','beta1','beta2','bias_correction1','bias_correction2','i']:
                        value=frame.f_locals.get(name)
                        if isinstance(value,(int,float,bool,str)):row[name]=value
                    value=frame.f_locals.get('step_t')
                    if torch.is_tensor(value):
                        row['step_tensor']=dict(device=str(value.device),dtype=str(value.dtype),value=float(value.detach().cpu()))
                    report['frames'].append(row)
                traceback=traceback.tb_next
            try:
                report['optimizer_step_values']=[dict(device=str(state['step'].device),value=float(state['step'].detach().cpu())) for state in optimizer.state.values() if 'step' in state]
            except Exception as diagnostic_error:report['diagnostic_error']=repr(diagnostic_error)
            args.diagnostic_path.write_text(json.dumps(report,indent=2)+'\n')
            raise
    torch.optim.Adam.step=diagnosed_step
    sys.path.insert(0,str(source));sys.argv=[str(source/'exp_runner.py'),*runner_args]
    runpy.run_path(str(source/'exp_runner.py'),run_name='__main__')

if __name__=='__main__':main()
