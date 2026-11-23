"""Export one selected EKF-trained AirIMU checkpoint and require numerical parity."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ekf import DifferentiableESKF
from ekf.experiment import file_sha256, load_airimu
from export_onnx_fp32 import CodeNetWrapper, export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--expected-epoch',type=int,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--validation-artifact',type=Path,required=True)
    parser.add_argument('--config',type=Path,
                        help='Exact model config; defaults to RUN_ROOT/configs/stage2.conf')
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    root=args.run_root.resolve();checkpoint=root/'training/best_model.ckpt'
    config=args.config.resolve() if args.config else root/'configs/stage2.conf'
    if file_sha256(checkpoint)!=args.expected_sha256:raise ValueError('Selected checkpoint hash differs.')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['epoch']!=args.expected_epoch:raise ValueError(f"Expected epoch {args.expected_epoch}, got {saved['epoch']}")
    output=root/'export';output.mkdir(exist_ok=True)
    if any((output/name).exists() for name in ('airimu_cpu_fp32.onnx','airimu_cpu_fp32.pt','COMPLETE')):
        raise FileExistsError('Export artifacts already exist; preserve the previous export.')
    model_path=output/'airimu_cpu_fp32.onnx';weights_path=output/'airimu_cpu_fp32.pt'
    print(f'Exporting epoch {saved["epoch"]}: {checkpoint}; log={output/"export.log"}',flush=True)
    try:
        with (output/'export.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
            export(str(config),str(checkpoint),str(model_path),str(weights_path),17,200)
        model=onnx.load(str(model_path));onnx.checker.check_model(model)
        reference=load_airimu(config,checkpoint,DifferentiableESKF(),'cpu').eval().requires_grad_(False)
        initial={name:value.clone() for name,value in reference.network.state_dict().items()}
        exported=torch.load(weights_path,map_location='cpu',weights_only=True)
        if set(exported)!=set(initial):raise ValueError('Exported weight keys differ.')
        for name,value in initial.items():
            expected=value.float() if value.is_floating_point() else value
            if not torch.equal(exported[name],expected):raise ValueError(f'Export changed weights/buffers: {name}')
        wrapper=CodeNetWrapper(reference.network).eval().requires_grad_(False)
        wrapper.net._inference_printed=True;wrapper.net._output_printed=True
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
        if [x.name for x in session.get_inputs()]!=['acc','gyro'] or any(x.shape!=[1,209,3] for x in session.get_inputs()):
            raise ValueError('Unexpected exported input interface.')
        if [x.name for x in session.get_outputs()]!=['corr_acc','corr_gyro','cov_acc','cov_gyro']:
            raise ValueError('Unexpected exported output interface.')
        flight=torch.load(args.validation_artifact,map_location='cpu',weights_only=True)
        cases=[('stationary',torch.tensor([0.,0.,9.80665]).repeat(200,1),torch.zeros(200,3)),
               ('random_motion',torch.randn(200,3)*3+torch.tensor([0.,0.,9.80665]),torch.randn(200,3)*.3)]
        for i,start in enumerate((0,len(flight['imu_time'])//2,len(flight['imu_time'])-200)):
            cases.append((f'real_validation_window_{i}',flight['raw_acceleration'][start:start+200].float(),
                          flight['raw_gyroscope'][start:start+200].float()))
        checks=[]
        with torch.no_grad():
            for label,acc,gyro in cases:
                padded_acc=torch.cat((acc[:1].expand(9,-1),acc))[None]
                padded_gyro=torch.cat((gyro[:1].expand(9,-1),gyro))[None]
                expected=wrapper(padded_acc,padded_gyro)
                actual=session.run(None,{'acc':padded_acc.numpy(),'gyro':padded_gyro.numpy()})
                differences={}
                for index,name in enumerate(('corr_acc','corr_gyro','cov_acc','cov_gyro')):
                    value=actual[index];target=expected[index].numpy()
                    if value.shape!=(1,200,3) or value.dtype!=np.float32 or not np.isfinite(value).all():
                        raise ValueError(f'Invalid {label}/{name} output.')
                    np.testing.assert_allclose(value,target,rtol=1e-4,atol=1e-8 if name=='cov_gyro' else 1e-6)
                    if name.startswith('cov_') and not (value>0).all():raise ValueError('Covariance is not positive.')
                    differences[name]=float(np.abs(value-target).max())
                q=reference.predict_variance(acc,gyro).numpy()
                actual_q=np.concatenate(actual[2:],axis=-1)[0]
                np.testing.assert_allclose(actual_q,q,rtol=1e-4,atol=1e-8)
                checks.append({'input':label,'max_absolute_difference':differences,
                    'covariance_vs_EKF_training_wrapper_max_absolute_difference':float(np.abs(actual_q-q).max())})
        if any(not torch.equal(value,initial[name]) for name,value in reference.network.state_dict().items()):
            raise ValueError('Verification changed network parameters or BatchNorm buffers.')
        if file_sha256(checkpoint)!=args.expected_sha256:raise ValueError('Source checkpoint changed during export.')
        report={'checkpoint':str(checkpoint),'checkpoint_epoch':saved['epoch'],'checkpoint_sha256':args.expected_sha256,
            'original_checkpoint_and_BatchNorm_preserved':True,'model_mode':'eval','optimizer_steps':0,
            'source_config':str(config),
            'residual_covariance_adapter_included':reference.network.raw_cov_adapter is not None,
            'precision':'float32','opset':17,'inputs':{'acc':[1,209,3],'gyro':[1,209,3]},
            'outputs':{name:[1,200,3] for name in ('corr_acc','corr_gyro','cov_acc','cov_gyro')},
            'padding':'9 leading context frames supplied by caller; checks repeat first raw sample',
            'GRU_state':'zero initial hidden states on each call; stateless fixed window',
            'covariance_outputs':'diagonal variances: accel m^2/s^4, gyro rad^2/s^2',
            'EKF_usage':'Feed raw IMU and covariance outputs to the EKF; learned corrections were not applied in these experiments.',
            'onnx_checker_passed':True,'onnxruntime_provider':'CPUExecutionProvider',
            'verification_inputs':checks,'source_config_sha256':file_sha256(config),
            'artifacts':{path.name:{'bytes':path.stat().st_size,'sha256':file_sha256(path)} for path in (model_path,weights_path)},
            'completed_sgt':datetime.now(ZoneInfo('Asia/Singapore')).isoformat()}
        (output/'export_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'COMPLETE').write_text(report['completed_sgt']+'\n')
        print(f'Export verified: {model_path}; 5 input cases, all four outputs and EKF covariance parity passed.',flush=True)
    except Exception as error:
        (output/'FAILED').write_text(f'{type(error).__name__}: {error}\n')
        raise


if __name__=='__main__':main()
