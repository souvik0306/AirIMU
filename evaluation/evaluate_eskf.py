"""Fresh AirIMU inference and read-only EKF evaluation on disjoint flight splits."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback

import torch

from ekf import DifferentiableESKF, velocity_evaluation_mask
from ekf.data import apply_sensor_metadata, prepare_flight, read_dataset_flights
from ekf.experiment import filter_metadata, load_airimu, xy_metrics, file_sha256
from ekf.regularization import (gyro_reference, gyro_calibration_loss, gyro_diagnostics,
                                log_variance_smoothness, compare_gyro_references)
from evaluation.eskf_metrics import covariance_distribution, orientation_metrics, numerical_stability


def write_json_atomic(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    requested_split = args.split
    split = 'test' if requested_split == 'eval' else requested_split
    all_flights = read_dataset_flights(args.dataset_config)
    counts = {name:sum(row['split']==name for row in all_flights) for name in ('train','validation','test')}
    flights = [row for row in all_flights if row['split']==split]
    if args.expected_flights is not None and len(flights)!=args.expected_flights:
        raise ValueError(f'Expected {args.expected_flights} {requested_split} flights, got {len(flights)}.')
    if args.limit_flights is not None:flights = flights[:args.limit_flights]
    if not flights:raise ValueError(f'No flights in split={requested_split}.')
    flights,cutoffs = apply_sensor_metadata(flights,args.sensor_metadata_root)
    ekf_device = args.ekf_device or args.device
    height_updates = args.height_updates
    if height_updates is None:
        checkpoint = torch.load(args.ckpt,map_location='cpu',weights_only=True) if args.ckpt else {}
        height_updates = bool(checkpoint.get('ekf_metadata',{}).get('height_updates',True));del checkpoint
    ekf = DifferentiableESKF(gravity=args.gravity,height_updates=height_updates).to(ekf_device)
    wrapper = load_airimu(args.config,args.ckpt,ekf,args.device) if args.ckpt else None
    before = {}
    if wrapper:
        wrapper.eval().requires_grad_(False)
        before = {name:value.detach().cpu().clone() for name,value in wrapper.network.state_dict().items()}
        if any(module.training for module in wrapper.modules()):raise RuntimeError('Evaluation has a training-mode module.')
    output = Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists():raise FileExistsError(f'Use a fresh evaluation directory: {output}')
    (output/'predictions').mkdir(exist_ok=True)
    checkpoint_hash = file_sha256(args.ckpt) if args.ckpt else None
    metadata = {'requested_split':requested_split,'internal_split':split,'dataset_split_counts':counts,
        'dataset_config':str(Path(args.dataset_config).resolve()),'dataset_config_sha256':file_sha256(args.dataset_config),
        'config':str(Path(args.config).resolve()) if args.config else None,
        'checkpoint':str(Path(args.ckpt).resolve()) if args.ckpt else None,'checkpoint_sha256':checkpoint_hash,
        'model_mode':'eval','torch_grad_enabled':False,'optimizer_steps':0,'fresh_forward_per_flight':bool(wrapper),
        'full_sequence':args.full_sequence,'evaluation_interval':'cutoff+5 to cutoff+25 seconds',
        'flights':[{**row,'cutoff_timestamp':cutoffs[row['id']]} for row in flights],
        'calibration_options':{'window_seconds':args.calibration_window,'orientation_std':args.orientation_std,
            'max_step_degrees':args.calibration_max_step_deg,'max_angular_rate':args.calibration_max_rate},
        **filter_metadata(ekf)}
    smoothing=getattr(args,'orientation_smoothing_window',0.)
    if smoothing:metadata['calibration_options']['orientation_smoothing_window']=smoothing
    write_json_atomic(output/'evaluation_metadata.json',metadata)
    reports = []
    try:
        with torch.no_grad():
            for index,record in enumerate(flights):
                started = perf_counter();cutoff = cutoffs[record['id']]
                print(f"split={requested_split} flight={record['id']} progress={index+1}/{len(flights)} status=started",flush=True)
                flight = prepare_flight(record,cutoff,ekf,full_sequence=args.full_sequence)
                if wrapper:
                    variance = wrapper.predict_variance(flight['raw_acceleration'],flight['raw_gyroscope'])
                else:
                    variance = torch.tensor([args.accel_variance]*3+[args.gyro_variance]*3,
                        dtype=torch.double,device=ekf_device).repeat(len(flight['imu_time']),1)
                if variance.requires_grad or variance.shape != (len(flight['imu_time']),6):
                    raise RuntimeError('Inference must produce [N,6] variances without an autograd graph.')
                if wrapper and bool((variance<=0).any()):raise RuntimeError('Learned variance must be positive.')
                distribution = covariance_distribution(variance)
                prediction_name = f'predictions/flight_{index:04d}.pt'
                torch.save({'id':record['id'],'checkpoint_sha256':checkpoint_hash,'fresh_airimu_forward':bool(wrapper),
                    'imu_time':flight['imu_time'].cpu(),'raw_acceleration':flight['raw_acceleration'].cpu(),
                    'raw_gyroscope':flight['raw_gyroscope'].cpu(),'imu_variance':variance.cpu()},output/prediction_name)
                inferred = perf_counter()
                trajectory = ekf.run_sequence({'acc':flight['raw_acceleration'],'gyro':flight['raw_gyroscope']},
                    flight['imu_time'],variance,flight['vicon_events'],cutoff,flight['initial_state'],
                    flight['initial_covariance'],return_covariance=True)
                replayed = perf_counter()
                mask = velocity_evaluation_mask(flight['imu_time'],cutoff)
                reference = gyro_reference(flight['imu_time'],flight['vicon_quaternion_gt'],
                    orientation_time=flight['vicon_time_gt'],window_seconds=args.calibration_window,
                    orientation_std=args.orientation_std,max_step_degrees=args.calibration_max_step_deg,
                    max_angular_rate=args.calibration_max_rate,smoothing_window_seconds=smoothing)
                diagnostic_args = {};gyro_loss = variance.new_zeros(())
                if bool(reference['mask'].any()):
                    gyro_loss,residual,effective_q,valid = gyro_calibration_loss(
                        variance,flight['raw_gyroscope'],trajectory['gyroscope_bias'],reference)
                    diagnostic_args = dict(residual=residual,effective_q=effective_q,mask=valid)
                stability = numerical_stability(trajectory,flight)
                if not stability['all_trajectory_values_finite']:raise RuntimeError('Non-finite evaluated trajectory.')
                report = {'id':record['id'],'split':requested_split,'noise_source':'airimu' if wrapper else 'fixed',
                    'checkpoint_sha256':checkpoint_hash,'fresh_airimu_forward':bool(wrapper),'prediction_file':prediction_name,
                    'cutoff_timestamp':cutoff,'evaluation_start':cutoff+5.,'evaluation_end':cutoff+25.,
                    'sequence_start':float(flight['imu_time'][0]),'sequence_end':float(flight['imu_time'][-1]),
                    'full_sequence':args.full_sequence,'gravity':list(args.gravity),
                    'observation_types':list(ekf.observation_types),'network_device':args.device,'ekf_device':ekf_device,
                    **filter_metadata(ekf),
                    'height_update_count':sum(event.observation_type=='height' for event in flight['vicon_events']),
                    'covariance':distribution,'covariance_scored_interval':covariance_distribution(variance[mask]),
                    'gyro_calibration_loss':float(gyro_loss),'gyro_reference_quality':reference['quality'],
                    'gyro_reference_comparison':compare_gyro_references(variance,flight['raw_gyroscope'],
                                                                        trajectory['gyroscope_bias'],reference),
                    'gyro_diagnostics':gyro_diagnostics(variance,flight['imu_time'],flight['raw_gyroscope'],
                        floor=float(wrapper.network.gyro_cov_min) if wrapper else 1e-8,**diagnostic_args),
                    'smoothness_loss':float(log_variance_smoothness(variance,flight['imu_time'])),
                    'calibration_window_seconds':args.calibration_window,'orientation_std_rad':args.orientation_std,
                    'orientation_errors':orientation_metrics(trajectory,flight,mask,reference['mask']),
                    'stability':stability,'ekf_vs_vicon':xy_metrics(trajectory['velocity'],flight['vicon_velocity_gt'],mask),
                    'velocity_scored_samples':int(mask.sum()),
                    'inference_seconds':inferred-started,'replay_seconds':replayed-inferred}
                artifact = f'flight_{index:04d}.pt'
                torch.save({'id':record['id'],'imu_time':flight['imu_time'].cpu(),
                    'trajectory':{key:value.cpu() for key,value in trajectory.items()},'imu_variance':variance.cpu(),
                    'vicon_velocity_gt':flight['vicon_velocity_gt'].cpu(),
                    'vicon_quaternion_gt':flight['vicon_quaternion_gt'].cpu(),'vicon_time_gt':flight['vicon_time_gt'].cpu(),
                    **({'gyro_reference_original':reference['original_rate'],
                        'gyro_reference_filtered':reference['rate'],'gyro_reference_variance':reference['variance'],
                        'gyro_reference_original_variance':reference['original_variance'],
                        'orientation_smoothing':reference['smoothing']} if smoothing else {}),
                    'evaluation_mask':mask.cpu(),'gyro_reference_mask':reference['mask'].cpu(),'report':report},output/artifact)
                report.update(trajectory_file=artifact,elapsed_seconds=perf_counter()-started);reports.append(report)
                write_json_atomic(output/'metrics.json',reports)
                print(json.dumps(report),flush=True)
                del trajectory,variance,flight
        if wrapper and any(not torch.equal(value.detach().cpu(),before[name]) for name,value in wrapper.network.state_dict().items()):
            raise RuntimeError('Evaluation changed model parameters or buffers.')
        if wrapper and any(p.grad is not None for p in wrapper.network.parameters()):
            raise RuntimeError('Evaluation accumulated parameter gradients.')
        verification = {'flights_completed':len(reports),'flight_ids':[row['id'] for row in reports],
            'checkpoint_sha256':checkpoint_hash,'model_mode':'eval','weights_and_buffers_unchanged':True,
            'parameter_gradients_absent':True,'optimizer_steps':0,'fresh_forward_per_flight':bool(wrapper),
            'full_sequence':args.full_sequence,'all_trajectories_finite':True}
        write_json_atomic(output/'evaluation_verification.json',verification)
        if args.plots:
            from evaluation.plot_eskf import plot_evaluation
            plot_evaluation(output)
        write_json_atomic(output/'COMPLETE',verification)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            (output/'FAILED').write_text(f'{type(error).__name__}: {error}\n')
        except OSError as marker_error:
            print(f'Could not write failure marker: {marker_error}',file=sys.stderr,flush=True)
        raise
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config',required=True)
    parser.add_argument('--sensor-metadata-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=('train','validation','test','eval'),default='validation',
                        help='eval/test both mean the reserved dataset.eval split; validation means dataset.test')
    parser.add_argument('--expected-flights',type=int)
    parser.add_argument('--limit-flights',type=int,help='Preflight only: evaluate the first N flights of the requested split')
    parser.add_argument('--full-sequence',action=argparse.BooleanOptionalAction,default=False,
                        help='Infer/replay all real IMU samples; score remains cutoff+5 to cutoff+25')
    parser.add_argument('--plots',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--calibration-window',type=float,default=.05)
    parser.add_argument('--orientation-std',type=float,default=.001)
    parser.add_argument('--calibration-max-step-deg',type=float,default=45.)
    parser.add_argument('--calibration-max-rate',type=float,default=20.)
    parser.add_argument('--orientation-smoothing-window',type=float,default=0.)
    parser.add_argument('--config');parser.add_argument('--ckpt')
    parser.add_argument('--accel-variance',type=float,default=.001)
    parser.add_argument('--gyro-variance',type=float,default=.001)
    parser.add_argument('--gravity',type=float,nargs=3,default=(0.,0.,-9.80665))
    parser.add_argument('--device',default='cpu');parser.add_argument('--ekf-device')
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--height-updates',action=argparse.BooleanOptionalAction,default=None)
    args = parser.parse_args()
    if not math.isfinite(args.orientation_smoothing_window) or args.orientation_smoothing_window<0:
        parser.error('orientation-smoothing-window must be finite and nonnegative.')
    if bool(args.config)!=bool(args.ckpt):parser.error('Pass both --config and --ckpt, or neither.')
    if args.cpu_threads<1 or any(not math.isfinite(x) or x<0 for x in (args.accel_variance,args.gyro_variance)):
        parser.error('Use positive cpu-threads and finite nonnegative fixed variances.')
    if args.expected_flights is not None and args.expected_flights<1:parser.error('expected-flights must be positive.')
    if args.limit_flights is not None and args.limit_flights<1:parser.error('limit-flights must be positive.')
    evaluate(args)


if __name__=='__main__':main()
