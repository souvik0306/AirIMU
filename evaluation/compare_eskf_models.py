"""Compare fresh Stage 1/A/B Strong/C inference on identical reserved flights."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import torch

from evaluation.eskf_metrics import covariance_distribution


def improvement(baseline, value):
    return 100*(1-value/baseline) if baseline else (0. if value==0 else None)


def write_csv(path, rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def canonical_flights(metadata):
    rows = json.loads(json.dumps(metadata['flights']))
    for row in rows:row.get('sensor_metadata',{}).pop('source_csv',None)
    return sorted(rows,key=lambda row:row['id'])


def summarize_evaluation(path, expected_flights, *, required_split='test'):
    path = Path(path)
    metadata = json.loads((path/'evaluation_metadata.json').read_text())
    verification = json.loads((path/'evaluation_verification.json').read_text())
    flights = json.loads((path/'metrics.json').read_text())
    if not (path/'COMPLETE').exists() or len(flights)!=expected_flights:
        raise ValueError(f'Incomplete evaluation: {path}')
    if metadata['internal_split']!=required_split or not metadata['full_sequence']:
        raise ValueError(f'Comparison requires complete {required_split} flights.')
    if verification['optimizer_steps'] or not all(verification[key] for key in
            ('weights_and_buffers_unchanged','parameter_gradients_absent','fresh_forward_per_flight','all_trajectories_finite')):
        raise ValueError('Evaluation was not verified as fresh, read-only inference.')
    if len({row['id'] for row in flights})!=expected_flights:raise ValueError('Duplicate evaluation flight IDs.')
    predictions = [];scored_predictions = [];fingerprints = {}
    for row in flights:
        artifact = torch.load(path/row['prediction_file'],map_location='cpu',weights_only=True)
        if artifact['id']!=row['id'] or artifact['checkpoint_sha256']!=metadata['checkpoint_sha256']:
            raise ValueError('Prediction artifact/checkpoint mismatch.')
        if not artifact['fresh_airimu_forward'] or not (path/row['trajectory_file']).is_file():
            raise ValueError('Missing fresh predictions or trajectory artifact.')
        q = artifact['imu_variance'];time = artifact['imu_time']
        if q.shape!=(len(time),6):raise ValueError('Predictions must contain six variances per IMU sample.')
        if row['cutoff_timestamp']+25.>float(time[-1]):raise ValueError('Incomplete evaluation interval.')
        predictions.append(q)
        scored_predictions.append(q[(time>=row['evaluation_start']) & (time<=row['evaluation_end'])])
        digest = hashlib.sha256()
        for key in ('imu_time','raw_acceleration','raw_gyroscope'):
            digest.update(artifact[key].contiguous().numpy().tobytes())
        fingerprints[row['id']] = digest.hexdigest()
    covariance = covariance_distribution(torch.cat(predictions));scored_covariance = covariance_distribution(torch.cat(scored_predictions))
    rmses = [row['ekf_vs_vicon']['xy_rmse'] for row in flights]
    valid = [row for row in flights if row['gyro_reference_quality']['valid_samples']>0]
    valid_samples = sum(row['gyro_reference_quality']['valid_samples'] for row in valid)
    coverage = ([sum(row['gyro_diagnostics']['one_sigma_coverage'][axis]*row['gyro_reference_quality']['valid_samples']
        for row in valid)/valid_samples for axis in range(3)] if valid_samples else None)
    orientations = [row['orientation_errors'] for row in flights]
    orientation_count = sum(row['geodesic_quality_filtered']['samples'] for row in orientations)
    yaw_count = sum(row['yaw_quality_filtered']['samples'] for row in orientations)
    mean_square = mean(row['ekf_vs_vicon']['xy_mse'] for row in flights)
    sample_count = sum(row['velocity_scored_samples'] for row in flights)
    result = {'directory':str(path.resolve()),'flights':len(flights),'checkpoint':metadata['checkpoint'],
        'checkpoint_sha256':metadata['checkpoint_sha256'],
        'aggregate_horizontal_velocity_rmse':math.sqrt(mean_square),'aggregate_horizontal_velocity_mse':mean_square,
        'aggregation':'sqrt(mean of per-flight mean squared XY-vector errors); equal flight weights',
        'sample_pooled_horizontal_velocity_rmse':math.sqrt(sum(row['ekf_vs_vicon']['xy_mse']*row['velocity_scored_samples']
            for row in flights)/sample_count),
        'mean_flight_rmse':mean(rmses),'median_flight_rmse':median(rmses),'worst_flight_rmse':max(rmses),
        'worst_flight_id':max(flights,key=lambda row:row['ekf_vs_vicon']['xy_rmse'])['id'],
        'covariance_full_sequence':covariance,'covariance_scored_interval':scored_covariance,
        'gyro_calibration':{'valid_samples':valid_samples,
            'mean_flight_loss':mean(row['gyro_calibration_loss'] for row in valid) if valid else None,
            'sample_weighted_loss':sum(row['gyro_calibration_loss']*row['gyro_reference_quality']['valid_samples']
                for row in valid)/valid_samples if valid_samples else None,
            'sample_weighted_one_sigma_coverage_xyz':coverage,
            'mean_flight_one_sigma_coverage_xyz':[mean(row['gyro_diagnostics']['one_sigma_coverage'][axis]
                for row in valid) for axis in range(3)] if valid else None},
        'orientation':{'quality_filtered_samples':orientation_count,
            'geodesic_rmse_degrees':math.degrees(math.sqrt(sum(row['quality_filtered_squared_error_sum_rad2']
                for row in orientations)/orientation_count)) if orientation_count else None,
            'yaw_rmse_degrees':math.degrees(math.sqrt(sum(row['quality_filtered_yaw_squared_error_sum_rad2']
                for row in orientations)/yaw_count)) if yaw_count else None},
        'stability':{'all_flights_finite':all(row['stability']['all_trajectory_values_finite'] for row in flights),
            'filter_initializations':sum(row['stability']['filter_initializations'] for row in flights),
            'automatic_filter_restarts':sum(row['stability']['automatic_filter_restarts'] for row in flights),
            'maximum_quaternion_norm_error':max(row['stability']['maximum_quaternion_norm_error'] for row in flights),
            'minimum_sampled_covariance_eigenvalue':min(row['stability']['sampled_minimum_covariance_eigenvalue'] for row in flights),
            'maximum_covariance_asymmetry':max(row['stability']['maximum_covariance_asymmetry'] for row in flights)},
        'optimizer_steps':0,'weights_and_buffers_unchanged':True,'fresh_forward_per_flight':True}
    return result,metadata,flights,fingerprints


def compare(evaluations, output, expected_flights=12, plots=True):
    labels = list(evaluations)
    if labels!=['Stage1','A','BStrong','C']:raise ValueError('Provide Stage1, A, BStrong and C, in that order.')
    loaded = {label:summarize_evaluation(path,expected_flights) for label,path in evaluations.items()}
    base_metadata = loaded['Stage1'][1]
    keys = ('dataset_config_sha256','internal_split','dataset_split_counts','full_sequence','evaluation_interval',
            'height_updates','height_measurement_std','calibration_options')
    for label,(_,metadata,_,fingerprints) in loaded.items():
        if any(metadata.get(key)!=base_metadata.get(key) for key in keys):raise ValueError(f'Evaluation policy differs: {label}')
        if canonical_flights(metadata)!=canonical_flights(base_metadata):raise ValueError(f'Flight identities/biases/cutoffs differ: {label}')
        if fingerprints!=loaded['Stage1'][3]:raise ValueError(f'Raw IMU samples/timestamps differ: {label}')
    by_flight = {label:{row['id']:row for row in values[2]} for label,values in loaded.items()}
    paired = []
    for flight in sorted(by_flight['Stage1']):
        row = {'flight':flight}
        baseline = by_flight['Stage1'][flight]['ekf_vs_vicon']['xy_rmse']
        for label in labels:
            current = by_flight[label][flight]
            if current['gyro_reference_quality']!=by_flight['Stage1'][flight]['gyro_reference_quality']:
                raise ValueError('Calibration masks differ across evaluated models.')
            value = current['ekf_vs_vicon']['xy_rmse']
            row[label+'_xy_rmse'] = value;row[label+'_improvement_percent_vs_Stage1'] = improvement(baseline,value)
        paired.append(row)
    summaries = {}
    for label,(record,_,_,_) in loaded.items():
        record['flights_improved_over_Stage1'] = sum(row[label+'_xy_rmse']<row['Stage1_xy_rmse'] for row in paired)
        record['aggregate_improvement_percent_vs_Stage1'] = improvement(
            loaded['Stage1'][0]['aggregate_horizontal_velocity_rmse'],record['aggregate_horizontal_velocity_rmse'])
        summaries[label] = record
    report = {'split':'reserved dataset.eval; never used for training or checkpoint selection',
        'flight_count':expected_flights,'models':summaries,'paired_flights':paired,
        'checkpoint_selection':'training checkpoints selected exclusively on validation; Stage1 is untouched',
        'policy':{key:base_metadata.get(key) for key in keys}}
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'comparison_flights.csv',paired)
    write_csv(output/'comparison_summary.csv',[{'model':label,**{key:record[key] for key in
        ('aggregate_horizontal_velocity_rmse','mean_flight_rmse','median_flight_rmse','worst_flight_rmse',
         'worst_flight_id','flights_improved_over_Stage1','aggregate_improvement_percent_vs_Stage1')}}
        for label,record in summaries.items()])
    cov_rows = []
    for label,record in summaries.items():
        for scope,key in (('full_sequence','covariance_full_sequence'),('scored_interval','covariance_scored_interval')):
            for sensor in ('acc_cov','gyro_cov'):
                for axis in range(3):
                    cov_rows.append({'model':label,'scope':scope,'sensor':sensor,'axis':'XYZ'[axis],
                        **{name:values[axis] for name,values in record[key][sensor].items()}})
    write_csv(output/'comparison_covariance.csv',cov_rows)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,3,figsize=(17,9),layout='constrained')
        axes[0,0].bar(labels,[summaries[label]['aggregate_horizontal_velocity_rmse'] for label in labels])
        axes[0,0].set(ylabel='Held-out horizontal velocity RMSE (m/s)')
        for label in labels:
            axes[0,1].plot(range(1,len(paired)+1),[row[label+'_xy_rmse'] for row in paired],marker='.',label=label)
            for axis,sensor in ((axes[0,2],'acc_cov'),(axes[1,0],'gyro_cov')):
                axis.plot(list('XYZ'),summaries[label]['covariance_full_sequence'][sensor]['sqrt_mean_variance'],marker='o',label=label)
            coverage = summaries[label]['gyro_calibration']['sample_weighted_one_sigma_coverage_xyz']
            if coverage:axes[1,1].plot(list('XYZ'),coverage,marker='o',label=label)
        axes[0,1].set(xlabel='Flight index (see CSV)',ylabel='Per-flight RMSE (m/s)')
        axes[0,2].set(ylabel='Accel sqrt(mean variance) (m/s²)')
        axes[1,0].set(ylabel='Gyro sqrt(mean variance) (rad/s)')
        axes[1,1].axhline(.682689,linestyle='--',color='gray');axes[1,1].set(ylabel='Gyro one-sigma coverage',ylim=(0,1))
        axes[1,2].bar(labels,[summaries[label]['orientation']['geodesic_rmse_degrees']
            if summaries[label]['orientation']['geodesic_rmse_degrees'] is not None else math.nan for label in labels])
        axes[1,2].set(ylabel='Quality-filtered orientation RMSE (degrees)')
        for axis in axes.flat:
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:axis.legend()
        fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    (output/'COMPLETE').write_text('All four models evaluated on identical held-out flights; no optimizer updates.\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',nargs=4,required=True,metavar='MODEL=PATH')
    parser.add_argument('--output',required=True);parser.add_argument('--expected-flights',type=int,default=12)
    parser.add_argument('--plots',action=argparse.BooleanOptionalAction,default=True)
    args = parser.parse_args()
    evaluations = dict(item.split('=',1) for item in args.evaluations)
    report = compare(evaluations,args.output,args.expected_flights,args.plots)
    for label,row in report['models'].items():
        print(f"{label}: eval_xy_rmse={row['aggregate_horizontal_velocity_rmse']:.9g} "
              f"improved_flights={row['flights_improved_over_Stage1']}/{report['flight_count']}",flush=True)


if __name__=='__main__':main()
