"""Read-only covariance, orientation and numerical diagnostics for EKF replay."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


def covariance_distribution(variance):
    values = variance.detach().double().cpu()
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError('Expected nonempty [N,6] IMU variances.')
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError('Evaluation requires finite nonnegative covariance predictions.')
    quantiles = values.quantile(torch.tensor([.01,.05,.5,.95,.99],dtype=values.dtype),dim=0)
    result = {'samples':len(values)}
    for name,offset in (('acc_cov',0),('gyro_cov',3)):
        q = values[:,offset:offset+3]
        result[name] = {'mean':q.mean(0).tolist(),'min':q.amin(0).tolist(),'max':q.amax(0).tolist(),
            'sqrt_mean_variance':q.mean(0).sqrt().tolist(),'mean_predicted_std':q.sqrt().mean(0).tolist(),
            **{key:quantiles[index,offset:offset+3].tolist()
               for index,key in enumerate(('p01','p05','median','p95','p99'))}}
    return result


def angle_statistics(radians):
    values = np.degrees(np.asarray(radians,dtype=float))
    if not len(values):return {'samples':0}
    return {'samples':len(values),'rmse_degrees':float(np.sqrt(np.mean(values**2))),
            'mean_degrees':float(values.mean()),'median_degrees':float(np.median(values)),
            'p95_degrees':float(np.quantile(values,.95)),'max_degrees':float(values.max())}


def orientation_metrics(trajectory, flight, evaluation_mask, reference_mask):
    """Geodesic attitude and wrapped heading errors with explicit GT quality masks."""
    estimated = trajectory['quaternion'].detach().cpu().double().numpy()
    time = flight['imu_time'].detach().cpu().double().numpy()
    gt_time = flight['vicon_time_gt'].detach().cpu().double().numpy()
    gt = flight['vicon_quaternion_gt'].detach().cpu().double().numpy()
    good = np.isfinite(gt).all(1) & (np.linalg.norm(np.nan_to_num(gt),axis=1)>1e-12)
    safe = gt.copy();safe[~good] = [1.,0.,0.,0.]
    rotations = Rotation.from_quat(safe[:,[1,2,3,0]])
    interpolated = Slerp(gt_time-gt_time[0],rotations)((time-gt_time[0]).clip(0,gt_time[-1]-gt_time[0]))
    est = Rotation.from_quat(estimated[:,[1,2,3,0]])
    error = (interpolated.inv()*est).magnitude()
    index = np.searchsorted(gt_time,time).clip(0,len(gt_time)-1)
    lower = (index-1).clip(0,len(gt_time)-1)
    valid = (time>=gt_time[0]) & (time<=gt_time[-1]) & good[index] & good[lower]
    scored = evaluation_mask.cpu().numpy().astype(bool)
    filtered = valid & scored & reference_mask.cpu().numpy().astype(bool)
    est_matrix,gt_matrix = est.as_matrix(),interpolated.as_matrix()
    heading_ok = (np.linalg.norm(est_matrix[:,:2,0],axis=1)>1e-6) & (np.linalg.norm(gt_matrix[:,:2,0],axis=1)>1e-6)
    delta = np.arctan2(est_matrix[:,1,0],est_matrix[:,0,0])-np.arctan2(gt_matrix[:,1,0],gt_matrix[:,0,0])
    yaw = np.abs(np.arctan2(np.sin(delta),np.cos(delta)))
    return {'scope':'cutoff+5 to cutoff+25 seconds',
        'geodesic_all_finite_reference':angle_statistics(error[valid & scored]),
        'geodesic_quality_filtered':angle_statistics(error[filtered]),
        'yaw_quality_filtered':angle_statistics(yaw[filtered & heading_ok]),
        'quality_filtered_reference_policy':'same Vicon validity mask as gyro calibration',
        'quality_filtered_squared_error_sum_rad2':float(np.square(error[filtered]).sum()),
        'quality_filtered_yaw_squared_error_sum_rad2':float(np.square(yaw[filtered & heading_ok]).sum())}


def numerical_stability(trajectory, flight):
    finite = {name:bool(torch.isfinite(values).all()) for name,values in trajectory.items()}
    covariance = trajectory['covariance'].detach()
    diagonal = covariance.diagonal(dim1=-2,dim2=-1)
    selected = torch.linspace(0,len(covariance)-1,min(256,len(covariance))).long().to(covariance.device)
    sampled = covariance.index_select(0,selected)
    smallest = float(torch.linalg.eigvalsh((sampled+sampled.transpose(-1,-2))*.5).amin())
    return {'all_trajectory_values_finite':all(finite.values()),'finite_by_field':finite,
        'minimum_state_covariance_diagonal':float(diagonal.amin()),
        'maximum_state_covariance_diagonal':float(diagonal.amax()),
        'maximum_covariance_asymmetry':float((covariance-covariance.transpose(-1,-2)).abs().amax()),
        'sampled_minimum_covariance_eigenvalue':smallest,'covariance_eigenvalue_snapshots':len(selected),
        'maximum_quaternion_norm_error':float((trajectory['quaternion'].norm(dim=-1)-1).abs().amax()),
        'maximum_speed_mps':float(trajectory['velocity'].norm(dim=-1).amax()),
        'maximum_accel_bias_norm':float(trajectory['accelerometer_bias'].norm(dim=-1).amax()),
        'maximum_gyro_bias_norm':float(trajectory['gyroscope_bias'].norm(dim=-1).amax()),
        'filter_initializations':1,'automatic_filter_restarts':0,
        'error_state_reset_policy':'existing measurement-update error injection/reset; no flight restarts',
        'scheduled_measurement_updates':len(flight['vicon_events']),
        'innovation_factorization_retry_count':None,
        'innovation_retry_instrumentation':'not exposed by unchanged EKF; numerical failures raise and abort evaluation'}
