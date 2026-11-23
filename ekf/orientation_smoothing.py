"""Offline calibration-only local polynomial smoothing in SO(3) tangent spaces."""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def smooth_rotations(time, rotations, good_samples, breaks, window_seconds):
    """Fit a Gaussian-weighted quadratic Log(R_i^-1 R_j) near each sample.

    The fitted intercept supplies a rotation correction, rather than averaging
    quaternion components. Actual timestamps preserve constant angular rate and
    locally quadratic angle trajectories. Invalid samples and every flagged
    interval split the fit. Short neighborhoods remain unchanged.

    Return the L2 norm of each intercept's linear weights for a first-order
    orientation-noise bound. This is not a claim to remove systematic Vicon
    errors or model all nonlinear smoothing bias.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('SO(3) smoothing window must be positive and finite.')
    n=len(time)
    if len(rotations)!=n or good_samples.shape!=(n,) or breaks.shape!=(n-1,):
        raise ValueError('Smoothing arrays must be aligned.')
    output=rotations.as_quat().copy();noise_norm=np.ones(n);segments=[]
    start=None
    for index in range(n):
        if not good_samples[index]:
            if start is not None:segments.append((start,index));start=None
            continue
        if start is not None and breaks[index-1]:
            segments.append((start,index));start=None
        if start is None:start=index
    if start is not None:segments.append((start,n))
    changed=0;half=window_seconds/2
    for start,end in segments:
        local_time=time[start:end]
        for index in range(start,end):
            lo=start+np.searchsorted(local_time,time[index]-half-1e-10,side='left')
            hi=start+np.searchsorted(local_time,time[index]+half+1e-10,side='right')
            if hi-lo<3:continue
            u=(time[lo:hi]-time[index])/half
            design=np.column_stack((np.ones(len(u)),u,u*u))
            weights=np.exp(-2*u*u)
            weighted=design*np.sqrt(weights[:,None])
            if np.linalg.matrix_rank(weighted)<3:continue
            coefficients=np.linalg.pinv(weighted)[0]*np.sqrt(weights)
            tangent=(rotations[index].inv()*rotations[lo:hi]).as_rotvec()
            intercept=coefficients@tangent
            output[index]=(rotations[index]*Rotation.from_rotvec(intercept)).as_quat()
            noise_norm[index]=np.linalg.norm(coefficients);changed+=1
    return Rotation.from_quat(output),noise_norm,{'method':'SO3_local_tangent_quadratic_gaussian',
        'window_seconds':window_seconds,'polynomial_order':2,'segments':len(segments),
        'samples_smoothed':changed,'short_neighborhood_samples_unchanged':int(good_samples.sum())-changed,
        'maximum_orientation_adjustment_rad':float((rotations.inv()*Rotation.from_quat(output)).magnitude().max()),
        'maximum_linear_weight_l2':float(noise_norm.max())}


def endpoint_noise_norm(time, norms, query):
    """Triangle bound for the norm of SLERP endpoint's linearized noise weights."""
    upper=np.searchsorted(time,query,side='right').clip(1,len(time)-1);lower=upper-1
    weight=((query-time[lower])/(time[upper]-time[lower])).clip(0,1)
    return (1-weight)*norms[lower]+weight*norms[upper]
