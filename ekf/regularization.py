"""Supervision and diagnostics only; none of these values are AirIMU inputs."""

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


def gyro_reference(time, quaternion_wxyz, *, orientation_time=None, window_seconds=.05, orientation_std=.001,
                   max_step_degrees=45., max_angular_rate=20., smoothing_window_seconds=0.):
    """Body rates from aligned Hamilton body-to-nav rotations.

    A centered SO(3) secant suppresses single-frame differentiation noise.
    Transport the secant from its left body frame to the sample's body frame.
    Independent isotropic orientation noise contributes 2*sigma_R**2/dt**2
    to the reference rate variance, added to q in the likelihood. Boundary
    samples without a full window and gaps larger than half the window are masked.
    Every window touching an invalid quaternion or an implausible adjacent SO(3)
    step/rate is also rejected. These limits validate Vicon references; they do
    not constrain learned covariance or use the learned predictions for masking.
    The noise assumption and window must be tuned on validation flights.
    """
    t = torch.as_tensor(time).detach().cpu().double().numpy()
    q = torch.as_tensor(quaternion_wxyz).detach().cpu().double().numpy()
    if (len(t) < 3 or q.shape != (len(t), 4) or not np.isfinite(t).all()
            or (np.diff(t) <= 0).any()):
        raise ValueError("Calibration requires finite aligned rotations and increasing timestamps.")
    if not math.isfinite(window_seconds) or window_seconds <= 0 or not math.isfinite(orientation_std) or orientation_std < 0:
        raise ValueError("Use a positive calibration window and nonnegative orientation std.")
    if (not math.isfinite(max_step_degrees) or not 0 < max_step_degrees <= 180
            or not math.isfinite(max_angular_rate) or max_angular_rate <= 0):
        raise ValueError("Use a reference step limit in (0, 180] degrees and a positive rate limit.")
    gt = t.copy() if orientation_time is None else torch.as_tensor(orientation_time).detach().cpu().double().numpy()
    if gt.shape != t.shape or not np.isfinite(gt).all() or (np.diff(gt) <= 0).any():
        raise ValueError("Vicon timestamps must be aligned and strictly increasing.")
    origin = t[0]
    t, gt = t - origin, gt - origin
    left_time, right_time = t - window_seconds/2, t + window_seconds/2
    base_mask = (left_time >= gt[0]) & (right_time <= gt[-1])
    left = (np.searchsorted(gt, left_time, side="right") - 1).clip(0, len(gt)-1)
    right = np.searchsorted(gt, right_time, side="left").clip(0, len(gt)-1)
    gap = np.diff(gt)
    finite_quaternion = np.isfinite(q).all(axis=1)
    norm = np.zeros(len(q))
    norm[finite_quaternion] = np.linalg.norm(q[finite_quaternion],axis=1)
    good_quaternion = finite_quaternion & np.isfinite(norm) & (norm > 1e-12)
    safe_q = q.copy()
    safe_q[~good_quaternion] = [1., 0., 0., 0.]
    rotations = Rotation.from_quat(safe_q[:, [1, 2, 3, 0]])
    step = (rotations[:-1].inv()*rotations[1:]).magnitude()
    rate_step = step/gap
    good_pair = good_quaternion[:-1] & good_quaternion[1:]

    def touching(intervals):
        prefix = np.r_[0, np.cumsum(intervals, dtype=np.int64)]
        return (prefix[right]-prefix[left]) > 0

    bad_gap = touching(gap > window_seconds/2)
    bad_quaternion = touching(~good_pair)
    bad_step = touching(good_pair & (step > math.radians(max_step_degrees)))
    bad_rate = touching(good_pair & (rate_step > max_angular_rate))
    valid = base_mask & ~bad_gap & ~bad_quaternion & ~bad_step & ~bad_rate
    if not math.isfinite(smoothing_window_seconds) or smoothing_window_seconds < 0:
        raise ValueError('Smoothing window must be finite and nonnegative.')
    original_rotations=rotations
    smoothing=None;noise_norm=None
    if smoothing_window_seconds:
        from .orientation_smoothing import smooth_rotations
        breaks=(gap>window_seconds/2) | ~good_pair | (step>math.radians(max_step_degrees)) | (rate_step>max_angular_rate)
        rotations,noise_norm,smoothing=smooth_rotations(gt,rotations,good_quaternion,breaks,smoothing_window_seconds)
    # Exact symmetric endpoint times avoid a half-sample timing bias in
    # angular acceleration, even with irregular spacing or roundoff at 200 Hz.
    interpolate = Slerp(gt, rotations)
    left_rotation = interpolate(left_time.clip(gt[0], gt[-1]))
    right_rotation = interpolate(right_time.clip(gt[0], gt[-1]))
    center_rotation = interpolate(t.clip(gt[0], gt[-1]))
    secant = (left_rotation.inv() * right_rotation).as_rotvec() / window_seconds
    rate = (center_rotation.inv() * left_rotation).apply(secant)
    noise = np.full(rate.shape, 2*orientation_std**2 / window_seconds**2)
    original_rate=None
    if smoothing is not None:
        from .orientation_smoothing import endpoint_noise_norm
        original=Slerp(gt,original_rotations)
        old_left=original(left_time.clip(gt[0],gt[-1]));old_right=original(right_time.clip(gt[0],gt[-1]))
        old_center=original(t.clip(gt[0],gt[-1]))
        original_rate=(old_center.inv()*old_left).apply((old_left.inv()*old_right).as_rotvec()/window_seconds)
        # Correlated filtered endpoint noise has Var(e_r-e_l) bounded by
        # sigma_R^2*(||w_r||_2+||w_l||_2)^2 in a local linearization. Keep the
        # original 2*sigma_R^2/window^2 as a conservative floor; do not credit
        # smoothing with eliminating reference error. Bias is not covered.
        left_norm=endpoint_noise_norm(gt,noise_norm,left_time.clip(gt[0],gt[-1]))
        right_norm=endpoint_noise_norm(gt,noise_norm,right_time.clip(gt[0],gt[-1]))
        factor=np.maximum(2.,(left_norm+right_norm)**2)
        noise[:]=((orientation_std/window_seconds)**2*factor)[:,None]
        smoothing.update(reference_noise_policy='max(original endpoint-noise floor, linearized correlated-endpoint bound)',
                         systematic_orientation_error_removed=False,
                         reference_variance_min=float(noise.min()),reference_variance_max=float(noise.max()))
    quality = {"policy": "vicon_so3_quality_v1", "samples": len(t),
               "valid_samples": int(valid.sum()), "invalid_samples": int((~valid).sum()),
               "boundary_windows": int((~base_mask).sum()),
               "gap_windows": int((base_mask & bad_gap).sum()),
               "invalid_quaternion_windows": int((base_mask & bad_quaternion).sum()),
               "orientation_jump_windows": int((base_mask & bad_step).sum()),
               "angular_rate_windows": int((base_mask & bad_rate).sum()),
               "invalid_quaternions": int((~good_quaternion).sum()),
               "orientation_jump_intervals": int((good_pair & (step > math.radians(max_step_degrees))).sum()),
               "angular_rate_intervals": int((good_pair & (rate_step > max_angular_rate)).sum()),
               "max_step_degrees": float(max_step_degrees), "max_angular_rate_rad_s": float(max_angular_rate),
               "largest_input_step_degrees": float(np.degrees(step[good_pair].max())) if good_pair.any() else None,
               "largest_valid_reference_rate_rad_s": float(np.linalg.norm(
                   (original_rate if original_rate is not None else rate)[valid], axis=1).max()) if valid.any() else None}
    # Masking is authoritative; finite placeholders prevent invalid references
    # accidentally propagating NaNs through arithmetic before boolean selection.
    rate[~valid] = 0.
    result={"rate": torch.from_numpy(rate), "variance": torch.from_numpy(noise),
            "mask": torch.from_numpy(valid), "quality": quality}
    if smoothing is not None:
        original_rate[~valid]=0.
        result.update(original_rate=torch.from_numpy(original_rate),
            original_variance=torch.full_like(result['variance'],2*orientation_std**2/window_seconds**2),
            smoothing=smoothing)
    return result


@torch.no_grad()
def compare_gyro_references(variance, raw_gyro, bias, reference):
    """Paired residual statistics on the exact same raw-reference validity mask."""
    if 'original_rate' not in reference:return None
    valid=reference['mask'].cpu();q=variance.detach().cpu().double()[:,3:]
    measured=raw_gyro.detach().cpu().double()-bias.detach().cpu().double()
    result={'valid_samples':int(valid.sum()),'rejected_samples':int((~valid).sum()),
            'same_raw_quality_mask':True,'smoothing':reference['smoothing']}
    for name,rate,noise in (('original',reference['original_rate'],reference['original_variance']),
                           ('filtered',reference['rate'],reference['variance'])):
        residual=(measured-rate)[valid];effective=(q+noise)[valid]
        if not len(residual):result[name]={'samples':0};continue
        absolute=residual.abs();terms=residual.square()/effective+effective.log()
        result[name]={'samples':len(residual),'residual_rms_xyz':residual.square().mean(0).sqrt().tolist(),
            'residual_abs_p95_xyz':absolute.quantile(.95,dim=0).tolist(),
            'residual_abs_max_xyz':absolute.amax(0).tolist(),
            'one_sigma_coverage_xyz':(absolute<=effective.sqrt()).double().mean(0).tolist(),
            'calibration_loss':float(terms.mean())}
    if bool(valid.any()):
        delta=(reference['rate']-reference['original_rate'])[valid]
        result['reference_change_rms_xyz']=delta.square().mean(0).sqrt().tolist()
        result['reference_change_max_xyz']=delta.abs().amax(0).tolist()
    return result


def gyro_calibration_loss(variance, raw_gyro, bias, reference):
    """Gaussian variance regularizer with explicit Vicon differentiation noise."""
    device, dtype = variance.device, variance.dtype
    rate = reference["rate"].to(device=device, dtype=dtype)
    noise = reference["variance"].to(device=device, dtype=dtype)
    mask = reference["mask"].to(device)
    if not bool(mask.any()):
        raise ValueError("No valid Vicon angular-rate supervision windows.")
    residual = raw_gyro.to(variance) - bias.detach().to(variance) - rate
    q = variance[:, 3:].clamp_min(torch.finfo(dtype).tiny) + noise
    terms = residual.square()/q + q.log()
    return terms[mask].mean(), residual, q, mask


def log_variance_smoothness(variance, time, *, nominal_dt=.005):
    """Time-average squared log-rate change, scaled to a nominal IMU step.

    At uniform nominal spacing this is mean_k sum_axis(delta log q)**2.
    Integral weighting gives a consistent value for irregular sample spacing.
    Called once per flight, so unrelated sequences are never connected.
    """
    if len(variance) < 2:
        return variance.sum()*0
    dt = torch.as_tensor(time, device=variance.device, dtype=torch.float64).diff()
    if not bool(torch.isfinite(dt).all()) or bool((dt <= 0).any()):
        raise ValueError("Smoothness requires increasing finite timestamps.")
    delta = variance[:, 3:].clamp_min(torch.finfo(variance.dtype).tiny).log().diff(dim=0)
    return (delta.square().sum(-1)/dt).sum() * nominal_dt**2 / dt.sum()


@torch.no_grad()
def gyro_diagnostics(variance, time, raw_gyro, *, floor, residual=None, effective_q=None, mask=None):
    q = variance.detach().double().cpu()[:, 3:]
    logs = q.clamp_min(torch.finfo(q.dtype).tiny).log()
    change = logs.diff(dim=0).abs()
    dt = torch.as_tensor(time).cpu().double().diff()
    raw = raw_gyro.detach().cpu().double()
    rate = raw.norm(dim=-1)
    acceleration = torch.cat((torch.zeros(1), raw.diff(dim=0).norm(dim=-1)/dt))
    result = {"mean": q.mean(0).tolist(), "median": q.quantile(.5, dim=0).tolist(),
              "p95": q.quantile(.95, dim=0).tolist(), "max": q.amax(0).tolist(),
              "log_change_mean": change.mean(0).tolist(),
              "log_change_p95": change.quantile(.95, dim=0).tolist(),
              "near_floor_fraction": (q <= float(floor)*1.01).double().mean(0).tolist()}
    # Fixed SI thresholds make categories comparable between flights/runs.
    conditions = {"quiet": (rate < .1) & (acceleration < 1.),
                  "rapid": (rate >= 1.) | (acceleration >= 10.)}
    conditions["moderate"] = ~(conditions["quiet"] | conditions["rapid"])
    result["motion"] = {}
    for name, select in conditions.items():
        summary = {"samples": int(select.sum())}
        if bool(select.any()):
            summary.update(mean=q[select].mean(0).tolist(), median=q[select].quantile(.5, dim=0).tolist(),
                           p95=q[select].quantile(.95, dim=0).tolist(), max=q[select].amax(0).tolist())
        result["motion"][name] = summary
    if residual is not None:
        r, eq, valid = residual.detach().cpu(), effective_q.detach().cpu(), mask.cpu()
        result["normalized_residual_square"] = (r[valid].square()/eq[valid]).mean(0).tolist()
        result["one_sigma_coverage"] = (r[valid].abs() <= eq[valid].sqrt()).double().mean(0).tolist()
    return result
