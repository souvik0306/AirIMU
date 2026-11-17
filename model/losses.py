import torch
from .loss_func import loss_fc_list
from utils import report_hasNan


def rmse_from_dist(dist):
    return torch.sqrt((dist.norm(dim=-1) ** 2).mean())


def _covariance_with_time_axis(cov, dist):
    """Give a final-state covariance the same explicit time axis as an error."""
    if cov.ndim == dist.ndim:
        cov = cov.unsqueeze(-3)
    return cov


def _stable_cholesky(cov, eps=1e-6):
    """Factor a covariance after symmetrization and scale-aware jitter."""
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    dim = cov.shape[-1]
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).abs().mean(dim=-1)
    scale = scale.clamp_min(torch.finfo(cov.dtype).tiny)
    jitter = eps * scale
    chol, info = torch.linalg.cholesky_ex(cov + jitter[..., None, None] * eye)

    # A propagated covariance can acquire a tiny negative eigenvalue from
    # roundoff. Increase jitter only for those samples instead of applying a
    # large global floor that would distort the calibrated covariance.
    if torch.any(info != 0):
        min_eig = torch.linalg.eigvalsh(cov.detach()).amin(dim=-1)
        adaptive_jitter = torch.clamp(-min_eig + 10.0 * jitter, min=jitter)
        chol = torch.linalg.cholesky(
            cov + adaptive_jitter[..., None, None] * eye
        )
    return chol


def _mahalanobis(dist, cov):
    cov = _covariance_with_time_axis(cov, dist)
    chol = _stable_cholesky(cov)
    whitened = torch.linalg.solve_triangular(
        chol, dist.unsqueeze(-1), upper=False
    ).squeeze(-1)
    return whitened.square().sum(dim=-1), chol


def cholesky_nll(dist, cov):
    """Gaussian NLL without the constant term, using a Cholesky solve."""
    mahalanobis, chol = _mahalanobis(dist, cov)
    logdet = 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)
    ).sum(dim=-1)
    return 0.5 * (mahalanobis + logdet).mean()


def compute_cholesky_nees(dist, cov):
    nees, _ = _mahalanobis(dist, cov)
    return nees.mean()


def compute_nees(dist, cov, eps=1e-8):
    """Existing Stage 2 NEES baseline, kept unchanged."""
    dim = dist.shape[-1]
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    cov_safe = cov + eps * eye
    solved = torch.linalg.solve(cov_safe, dist.unsqueeze(-1)).squeeze(-1)
    nees = torch.sum(dist * solved, dim=-1)
    return nees.mean()


def compute_sigma_coverage(dist, cov, sigma=1.0, eps=1e-8):
    cov = _covariance_with_time_axis(cov, dist)
    var = torch.diagonal(cov, dim1=-2, dim2=-1)
    std = torch.sqrt(torch.clamp(var, min=eps))
    return (dist.abs() <= sigma * std).float().mean()


def covariance_rotation_residual(pred, targ, sampling=None):
    """Local SO(3) residual matching the propagated rotation covariance."""
    if sampling:
        pred = pred[:, sampling - 1::sampling, :]
        targ = targ[:, sampling - 1::sampling, :]
    else:
        pred = pred[:, -1:, :]
        targ = targ[:, -1:, :]
    return (pred.Inv() * targ).Log()


def loss_(fc, pred, targ, sampling = None, dtype = 'trans'):
    ## reshape or sample the input targ and pred
    ## cov and error is for reference
    if sampling:
        pred = pred[:,sampling-1::sampling,:]
        targ = targ[:,sampling-1::sampling,:]
    else:
        pred = pred[:,-1:,:]
        targ = targ[:,-1:,:]

    if dtype == 'rot':
        dist = (pred * targ.Inv()).Log()
    else:
        dist = pred - targ
    loss = fc(dist)
    return loss, dist


def get_loss(inte_state, data, confs):
    ## The state loss for evaluation
    loss, state_losses, cov_losses = 0, {}, {}
    loss_fc = loss_fc_list[confs.loss]
    rotloss_fc = loss_fc_list[confs.rotloss]
    
    rot_loss, rot_dist = loss_(rotloss_fc, inte_state['rot'], data['gt_rot'], sampling = confs.sampling, dtype='rot')
    vel_loss, vel_dist = loss_(loss_fc, inte_state['vel'], data['gt_vel'], sampling = confs.sampling)
    pos_loss, pos_dist = loss_(loss_fc, inte_state['pos'], data['gt_pos'], sampling = confs.sampling)

    state_losses['pos'] = pos_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['rot'] = rot_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['vel'] = vel_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['corrected_pos_rmse'] = rmse_from_dist(pos_dist).detach()
    state_losses['corrected_rot_rmse'] = rmse_from_dist(rot_dist).detach()
    state_losses['corrected_vel_rmse'] = rmse_from_dist(vel_dist).detach()

    # Apply the covariance loss
    if confs.propcov:
        is_stage2 = confs.get('stage', 1) == 2
        if is_stage2:
            cov_rot = inte_state['raw_rot']
            cov_vel = inte_state['raw_vel']
            cov_pos = inte_state['raw_pos']
        else:
            cov_rot = inte_state['rot']
            cov_vel = inte_state['vel']
            cov_pos = inte_state['pos']

        # Rotation uncertainty is in the local tangent frame for both stages.
        # The corrected-trajectory rotation Huber loss above is independent.
        cov_rot_dist = covariance_rotation_residual(
            cov_rot, data['gt_rot'], sampling=confs.sampling
        )
        _, cov_vel_dist = loss_(
            loss_fc, cov_vel, data['gt_vel'], sampling=confs.sampling
        )
        _, cov_pos_dist = loss_(
            loss_fc, cov_pos, data['gt_pos'], sampling=confs.sampling
        )

        rot_cov = inte_state['cov'][..., :3, :3]
        vel_cov = inte_state['cov'][..., 3:6, 3:6]
        pos_cov = inte_state['cov'][..., 6:9, 6:9]
        cov_diag = torch.diagonal(inte_state['cov'], dim1=-2, dim2=-1)
        cov_losses['pred_cov_rot'] = cov_diag[..., :3].mean()
        cov_losses['pred_cov_vel'] = cov_diag[..., 3:6].mean()
        cov_losses['pred_cov_pos'] = cov_diag[..., 6:9].mean()

        nll_rot_dist = cov_rot_dist if confs.get('covaug', False) else cov_rot_dist.detach()
        nll_vel_dist = cov_vel_dist if confs.get('covaug', False) else cov_vel_dist.detach()
        nll_pos_dist = cov_pos_dist if confs.get('covaug', False) else cov_pos_dist.detach()

        # Cross-correlations require a consistent truth-minus-estimate sign
        # convention, ordered as [local rotation, velocity, position].
        full_dist = torch.cat(
            [nll_rot_dist, -nll_vel_dist, -nll_pos_dist], dim=-1
        )
        combined_cov_nll = cholesky_nll(full_dist, inte_state['cov'])

        # Marginal NLLs are diagnostics; only the joint 9x9 NLL is optimized.
        with torch.no_grad():
            rot_nll = cholesky_nll(cov_rot_dist.detach(), rot_cov.detach())
            vel_nll = cholesky_nll(cov_vel_dist.detach(), vel_cov.detach())
            pos_nll = cholesky_nll(cov_pos_dist.detach(), pos_cov.detach())

        if is_stage2:
            cov_losses['rot_nll'] = rot_nll
            cov_losses['vel_nll'] = vel_nll
            cov_losses['cov_nll'] = combined_cov_nll.detach()
            cov_losses['pos_nll'] = pos_nll
            cov_losses['raw_vel_rmse'] = rmse_from_dist(cov_vel_dist).detach()
            rot_calibration_dist = cov_rot_dist.detach()
            cov_losses['rot_nees'] = compute_cholesky_nees(
                rot_calibration_dist, rot_cov.detach(),
            ).detach()
            cov_losses['rot_1sigma_coverage'] = compute_sigma_coverage(
                rot_calibration_dist, rot_cov.detach(), sigma=1.0,
                eps=torch.finfo(rot_cov.dtype).tiny,
            ).detach()
            cov_losses['vel_nees'] = compute_nees(
                cov_vel_dist.detach(), vel_cov
            ).detach()
            cov_losses['vel_1sigma_coverage'] = compute_sigma_coverage(
                cov_vel_dist.detach(), vel_cov, sigma=1.0
            ).detach()
            cov_losses['vel_2sigma_coverage'] = compute_sigma_coverage(
                cov_vel_dist.detach(), vel_cov, sigma=2.0
            ).detach()
        else:
            cov_losses['corrected_rot_nll'] = rot_nll
            cov_losses['corrected_vel_nll'] = vel_nll
            cov_losses['corrected_pos_nll'] = pos_nll
            cov_losses['corrected_cov_nll'] = combined_cov_nll.detach()
            cov_losses['corrected_vel_nees'] = compute_cholesky_nees(
                cov_vel_dist.detach(), vel_cov
            ).detach()
            cov_losses['corrected_vel_1sigma'] = compute_sigma_coverage(
                cov_vel_dist.detach(), vel_cov, sigma=1.0
            ).detach()
            cov_losses['corrected_vel_2sigma'] = compute_sigma_coverage(
                cov_vel_dist.detach(), vel_cov, sigma=2.0
            ).detach()

        # Optional direct pressure on the per-step acc/gyro variance,
        # independent of the long-horizon integrated NLL, so the model can't
        # settle for a merely "legitimate" but inflated covariance driven by
        # multi-step drift rather than sensor noise.
        # Normalized by the desired target (not the hard ceiling) and split
        # per-sensor since drone performance is more gyro-sensitive.
        cov_losses['acc_cov_mean'] = inte_state['acc_cov'].mean().detach()
        cov_losses['gyro_cov_mean'] = inte_state['gyro_cov'].mean().detach()
        acc_cov_reg_weight = confs.get('acc_cov_reg_weight', confs.get('cov_reg_weight', 0.0))
        gyro_cov_reg_weight = confs.get('gyro_cov_reg_weight', confs.get('cov_reg_weight', 0.0))
        if acc_cov_reg_weight > 0 or gyro_cov_reg_weight > 0:
            acc_cov_target = confs.get('acc_cov_target', 0.07 ** 2)
            gyro_cov_target = confs.get('gyro_cov_target', 0.003 ** 2)
            acc_cov_reg_loss = (inte_state['acc_cov'] / acc_cov_target).mean()
            gyro_cov_reg_loss = (inte_state['gyro_cov'] / gyro_cov_target).mean()
            cov_losses['acc_cov_reg'] = acc_cov_reg_loss.detach()
            cov_losses['gyro_cov_reg'] = gyro_cov_reg_loss.detach()

    loss += (confs.pos_weight * pos_loss + confs.rot_weight * rot_loss + confs.vel_weight * vel_loss)
    if confs.propcov:
        loss += confs.cov_weight * combined_cov_nll
        if acc_cov_reg_weight > 0:
            loss += acc_cov_reg_weight * acc_cov_reg_loss
        if gyro_cov_reg_weight > 0:
            loss += gyro_cov_reg_weight * gyro_cov_reg_loss
    # report_hasNan(loss)

    return {'loss':loss, **state_losses, **cov_losses}


def get_RMSE(inte_state, data):
    '''
    get the RMSE of the last state in one segment
    '''
    dist_pos = (inte_state['pos'][:,-1,:] - data['gt_pos'][:,-1,:])
    dist_vel = (inte_state['vel'][:,-1,:] - data['gt_vel'][:,-1,:])
    dist_rot = (data['gt_rot'][:,-1,:] * inte_state['rot'][:,-1,:].Inv()).Log()

    pos_loss = rmse_from_dist(dist_pos)[None,...]
    vel_loss = rmse_from_dist(dist_vel)[None,...]
    rot_loss = rmse_from_dist(dist_rot)[None,...]

    ## Relative pos error
    return {'pos': pos_loss, 'rot': rot_loss, 'vel': vel_loss, 
            'pos_dist': dist_pos.norm(dim=-1).mean(),
            'vel_dist': dist_vel.norm(dim=-1).mean(),
            'rot_dist': dist_rot.norm(dim=-1).mean(),}
