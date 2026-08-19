import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan


def rmse_from_dist(dist):
    return torch.sqrt((dist.norm(dim=-1) ** 2).mean())


def compute_nees(dist, cov, eps=1e-8):
    dim = dist.shape[-1]
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    cov_safe = cov + eps * eye
    solved = torch.linalg.solve(cov_safe, dist.unsqueeze(-1)).squeeze(-1)
    nees = torch.sum(dist * solved, dim=-1)
    return nees.mean()


def compute_sigma_coverage(dist, cov, sigma=1.0, eps=1e-8):
    var = torch.diagonal(cov, dim1=-2, dim2=-1)
    std = torch.sqrt(torch.clamp(var, min=eps))
    return (dist.abs() <= sigma * std).float().mean()


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

    # Apply the covariance loss
    if confs.propcov:
        _, raw_rot_dist = loss_(rotloss_fc, inte_state['raw_rot'], data['gt_rot'], sampling = confs.sampling, dtype='rot')
        _, raw_vel_dist = loss_(loss_fc, inte_state['raw_vel'], data['gt_vel'], sampling = confs.sampling)
        _, raw_pos_dist = loss_(loss_fc, inte_state['raw_pos'], data['gt_pos'], sampling = confs.sampling)

        cov_diag = torch.diagonal(inte_state['cov'], dim1=-2, dim2=-1)
        cov_losses['raw_vel_rmse'] = rmse_from_dist(raw_vel_dist).detach()

        vel_cov = inte_state['cov'][..., 3:6, 3:6]
        cov_losses['vel_nees'] = compute_nees(raw_vel_dist.detach(), vel_cov).detach()
        cov_losses['vel_1sigma'] = compute_sigma_coverage(
            raw_vel_dist.detach(), vel_cov, sigma=1.0
        ).detach()
        cov_losses['vel_2sigma'] = compute_sigma_coverage(
            raw_vel_dist.detach(), vel_cov, sigma=2.0
        ).detach()

        cov_losses['pred_cov_rot'] = cov_diag[...,:3].mean()
        cov_losses['pred_cov_vel'] = cov_diag[...,3:6].mean()
        cov_losses['pred_cov_pos'] = cov_diag[...,-3:].mean()

        if "covaug" in confs and confs["covaug"] is True:
            cov_rot_loss = diag_ln_cov_loss(raw_rot_dist, cov_diag[...,:3])
            cov_vel_loss = diag_ln_cov_loss(raw_vel_dist, cov_diag[...,3:6])
            cov_pos_loss = diag_ln_cov_loss(raw_pos_dist, cov_diag[...,-3:])
        else:
            cov_rot_loss = diag_ln_cov_loss(raw_rot_dist.detach(), cov_diag[...,:3])
            cov_vel_loss = diag_ln_cov_loss(raw_vel_dist.detach(), cov_diag[...,3:6])
            cov_pos_loss = diag_ln_cov_loss(raw_pos_dist.detach(), cov_diag[...,-3:])
        # cov losses added directly so they are scaled only by cov_weight, not by state weights
        cov_losses['rot_nll'] = cov_rot_loss.detach()
        cov_losses['vel_nll'] = cov_vel_loss.detach()
        cov_losses['pos_nll'] = cov_pos_loss.detach()
        cov_losses['cov_nll'] = (cov_rot_loss + cov_vel_loss + cov_pos_loss).detach()

    loss += (confs.pos_weight * pos_loss + confs.rot_weight * rot_loss + confs.vel_weight * vel_loss)
    if confs.propcov:
        loss += confs.cov_weight * (cov_rot_loss + cov_vel_loss + cov_pos_loss)
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
