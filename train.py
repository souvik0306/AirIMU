import os
import torch
import numpy as np

import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import tqdm, wandb
from utils import move_to
from utils.output_metrics import OutputMetrics
from model import net_dict
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert

from datasets import SeqeuncesDataset, collate_fcs
from model.losses import get_loss
from eval import evaluate


WEIGHTED_LOSS_KEYS = ('loss_corrected', 'loss_covariance', 'loss_variance_reg')
ENCODER_GRADIENT_KEYS = ('encoder_grad_norm_corrected', 'encoder_grad_norm_covariance')
STAGE2_CALIBRATION_KEYS = (
    'rot_nees', 'rot_1sigma_coverage', 'vel_nees',
    'vel_1sigma_coverage', 'vel_2sigma_coverage',
)
WANDB_PHASE_METRIC_KEYS = frozenset({
    'loss', *WEIGHTED_LOSS_KEYS, *ENCODER_GRADIENT_KEYS,
    'corrected_pos_rmse', 'corrected_rot_rmse', 'corrected_vel_rmse', 'raw_vel_rmse',
    'rot_nll', 'vel_nll', 'cov_nll',
    'corrected_rot_nll', 'corrected_vel_nll', 'corrected_cov_nll',
    *STAGE2_CALIBRATION_KEYS,
    # Stage 1 uses these names for corrected-trajectory calibration.
    'corrected_vel_nees', 'corrected_vel_1sigma', 'corrected_vel_2sigma',
    *(f'outputs/{sensor}_noise_std/mean/{axis}'
      for sensor in ('gyro', 'acc') for axis in 'xyz'),
})


def shared_encoder_gradient_norm(loss, parameters):
    """Global L2 norm from one weighted loss, without accumulating .grad."""
    if not parameters or not loss.requires_grad:
        return loss.detach().new_zeros(())
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    norms = [torch.linalg.vector_norm(gradient.detach().double())
             for gradient in gradients if gradient is not None]
    if not norms:
        return loss.detach().new_zeros(())
    return torch.linalg.vector_norm(torch.stack(norms))


def print_weighted_loss_contributions(context, values):
    tqdm.tqdm.write(
        f"{context} weighted loss contributions: "
        f"loss_corrected={values['loss_corrected']:.6e}, "
        f"loss_covariance={values['loss_covariance']:.6e}, "
        f"loss_variance_reg={values['loss_variance_reg']:.6e}, "
        f"loss={values['loss']:.6e}"
    )


def print_stage2_calibration_metrics(context, metrics):
    values = ', '.join(f'{key}={metrics[key]:.6f}' for key in STAGE2_CALIBRATION_KEYS)
    tqdm.tqdm.write(
        f"{context} covariance calibration: "
        f"rot_nll={metrics['rot_nll']:.6e}, vel_nll={metrics['vel_nll']:.6e}, "
        f"cov_nll={metrics['cov_nll']:.6e} (9x9), {values}"
    )


class _LossMetrics:
    """Keep detached loss totals on the device until a reporting interval."""

    def __init__(self, propcov, uncertainty_keys=()):
        self.sources = {
            'loss': 'loss',
            'corrected_pos_rmse': 'corrected_pos_rmse',
            'corrected_rot_rmse': 'corrected_rot_rmse',
            'corrected_vel_rmse': 'corrected_vel_rmse',
            **{key: key for key in WEIGHTED_LOSS_KEYS},
        }
        if propcov:
            self.sources.update({key: key for key in uncertainty_keys})
        self.total = None
        self.count = 0

    def update(self, loss_state):
        values = torch.stack([
            loss_state[source].detach().reshape(()) for source in self.sources.values()
        ]).double()
        if self.total is None:
            self.total = values
        else:
            self.total.add_(values)
        self.count += 1

    def averages(self):
        values = (self.total / self.count).cpu().tolist()
        return dict(zip(self.sources, values))


def train(network, loader, confs, epoch, optimizer):
    """Train for one epoch, transferring scalar metrics only when reporting."""
    network.train()
    if confs.get('raw_cov_adapter', False) and confs.get('stage', 1) == 2:
        for name in ('cnn', 'gru1', 'gru2', 'accdecoder', 'gyrodecoder'):
            getattr(network, name).eval()
    if confs.get('stage', 1) == 2:
        uncertainty_keys = (
            'raw_vel_rmse', 'rot_nll', 'vel_nll', 'cov_nll',
            *STAGE2_CALIBRATION_KEYS,
        )
    else:
        uncertainty_keys = (
            'corrected_rot_nll', 'corrected_vel_nll',
            'corrected_cov_nll', 'corrected_vel_nees',
            'corrected_vel_1sigma', 'corrected_vel_2sigma',
        )
    metrics = _LossMetrics(confs.propcov, uncertainty_keys)
    output_metrics = OutputMetrics()
    log_interval = max(1, int(confs.get('log_interval', 10)))
    is_stage2 = confs.get('stage', 1) == 2
    encoder_parameters = tuple(
        parameter
        for name in ('cnn', 'gru1', 'gru2')
        for parameter in getattr(network, name).parameters()
        if parameter.requires_grad
    ) if is_stage2 else ()
    gradient_totals = [0.0, 0.0]
    gradient_count = 0

    t_range = tqdm.tqdm(loader)
    # Anomaly tracing is useful for debugging, but expensive on every step.
    with torch.autograd.set_detect_anomaly(bool(confs.get('detect_anomaly', False))):
        for i, (data, init_state, label) in enumerate(t_range):
            data, init_state, label = move_to(
                [data, init_state, label], confs.device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            inte_state = network(data, init_state)
            loss_state = get_loss(inte_state, label, confs)
            output_metrics.update(inte_state)
            metrics.update(loss_state)
            should_report = (i + 1) % log_interval == 0 or i + 1 == len(loader)
            if is_stage2 and should_report:
                # Use the same forward pass for both weighted objectives.
                # autograd.grad leaves .grad untouched, and retain_graph keeps
                # the complete objective available for the normal backward.
                corrected_gradient_norm = shared_encoder_gradient_norm(
                    loss_state['loss_corrected'], encoder_parameters,
                )
                covariance_gradient_norm = shared_encoder_gradient_norm(
                    loss_state['loss_covariance'], encoder_parameters,
                )
            loss_state['loss'].backward()
            if confs.get('raw_cov_adapter', False):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in network.parameters() if p.requires_grad],
                    confs.get('gradient_clip', 1.), error_if_nonfinite=True,
                )
            optimizer.step()

            if should_report:
                current = metrics.averages()
                description = (
                    f"training epoch: {epoch:03d}, loss: {current['loss']:.6f}, RMSE: "
                    f"position {current['corrected_pos_rmse']:.6f}, "
                    f"rotation {current['corrected_rot_rmse']:.6f}, "
                    f"velocity {current['corrected_vel_rmse']:.6f}"
                )
                t_range.set_description(description, refresh=False)
                if is_stage2:
                    context = f"TRAIN Stage 2 epoch={epoch + 1} batch={i + 1}/{len(loader)}"
                    batch_losses = dict(zip(
                        (*WEIGHTED_LOSS_KEYS, 'loss'),
                        torch.stack([loss_state[key].detach() for key in
                                     (*WEIGHTED_LOSS_KEYS, 'loss')]).cpu().tolist(),
                    ))
                    print_weighted_loss_contributions(f"{context} (current batch)", batch_losses)
                    grad_corrected, grad_covariance = torch.stack((
                        corrected_gradient_norm, covariance_gradient_norm,
                    )).cpu().tolist()
                    gradient_totals[0] += grad_corrected
                    gradient_totals[1] += grad_covariance
                    gradient_count += 1
                    tqdm.tqdm.write(
                        f"{context} shared encoder gradient L2 norms "
                        f"(CNN+GRU1+GRU2, current batch, before optimizer step): "
                        f"encoder_grad_norm_corrected={grad_corrected:.6e}, "
                        f"encoder_grad_norm_covariance={grad_covariance:.6e}"
                    )
                    print_stage2_calibration_metrics(f'{context} (running mean)', current)

    result = metrics.averages()
    if gradient_count:
        # Average norms over the batches measured at log_interval and the
        # final batch. Validation does not run backward or measure gradients.
        result.update(zip(ENCODER_GRADIENT_KEYS, (
            total / gradient_count for total in gradient_totals
        )))
    result.update(output_metrics.summarize())
    print_metric_summary('train', result, epoch + 1)
    return result


def test(network, loader, confs, epoch=None):
    network.eval()
    output_metrics = OutputMetrics()
    if confs.get('stage', 1) == 2:
        uncertainty_keys = (
            'raw_vel_rmse', 'rot_nll', 'vel_nll', 'cov_nll',
            *STAGE2_CALIBRATION_KEYS,
        )
    else:
        uncertainty_keys = (
            'corrected_rot_nll', 'corrected_vel_nll',
            'corrected_cov_nll', 'corrected_vel_nees',
            'corrected_vel_1sigma', 'corrected_vel_2sigma',
        )
    metrics = _LossMetrics(confs.propcov, uncertainty_keys)
    log_interval = max(1, int(confs.get('log_interval', 10)))

    with torch.no_grad():
        t_range = tqdm.tqdm(loader)
        for i, (data, init_state, label) in enumerate(t_range):
            data, init_state, label = move_to(
                [data, init_state, label], confs.device, non_blocking=True
            )
            inte_state = network(data, init_state)
            loss_state = get_loss(inte_state, label, confs)
            output_metrics.update(inte_state)
            metrics.update(loss_state)

            if (i + 1) % log_interval == 0 or i + 1 == len(loader):
                current = metrics.averages()
                t_range.set_description(
                    f"testing loss: {current['loss']:.6f}, RMSE: "
                    f"position {current['corrected_pos_rmse']:.6f}, "
                    f"rotation {current['corrected_rot_rmse']:.6f}, "
                    f"velocity {current['corrected_vel_rmse']:.6f}", refresh=False,
                )
                if confs.get('stage', 1) == 2:
                    print_stage2_calibration_metrics(
                        f'VALIDATION Stage 2 batch={i + 1}/{len(loader)} (running mean)', current,
                    )

    result = metrics.averages()
    result.update(output_metrics.summarize())
    print_metric_summary('test', result, epoch)
    return result


def scalar_metric_payload(header, objs):
    """Use one scalar schema for terminal summaries and W&B payloads."""
    if isinstance(objs, dict):
        # Guard the compact phase schema if future callers add diagnostics.
        if header in ('train', 'test', 'tests', 'validation'):
            objs = {key: value for key, value in objs.items() if key in WANDB_PHASE_METRIC_KEYS}
        return {os.path.join(header, k): v for k, v in objs.items() if isinstance(v, float)}
    if isinstance(objs, torch.Tensor):
        objs = objs.detach().item()
    return {header: float(objs)}


def print_metric_summary(header, objs, epoch_i=None):
    epoch_label = f' epoch={epoch_i}' if epoch_i is not None else ''
    tqdm.tqdm.write(f'{header}{epoch_label} scalar metrics:')
    for key, value in scalar_metric_payload(header, objs).items():
        tqdm.tqdm.write(f'  {key}={value:.8e}')


def write_wandb(header, objs, epoch_i):
    metrics = scalar_metric_payload(header, objs)
    if metrics:
        wandb.log(metrics, step=epoch_i)


def report_metrics(header, objs, epoch_i, log_enabled):
    """Report auxiliary metrics to the terminal and optionally to W&B."""
    print_metric_summary(header, objs, epoch_i)
    if log_enabled:
        write_wandb(header, objs, epoch_i)


def log_used_data_paths(conf):
    if wandb.run is None:
        return

    columns = ["split", "dataset", "data_root", "data_drive", "path", "mode", "window_size", "step_size"]
    table = wandb.Table(columns=columns)
    used_paths = {}

    for split, split_conf in (("train", conf.dataset.train), ("test", conf.dataset.test), ("eval", conf.dataset.eval)):
        used_paths[split] = []
        for data_conf in split_conf.data_list:
            data_root = str(data_conf["data_root"])
            for drive in data_conf.data_drive:
                data_drive = str(drive)
                path = os.path.normpath(os.path.join(data_root, data_drive))
                used_paths[split].append(path)
                table.add_data(
                    split, data_conf.name, data_root, data_drive, path,
                    split_conf.mode, data_conf.window_size, data_conf.step_size,
                )

    wandb.config.update({"used_data_paths": used_paths}, allow_val_change=True)
    wandb.log({"data/used_paths": table})


def save_best_ckpt(
    network,
    optimizer,
    scheduler,
    epoch_i,
    test_loss,
    conf,
    checkpoint_name="best_model.ckpt",
    metric_name="best_loss",
    extra_state=None,
):
    print(f"saving {checkpoint_name}: {metric_name}={test_loss}")
    best_ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", checkpoint_name)
    checkpoint = {
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        'metric_name': metric_name,
        'metric_value': test_loss,
    }
    if hasattr(network, 'acc_cov_min'):
        from model.checkpoints import covariance_policy
        checkpoint['covariance_policy'] = covariance_policy(network)
    if extra_state:
        checkpoint.update(extra_state)
    torch.save(checkpoint, best_ckpt_path)


def save_ckpt(
    network,
    optimizer,
    scheduler,
    epoch_i,
    test_loss,
    conf,
    save_best=False,
    extra_state=None,
):
    checkpoint = {
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
    }
    if hasattr(network, 'acc_cov_min'):
        from model.checkpoints import covariance_policy
        checkpoint['covariance_policy'] = covariance_policy(network)
    if extra_state:
        checkpoint.update(extra_state)

    if epoch_i%conf.train.save_freq==conf.train.save_freq-1:
        torch.save(
            checkpoint,
            os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt" % epoch_i),
        )

    if save_best:
        save_best_ckpt(
            network, optimizer, scheduler, epoch_i, test_loss, conf,
            extra_state=extra_state,
        )

    torch.save(
        checkpoint,
        os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
    )


def upload_best_ckpt_artifact(conf, best_loss):
    if wandb.run is None:
        return

    best_ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    if not os.path.isfile(best_ckpt_path):
        print("best_model.ckpt not found, skipping W&B artifact upload")
        return

    artifact = wandb.Artifact(
        name=f"{wandb.run.name}-best-model",
        type="model",
        metadata={"best_loss": best_loss},
    )
    artifact.add_file(best_ckpt_path)
    wandb.log_artifact(artifact, aliases=["best", "latest"])
    print(f"uploaded best model artifact from {best_ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/EuRoC/codenet.conf', help='config file path')
    parser.add_argument('--device', type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0")
    parser.add_argument('--load_ckpt', default=False, action="store_true", help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.")
    parser.add_argument('--log', default=True, action="store_false", help="if True, save the meta data with wandb")
    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    # torch.cuda.set_device(args.device)

    conf.train.device = args.device
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate]
    else:
        collate_fn = collate_fcs['base']

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=conf.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=conf.train.batch_size, shuffle=False, collate_fn=collate_fn)
    eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        print("wandb is disabled")
    else:
        wandb.init(project= "AirIMU_" + exp_folder,
                    config= conf.train, 
                    group = conf.train.network, 
                    name  = conf_name,)
        log_used_data_paths(conf)

    ## optimizer
    network = net_dict[conf.train.network](conf.train).to(device = args.device, dtype = train_dataset.get_dtype())
    optimizer = torch.optim.Adam(network.parameters(), lr = conf.train.lr, weight_decay=conf.train.weight_decay)  # to use with ViTs
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = conf.train.factor, patience = conf.train.patience, min_lr = conf.train.min_lr)
    best_loss = np.inf
    epoch = 0

    ## load the chkp if there exist
    if args.load_ckpt:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
            checkpoint = torch.load(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), map_location = args.device)
            network.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("loaded state dict %s best_loss %f"%(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), best_loss))
        else:
            print("Can't find the checkpoint")

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer)
        test_loss = test(network, test_loader, conf.train, epoch=epoch_i + 1)
        print("train loss: %f test loss: %f"%(train_loss["loss"], test_loss["loss"]))

        # save the training meta information
        if args.log:
            write_wandb("train", train_loss, epoch_i + 1)
            write_wandb("test", test_loss, epoch_i + 1)
        report_metrics("lr", scheduler.optimizer.param_groups[0]['lr'], epoch_i + 1, args.log)

        if epoch_i%conf.train.eval_freq == conf.train.eval_freq-1:
            eval_state = evaluate(network=network, loader = eval_loader, confs=conf.train)
            report_metrics('eval', {
                f'{state}_rmse': eval_state['loss'][state].mean().item()
                for state in ('pos', 'rot', 'vel')
            }, epoch_i + 1, args.log)

        scheduler.step(test_loss['loss'])
        if test_loss['loss'] < best_loss:
            best_loss = test_loss['loss'];save_best = True
        else:
            save_best = False
            
        save_ckpt(network, optimizer, scheduler, epoch_i, best_loss, conf, save_best=save_best,)

    if args.log:
        upload_best_ckpt_artifact(conf, best_loss)
        wandb.finish()
