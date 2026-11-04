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


class _LossMetrics:
    """Keep detached loss totals on the device until a reporting interval."""

    def __init__(self, propcov, uncertainty_keys=()):
        self.sources = {
            'loss': 'loss', 'pos_loss': 'pos', 'rot_loss': 'rot', 'vel_loss': 'vel',
        }
        if propcov:
            self.sources.update({key: key for key in (
                'pred_cov_rot', 'pred_cov_vel', 'pred_cov_pos', *uncertainty_keys,
            )})
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
        result = dict(zip(self.sources, values))
        for key in ('pred_cov_rot', 'pred_cov_vel', 'pred_cov_pos'):
            result.setdefault(key, 0.0)
        return result


def train(network, loader, confs, epoch, optimizer):
    """Train for one epoch, transferring scalar metrics only when reporting."""
    network.train()
    metrics = _LossMetrics(confs.propcov, (
        'raw_vel_rmse', 'vel_nees', 'vel_1sigma', 'vel_2sigma', 'vel_nll',
    ))
    output_metrics = OutputMetrics()
    log_interval = max(1, int(confs.get('log_interval', 10)))

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
            loss_state['loss'].backward()
            optimizer.step()

            if (i + 1) % log_interval == 0 or i + 1 == len(loader):
                current = metrics.averages()
                description = (
                    f"training epoch: {epoch:03d}, losses: {current['loss']:.6f}, "
                    f"position {current['pos_loss']:.6f}, rotation {current['rot_loss']:.6f}, "
                    f"velocity {current['vel_loss']:.6f}"
                )
                if confs.propcov:
                    description += (
                        f", pred_rot {current['pred_cov_rot']:.6f}, "
                        f"pred_cov {current['pred_cov_pos']:.6f}"
                    )
                t_range.set_description(description, refresh=False)

    result = metrics.averages()
    result.update(output_metrics.summarize('TRAIN', epoch=epoch + 1, propcov=confs.propcov))
    return result


def test(network, loader, confs, epoch=None):
    network.eval()
    output_metrics = OutputMetrics()
    metrics = _LossMetrics(confs.propcov, (
        'raw_vel_rmse', 'vel_nees', 'vel_1sigma', 'vel_2sigma',
        'rot_nll', 'vel_nll', 'pos_nll', 'cov_nll',
    ))
    log_interval = max(1, int(confs.get('log_interval', 10)))
    acc_covs, gyro_covs = [], []

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

            if confs.propcov:
                acc_covs.append(inte_state['acc_cov'].reshape(-1))
                gyro_covs.append(inte_state['gyro_cov'].reshape(-1))

            if (i + 1) % log_interval == 0 or i + 1 == len(loader):
                current = metrics.averages()
                t_range.set_description(
                    f"testing losses: {current['loss']:.6f}, "
                    f"position {current['pos_loss']:.6f}, rotation {current['rot_loss']:.6f}, "
                    f"velocity {current['vel_loss']:.6f}", refresh=False,
                )

        if acc_covs:
            acc_covs = torch.cat(acc_covs)
        if gyro_covs:
            gyro_covs = torch.cat(gyro_covs)

    result = metrics.averages()
    result.update({'acc_covs': acc_covs, 'gyro_covs': gyro_covs})
    result.update(output_metrics.summarize('VALIDATION', epoch=epoch, propcov=confs.propcov))
    return result


def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        metrics = {os.path.join(header, k): v for k, v in objs.items() if isinstance(v, float)}
        if metrics:
            wandb.log(metrics, step=epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


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


def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best = False):
    if epoch_i%conf.train.save_freq==conf.train.save_freq-1:
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt"%epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        best_ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, best_ckpt_path)

    torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))


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
            write_wandb("train", train_loss, epoch_i)
            write_wandb("test", test_loss, epoch_i)
            write_wandb("lr", scheduler.optimizer.param_groups[0]['lr'], epoch_i)

        if epoch_i%conf.train.eval_freq == conf.train.eval_freq-1:
            eval_state = evaluate(network=network, loader = eval_loader, confs=conf.train)
            if args.log:
                write_wandb('eval/pos_rmse', eval_state['loss']['pos'].mean(), epoch_i)
                write_wandb('eval/rot_rmse', eval_state['loss']['rot'].mean(), epoch_i)
                write_wandb('eval/vel_rmse', eval_state['loss']['vel'].mean(), epoch_i)
                write_wandb('eval/rot_dist', eval_state['loss']['rot_dist'].mean(), epoch_i)
                write_wandb('eval/vel_dist', eval_state['loss']['vel_dist'].mean(), epoch_i)
                write_wandb('eval/pos_dist', eval_state['loss']['pos_dist'].mean(), epoch_i)
                
            print("eval pos: %f eval rot: %f"%(eval_state['loss']['pos'].mean(), eval_state['loss']['rot'].mean()))

        scheduler.step(test_loss['loss'])
        if test_loss['loss'] < best_loss:
            best_loss = test_loss['loss'];save_best = True
        else:
            save_best = False
            
        save_ckpt(network, optimizer, scheduler, epoch_i, best_loss, conf, save_best=save_best,)

    if args.log:
        upload_best_ckpt_artifact(conf, best_loss)
        wandb.finish()
