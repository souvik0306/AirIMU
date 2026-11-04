import argparse
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.utils.data as Data
import wandb
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from utils import move_to
from train import train, test, evaluate, save_ckpt, write_wandb, upload_best_ckpt_artifact


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_pretrained_model(network, ckpt_path, device, strict=True):
    if ckpt_path is None or str(ckpt_path).lower() in ("", "none", "false"):
        print("No pretrained checkpoint configured; training from scratch.")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint was not found: {ckpt_path}. "
            "Set train.pretrained_ckpt in the config or pass --pretrained_ckpt."
        )

    checkpoint = torch_load(ckpt_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_result = network.load_state_dict(state_dict, strict=strict)
    print(f"loaded pretrained model weights from {ckpt_path}")
    if not strict:
        print(f"missing keys: {load_result.missing_keys}")
        print(f"unexpected keys: {load_result.unexpected_keys}")


def resolve_stage1_vel_ref(conf, pretrained_override=None):
    """Resolve Stage 2's velocity guard from Stage 1's best checkpoint."""
    configured_ref = conf.train.get("stage1_vel_ref", "auto")
    if str(configured_ref).strip().lower() != "auto":
        stage1_vel_ref = float(configured_ref)
        source = "train.stage1_vel_ref"
    else:
        ckpt_path = pretrained_override or conf.train.get("pretrained_ckpt")
        if not ckpt_path:
            raise ValueError(
                "train.stage1_vel_ref is 'auto', but no Stage 1 pretrained_ckpt "
                "was configured."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Cannot resolve Stage 1 velocity reference; checkpoint not found: {ckpt_path}"
            )

        checkpoint = torch_load(ckpt_path, "cpu")
        if "best_loss" not in checkpoint:
            raise KeyError(
                f"Cannot resolve Stage 1 velocity reference; {ckpt_path} has no best_loss."
            )
        stage1_vel_ref = float(checkpoint["best_loss"])
        source = f"{ckpt_path}:best_loss"

    if not np.isfinite(stage1_vel_ref) or stage1_vel_ref < 0:
        raise ValueError(
            f"Stage 1 velocity reference must be finite and non-negative, got {stage1_vel_ref}."
        )

    print(f"Stage 1 velocity reference: {stage1_vel_ref:.12g} (from {source})")
    return stage1_vel_ref


def load_resume_state(network, optimizer, scheduler, ckpt_path, device):
    checkpoint = torch_load(ckpt_path, device)
    network.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    print(f"resumed {ckpt_path} with best_loss {best_loss:f}")
    return epoch, best_loss


NLL_KEYS = ("rot_nll", "vel_nll", "pos_nll", "cov_nll")


def extract_nll_metrics(test_loss):
    """Reuse NLL metrics already accumulated by the normal test pass."""
    return {key: test_loss[key] for key in NLL_KEYS}


def build_loader(
    dataset,
    batch_size,
    shuffle,
    collate_fn,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    # PyTorch only accepts prefetch_factor when multiprocessing is enabled.
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return Data.DataLoader(
        **loader_kwargs,
    )


def log_validation_metrics(network, test_loader, eval_loader, conf, log_enabled, step):
    test_loss = test(network, test_loader, conf.train)
    print("test loss: %f" % (test_loss["loss"]))
    if log_enabled:
        write_wandb("test", test_loss, step)

    eval_state = None
    if step % conf.train.eval_freq == conf.train.eval_freq - 1:
        eval_state = evaluate(network=network, loader=eval_loader, confs=conf.train)
        if log_enabled:
            write_wandb('eval/pos_rmse', eval_state['loss']['pos'].mean(), step)
            write_wandb('eval/rot_rmse', eval_state['loss']['rot'].mean(), step)
            write_wandb('eval/vel_rmse', eval_state['loss']['vel'].mean(), step)
            write_wandb('eval/rot_dist', eval_state['loss']['rot_dist'].mean(), step)
            write_wandb('eval/vel_dist', eval_state['loss']['vel_dist'].mean(), step)
            write_wandb('eval/pos_dist', eval_state['loss']['pos_dist'].mean(), step)
        print("eval pos: %f eval rot: %f" % (eval_state['loss']['pos'].mean(), eval_state['loss']['rot'].mean()))

    return test_loss, eval_state


def collect_data_drives(dataset_conf):
    data_drives = {}
    for split in ("train", "test", "eval"):
        if split not in dataset_conf or "data_list" not in dataset_conf[split]:
            continue

        split_drives = []
        for idx, data_conf in enumerate(dataset_conf[split].data_list):
            source_name = data_conf["name"] if "name" in data_conf else f"source_{idx}"
            drives = list(data_conf["data_drive"]) if "data_drive" in data_conf else []
            split_drives.append({"name": source_name, "data_drive": drives})

        data_drives[split] = split_drives
    return data_drives


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/TLab/finetune_motion_body.conf",
        help="finetuning config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="override train.pretrained_ckpt from the config",
    )
    parser.add_argument(
        "--load_ckpt",
        default=False,
        action="store_true",
        help="resume newest.ckpt from this finetuning experiment instead of loading pretrained weights",
    )
    parser.add_argument(
        "--non_strict_pretrained",
        default=False,
        action="store_true",
        help="allow missing/unexpected keys when loading pretrained model weights",
    )
    parser.add_argument(
        "--log",
        default=True,
        action="store_false",
        help="disable wandb logging",
    )

    args = parser.parse_args()
    print(args)
    conf = ConfigFactory.parse_file(args.config)

    conf.train.device = args.device
    is_stage2 = "stage" in conf.train and conf.train.stage == 2
    if is_stage2:
        # Resolve before saving parameters or starting W&B so the exact value
        # used by this run is captured in both places.
        conf.train.stage1_vel_ref = resolve_stage1_vel_ref(
            conf, pretrained_override=args.pretrained_ckpt
        )
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf["general"]["exp_dir"] = os.path.join(conf.general.exp_dir, conf_name)

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)

    if 'collate' in conf.dataset.keys():
        collate_fn_train = collate_fcs[conf.dataset.collate]
        collate_fn_test = collate_fcs[conf.dataset.collate]
    else:
        collate_fn_train = collate_fcs['base']
        collate_fn_test = collate_fcs['base']

    loader_options = {
        "num_workers": int(conf.train.get("num_workers", 0)),
        "pin_memory": bool(conf.train.get("pin_memory", False)),
        "prefetch_factor": int(conf.train.get("prefetch_factor", 2)),
        "persistent_workers": bool(conf.train.get("persistent_workers", False)),
    }
    print(
        "DataLoader settings: workers={num_workers}, pin_memory={pin_memory}, "
        "prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}".format(
            **loader_options
        )
    )

    train_loader = build_loader(
        train_dataset, conf.train.batch_size, True, collate_fn_train,
        **loader_options,
    )
    test_loader = build_loader(
        test_dataset, conf.train.batch_size, False, collate_fn_test,
        **loader_options,
    )
    eval_loader = build_loader(
        eval_dataset, conf.train.batch_size, False, collate_fn_test,
        drop_last=True, **loader_options,
    )

    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        print("wandb is disabled")
    else:
        sgt_tz = timezone(timedelta(hours=8))
        run_name = f"{conf_name}_{datetime.now(sgt_tz).strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="AirIMU_finetuning",
            config=conf.train,
            group=conf.train.network,
            name=run_name,
        )
        wandb.config.update({"data_drives": collect_data_drives(conf.dataset)})

    network = net_dict[conf.train.network](conf.train).to(
        device=args.device, dtype=train_dataset.get_dtype()
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=conf.train.lr, weight_decay=conf.train.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )
    best_vel_rmse = float("inf")
    epoch = 0
    if is_stage2:
        best_vel_nll = float("inf")
        vel_limit = conf.train.stage1_vel_ref * (1.0 + conf.train.vel_degradation_limit)
    else:
        rot_limit = conf.train.rot_limit if "rot_limit" in conf.train else float("inf")

    resume_path = os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")
    if args.load_ckpt:
        if os.path.isfile(resume_path):
            loaded_epoch, loaded_best = load_resume_state(
                network, optimizer, scheduler, resume_path, args.device
            )
            epoch = loaded_epoch + 1
            if is_stage2:
                best_vel_nll = loaded_best
            else:
                best_vel_rmse = loaded_best
        else:
            print("Can't find the finetuning checkpoint; loading pretrained weights.")
            pretrained_ckpt = args.pretrained_ckpt
            if pretrained_ckpt is None and "pretrained_ckpt" in conf.train:
                pretrained_ckpt = conf.train.pretrained_ckpt
            load_pretrained_model(
                network,
                pretrained_ckpt,
                args.device,
                strict=not args.non_strict_pretrained,
            )
            if is_stage2:
                # The pretrained cov heads are dead weight from a propcov=False
                # stage; re-seed them at the target instead of keeping those stale values.
                network.reset_cov_heads()
    else:
        pretrained_ckpt = args.pretrained_ckpt
        if pretrained_ckpt is None and "pretrained_ckpt" in conf.train:
            pretrained_ckpt = conf.train.pretrained_ckpt
        load_pretrained_model(
            network,
            pretrained_ckpt,
            args.device,
            strict=not args.non_strict_pretrained,
        )
        if is_stage2:
            network.reset_cov_heads()

    initial_step = epoch
    initial_test_loss, initial_eval_state = log_validation_metrics(
        network,
        test_loader,
        eval_loader,
        conf,
        args.log,
        initial_step,
    )

    if is_stage2:
        initial_nll = extract_nll_metrics(initial_test_loss)
        initial_vel_rmse = initial_eval_state['loss']['vel'].mean().item()
        initial_vel_nll = initial_nll["vel_nll"]
        if isinstance(initial_vel_nll, torch.Tensor): initial_vel_nll = initial_vel_nll.item()
        if initial_vel_rmse <= vel_limit:
            best_vel_nll = min(best_vel_nll, initial_vel_nll)
        print(
            "Stage 2 baseline: vel_rmse = %.6f, vel_nll = %.6f"
            % (initial_vel_rmse, initial_vel_nll)
        )
        if args.log:
            write_wandb("monitor/vel_rmse", initial_vel_rmse, initial_step)
            write_wandb("nll", initial_nll, initial_step)

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer)
        log_step = epoch_i + 1
        test_loss, eval_state = log_validation_metrics(
            network,
            test_loader,
            eval_loader,
            conf,
            args.log,
            log_step,
        )
        print("train loss: %f test loss: %f" % (train_loss["loss"], test_loss["loss"]))

        if args.log:
            write_wandb("train", train_loss, log_step)
            write_wandb("lr", scheduler.optimizer.param_groups[0]["lr"], log_step)

        if is_stage2:
            nll_metrics = extract_nll_metrics(test_loss)
            val_vel_rmse = eval_state['loss']['vel'].mean().item()
            vel_nll = nll_metrics["vel_nll"]
            cov_nll = nll_metrics["cov_nll"]
            if isinstance(vel_nll, torch.Tensor): vel_nll = vel_nll.item()
            if isinstance(cov_nll, torch.Tensor): cov_nll = cov_nll.item()
            if args.log:
                write_wandb("monitor/vel_rmse", val_vel_rmse, log_step)
                write_wandb("nll", nll_metrics, log_step)
            scheduler.step(vel_nll)
            if vel_nll < best_vel_nll and val_vel_rmse <= vel_limit:
                best_vel_nll = vel_nll
                save_best = True
                print(
                    "Saving best Stage 2 model: "
                    "vel_rmse = %.6f, vel_nll = %.6f, cov_nll = %.6f"
                    % (val_vel_rmse, vel_nll, cov_nll)
                )
            else:
                save_best = False
            save_ckpt(network, optimizer, scheduler, epoch_i, best_vel_nll, conf, save_best=save_best)
        else:
            val_vel_rmse = eval_state['loss']['vel'].mean().item()
            val_rot_rmse = eval_state['loss']['rot'].mean().item()
            scheduler.step(val_vel_rmse)
            if val_vel_rmse < best_vel_rmse and val_rot_rmse < rot_limit:
                best_vel_rmse = val_vel_rmse
                save_best = True
                print(
                    "Saving best Stage 1 model: vel_rmse = %.6f, rot_rmse = %.6f"
                    % (val_vel_rmse, val_rot_rmse)
                )
            else:
                save_best = False
            save_ckpt(network, optimizer, scheduler, epoch_i, best_vel_rmse, conf, save_best=save_best)

    if args.log:
        if is_stage2:
            upload_best_ckpt_artifact(conf, best_vel_nll)
        else:
            upload_best_ckpt_artifact(conf, best_vel_rmse)
        wandb.finish()
