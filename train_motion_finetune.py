import argparse
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.utils.data as Data
import wandb
from pyhocon import ConfigFactory, ConfigTree
from pyhocon import HOCONConverter as conf_convert
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import SeqeuncesDataset, collate_fcs
from datasets.initial_sensor_bias import resolve_initial_sensor_bias
from model import net_dict
from utils import move_to
from train import (
    train,
    test,
    evaluate,
    save_best_ckpt,
    save_ckpt,
    write_wandb,
    save_wandb_run_metadata,
    upload_best_ckpt_artifact,
)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_pretrained_model(network, ckpt_path, device, strict=True, required=False):
    if ckpt_path is None or str(ckpt_path).lower() in ("", "none", "false"):
        if required:
            raise ValueError(
                "Stage 2 requires the best Stage 1 checkpoint. Pass it with "
                "--pretrained_ckpt."
            )
        print("No pretrained checkpoint configured; training from scratch.")
        return False
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
    return True


def covariance_weight(epoch):
    """Stage 1 covariance-loss multiplier for a zero-based epoch."""
    schedule = {8: 0.1, 9: 0.3, 10: 0.6, 11: 1.0}
    if epoch < 8:
        return 0.0
    return schedule.get(epoch, 1.0)


def configure_stage1_covariance(
    network, confs, epoch, base_cov_weight, covariance_heads_initialized
):
    """Apply the Stage 1 activation schedule before an epoch or validation."""
    multiplier = covariance_weight(epoch)
    enabled = multiplier > 0.0
    if enabled and not covariance_heads_initialized:
        network.reset_cov_heads()
        covariance_heads_initialized = True
        print("Initialized Stage 1 covariance heads before their first training epoch.")

    confs.propcov = enabled
    confs.cov_weight = base_cov_weight * multiplier
    print(
        "Stage 1 covariance schedule: epoch %d, enabled=%s, "
        "multiplier=%.1f, effective_weight=%.12g"
        % (epoch + 1, enabled, multiplier, confs.cov_weight)
    )
    return covariance_heads_initialized, multiplier


def resolve_stage1_vel_ref(conf, pretrained_override=None):
    """Record the corrected-velocity metric from Stage 1's best checkpoint."""
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
        reference_key = (
            "best_corrected_vel_rmse"
            if "best_corrected_vel_rmse" in checkpoint
            else "best_loss"
        )
        if reference_key not in checkpoint:
            raise KeyError(
                f"Cannot resolve Stage 1 velocity reference; {ckpt_path} has no "
                "best_corrected_vel_rmse or best_loss."
            )
        stage1_vel_ref = float(checkpoint[reference_key])
        source = f"{ckpt_path}:{reference_key}"

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
    return epoch, best_loss, checkpoint.get("best_cov_nll", float("inf"))


NLL_KEYS = ("rot_nll", "vel_nll", "pos_nll", "cov_nll")


def extract_nll_metrics(test_loss):
    """Reuse NLL metrics already accumulated by the normal test pass."""
    return {key: test_loss[key] for key in NLL_KEYS}


def configure_stage2_windows(conf):
    """Apply and validate the Stage 2 raw-IMU integration horizon."""
    rate_hz = float(conf.train.get("imu_rate_hz", 200))
    duration_s = float(conf.train.get("integration_window_seconds", 1.0))
    if rate_hz <= 0:
        raise ValueError(f"train.imu_rate_hz must be positive, got {rate_hz}.")
    if not 1.0 <= duration_s <= 2.0:
        raise ValueError(
            "Stage 2 integration_window_seconds must be in [1.0, 2.0]; "
            f"got {duration_s}."
        )

    samples_float = rate_hz * duration_s
    samples = int(round(samples_float))
    if not np.isclose(samples_float, samples):
        raise ValueError(
            "imu_rate_hz * integration_window_seconds must be an integer "
            f"sample count; got {samples_float}."
        )

    for split in ("train", "test", "eval"):
        for data_conf in conf.dataset[split].data_list:
            data_conf["window_size"] = samples
    conf.train["integration_window_samples"] = samples
    print(
        "Stage 2 raw-IMU window: %.3f s at %.3f Hz = %d samples"
        % (duration_s, rate_hz, samples)
    )
    return samples


def configure_stage2_initial_biases(conf):
    """Validate and expose one fixed initial bias for every Stage 2 flight."""
    enabled = bool(conf.train.get("use_initial_sensor_bias", False))
    if not enabled:
        print("Stage 2 initial sensor bias subtraction is disabled.")
        return 0

    bias_root = conf.train.get("initial_sensor_bias_root")
    if not bias_root:
        raise ValueError(
            "Stage 2 use_initial_sensor_bias=true requires "
            "train.initial_sensor_bias_root."
        )
    bias_root = os.path.abspath(str(bias_root))
    if not os.path.isdir(bias_root):
        raise FileNotFoundError(
            f"Initial sensor bias directory was not found: {bias_root}"
        )
    conf.train["initial_sensor_bias_root"] = bias_root

    total = 0
    unique_sequences = set()
    split_counts = {}
    for split in ("train", "test", "eval"):
        split_conf = conf.dataset[split]
        # ConfigTree attribute assignment does not reliably add a new mapping
        # key. SeqeuncesDataset expands the split mapping into TLab, so this
        # must be an actual key for the loader to receive the bias root.
        split_conf["initial_sensor_bias_root"] = bias_root
        split_count = 0
        for data_conf in split_conf.data_list:
            for data_name in data_conf.data_drive:
                resolve_initial_sensor_bias(
                    bias_root, data_conf.data_root, data_name
                )
                sequence = (
                    os.path.basename(os.path.normpath(str(data_conf.data_root))),
                    str(data_name),
                )
                if sequence in unique_sequences:
                    raise ValueError(
                        "A Stage 2 trajectory appears in more than one split: "
                        f"{sequence[0]}/{sequence[1]}"
                    )
                unique_sequences.add(sequence)
                split_count += 1
                total += 1
        split_counts[split] = split_count

    print(
        "Stage 2 initial sensor biases: resolved %d flights from %s "
        "(train=%d, test=%d, eval=%d)"
        % (
            total,
            bias_root,
            split_counts["train"],
            split_counts["test"],
            split_counts["eval"],
        )
    )
    return total


def validate_stage2_dataset_initial_biases(dataset, split):
    """Fail before model execution if a Stage 2 sequence lost its bias pair."""
    missing = []
    incomplete = []
    for sequence_id, (acc_bias, gyro_bias) in enumerate(
        zip(dataset.initial_acc_bias, dataset.initial_gyro_bias)
    ):
        if acc_bias is None and gyro_bias is None:
            missing.append(sequence_id)
        elif acc_bias is None or gyro_bias is None:
            incomplete.append(sequence_id)

    if incomplete:
        raise RuntimeError(
            f"Stage 2 {split} dataset has incomplete initial sensor bias pairs "
            f"for sequence indices {incomplete}."
        )
    if missing:
        raise RuntimeError(
            f"Stage 2 {split} dataset is missing initial sensor biases for "
            f"sequence indices {missing}."
        )
    print(
        f"Stage 2 {split} dataset: attached initial sensor biases to "
        f"{len(dataset.initial_acc_bias)} sequences."
    )


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
    test_loss = test(network, test_loader, conf.train, epoch=step)
    print("test loss: %f" % (test_loss["loss"]))
    if log_enabled:
        write_wandb("test", test_loss, step)
        corrected_metrics = {
            key: value for key, value in test_loss.items()
            if key.startswith("corrected_") and isinstance(value, float)
        }
        write_wandb("validation", corrected_metrics, step)

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
        "--exp_dir",
        type=str,
        default=None,
        help=(
            "exact output directory for this training stage; when omitted, a "
            "unique experiments/<SGT-time>_pid<PID>/<config-name> directory is used"
        ),
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
    parser.add_argument(
        "--max_epoches",
        type=int,
        default=None,
        help="override train.max_epoches from the config",
    )

    args = parser.parse_args()
    print(args)
    conf = ConfigFactory.parse_file(args.config)
    if args.max_epoches is not None:
        conf.train.max_epoches = args.max_epoches

    conf.train.device = args.device
    is_stage2 = "stage" in conf.train and conf.train.stage == 2
    if is_stage2:
        # Resolve before saving parameters or starting W&B so the exact value
        # used by this run is captured in both places.
        conf.train.stage1_vel_ref = resolve_stage1_vel_ref(
            conf, pretrained_override=args.pretrained_ckpt
        )
        configure_stage2_windows(conf)
        configure_stage2_initial_biases(conf)
        if not conf.train.propcov:
            raise ValueError("Stage 2 requires train.propcov=true.")
    else:
        # Stage 1 starts with covariance disabled at runtime, but its shared
        # preintegrator must be constructed with covariance support so epoch 9
        # can activate propagation without rebuilding the model.
        conf.train.propcov = True
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    if "general" not in conf:
        conf["general"] = ConfigTree()
    if args.exp_dir is None:
        sgt_tz = timezone(timedelta(hours=8))
        run_id = datetime.now(sgt_tz).strftime("%Y%m%d_%H%M%S_SGT")
        run_id = f"{run_id}_pid{os.getpid()}"
        conf["general"]["exp_dir"] = os.path.join(
            "experiments", run_id, conf_name
        )
    else:
        conf["general"]["exp_dir"] = os.path.normpath(args.exp_dir)
    print(f"Experiment directory: {conf.general.exp_dir}")

    train_dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    if is_stage2 and conf.train.get("use_initial_sensor_bias", False):
        validate_stage2_dataset_initial_biases(train_dataset, "train")
        validate_stage2_dataset_initial_biases(test_dataset, "test")
        validate_stage2_dataset_initial_biases(eval_dataset, "eval")

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
        save_wandb_run_metadata(conf.general.exp_dir)

    network = net_dict[conf.train.network](conf.train).to(
        device=args.device, dtype=train_dataset.get_dtype()
    )
    base_cov_weight = float(conf.train.get("cov_weight", 0.0))
    if is_stage2:
        if not hasattr(network, "configure_stage2_training"):
            raise TypeError(
                f"{type(network).__name__} does not define Stage 2 uncertainty heads."
            )
        trainable_parameters = network.configure_stage2_training()
        trainable_names = [
            name for name, parameter in network.named_parameters()
            if parameter.requires_grad
        ]
        print("Stage 2 trainable parameters: " + ", ".join(trainable_names))
    else:
        trainable_parameters = list(network.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=conf.train.lr, weight_decay=conf.train.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )
    best_vel_rmse = float("inf")
    best_cov_nll = float("inf")
    epoch = 0
    resumed = False
    pretrained_loaded = False
    # Fresh models initialize these heads in CodeNet.__init__. A legacy
    # correction-only checkpoint overwrites that initialization, so it is
    # re-seeded once immediately before epoch 9.
    covariance_heads_initialized = True
    if is_stage2:
        best_loss = float("inf")
    else:
        rot_limit = conf.train.rot_limit if "rot_limit" in conf.train else float("inf")

    resume_path = os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")
    if args.load_ckpt:
        if os.path.isfile(resume_path):
            loaded_epoch, loaded_best, loaded_best_cov_nll = load_resume_state(
                network, optimizer, scheduler, resume_path, args.device
            )
            resumed = True
            epoch = loaded_epoch + 1
            if is_stage2:
                best_loss = loaded_best
            else:
                best_vel_rmse = loaded_best
                best_cov_nll = loaded_best_cov_nll
                # A checkpoint from before epoch 9 has never trained its
                # covariance heads. Re-seed them once at activation; never
                # reset a checkpoint that has already entered covariance
                # training.
                covariance_heads_initialized = loaded_epoch >= 8
        else:
            print("Can't find the finetuning checkpoint; loading pretrained weights.")
            pretrained_ckpt = args.pretrained_ckpt
            if pretrained_ckpt is None and "pretrained_ckpt" in conf.train:
                pretrained_ckpt = conf.train.pretrained_ckpt
            pretrained_loaded = load_pretrained_model(
                network,
                pretrained_ckpt,
                args.device,
                strict=not args.non_strict_pretrained,
                required=is_stage2,
            )
            if is_stage2:
                # The pretrained cov heads are dead weight from a propcov=False
                # stage; re-seed them at the target instead of keeping those stale values.
                network.reset_cov_heads()
    else:
        pretrained_ckpt = args.pretrained_ckpt
        if pretrained_ckpt is None and "pretrained_ckpt" in conf.train:
            pretrained_ckpt = conf.train.pretrained_ckpt
        pretrained_loaded = load_pretrained_model(
            network,
            pretrained_ckpt,
            args.device,
            strict=not args.non_strict_pretrained,
            required=is_stage2,
        )
        if is_stage2:
            network.reset_cov_heads()

    if not is_stage2:
        if pretrained_loaded and not resumed:
            covariance_heads_initialized = False
        covariance_heads_initialized, _ = configure_stage1_covariance(
            network,
            conf.train,
            epoch,
            base_cov_weight,
            covariance_heads_initialized,
        )

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
        initial_vel_rmse = (
            initial_eval_state['loss']['vel'].mean().item()
            if initial_eval_state is not None
            else initial_test_loss['corrected_vel_rmse']
        )
        initial_cov_nll = initial_nll["cov_nll"]
        if initial_cov_nll < best_loss:
            best_loss = initial_cov_nll
            save_best_ckpt(
                network,
                optimizer,
                scheduler,
                epoch - 1,
                best_loss,
                conf,
                metric_name="raw_cov_nll",
                extra_state={"best_cov_nll": best_loss},
            )
            print(
                "Saved Stage 2 baseline as best model: "
                "corrected_vel_rmse = %.6f, raw_cov_nll = %.6f"
                % (initial_vel_rmse, initial_cov_nll)
            )
        print(
            "Stage 2 baseline: corrected_vel_rmse = %.6f, raw_cov_nll = %.6f"
            % (initial_vel_rmse, initial_cov_nll)
        )
        if args.log:
            write_wandb("monitor/vel_rmse", initial_vel_rmse, initial_step)
            write_wandb("nll", initial_nll, initial_step)

    for epoch_i in range(epoch, conf.train.max_epoches):
        if not is_stage2:
            covariance_heads_initialized, cov_weight_multiplier = configure_stage1_covariance(
                network,
                conf.train,
                epoch_i,
                base_cov_weight,
                covariance_heads_initialized,
            )
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
            if not is_stage2:
                write_wandb(
                    "schedule/covariance_weight_multiplier",
                    cov_weight_multiplier,
                    log_step,
                )
                write_wandb(
                    "schedule/effective_covariance_weight",
                    float(conf.train.cov_weight),
                    log_step,
                )

        if is_stage2:
            nll_metrics = extract_nll_metrics(test_loss)
            val_vel_rmse = (
                eval_state['loss']['vel'].mean().item()
                if eval_state is not None
                else test_loss['corrected_vel_rmse']
            )
            vel_nll = nll_metrics["vel_nll"]
            cov_nll = nll_metrics["cov_nll"]
            if args.log:
                write_wandb("monitor/vel_rmse", val_vel_rmse, log_step)
                write_wandb("nll", nll_metrics, log_step)
            scheduler.step(cov_nll)
            if cov_nll < best_loss:
                best_loss = cov_nll
                save_best = True
                print(
                    "Saving best Stage 2 model: "
                    "corrected_vel_rmse = %.6f, vel_nll = %.6f, raw_cov_nll = %.6f"
                    % (val_vel_rmse, vel_nll, cov_nll)
                )
            else:
                save_best = False
            if save_best:
                save_best_ckpt(
                    network, optimizer, scheduler, epoch_i, best_loss, conf,
                    metric_name="raw_cov_nll",
                    extra_state={"best_cov_nll": best_loss},
                )
            save_ckpt(
                network, optimizer, scheduler, epoch_i, best_loss, conf,
                save_best=False,
                extra_state={"best_cov_nll": best_loss},
            )
        else:
            if eval_state is not None:
                val_vel_rmse = eval_state['loss']['vel'].mean().item()
                val_rot_rmse = eval_state['loss']['rot'].mean().item()
            else:
                val_vel_rmse = test_loss['corrected_vel_rmse']
                val_rot_rmse = test_loss['corrected_rot_rmse']
            scheduler.step(val_vel_rmse)
            if val_vel_rmse < best_vel_rmse and val_rot_rmse < rot_limit:
                best_vel_rmse = val_vel_rmse
                save_best_corrected = True
                print(
                    "Saving best Stage 1 model: vel_rmse = %.6f, rot_rmse = %.6f"
                    % (val_vel_rmse, val_rot_rmse)
                )
            else:
                save_best_corrected = False

            if save_best_corrected:
                save_best_ckpt(
                    network, optimizer, scheduler, epoch_i, best_vel_rmse, conf,
                    checkpoint_name="best_corrected_model.ckpt",
                    metric_name="corrected_vel_rmse",
                    extra_state={"best_corrected_vel_rmse": best_vel_rmse},
                )
                # Keep the established path as a compatibility alias for
                # existing scripts; the explicit filename above is canonical.
                save_best_ckpt(
                    network, optimizer, scheduler, epoch_i, best_vel_rmse, conf,
                    checkpoint_name="best_model.ckpt",
                    metric_name="corrected_vel_rmse",
                    extra_state={"best_corrected_vel_rmse": best_vel_rmse},
                )

            if conf.train.propcov:
                val_cov_nll = test_loss['corrected_cov_nll']
                if val_cov_nll < best_cov_nll:
                    best_cov_nll = val_cov_nll
                    print(
                        "Saving best calibrated Stage 1 model: "
                        "corrected_cov_nll = %.6f" % best_cov_nll
                    )
                    save_best_ckpt(
                        network, optimizer, scheduler, epoch_i, best_cov_nll, conf,
                        checkpoint_name="best_covariance_model.ckpt",
                        metric_name="corrected_cov_nll",
                        extra_state={
                            # Preserve corrected-trajectory provenance even
                            # when this calibration-selected checkpoint loads.
                            "best_loss": best_vel_rmse,
                            "best_corrected_vel_rmse": best_vel_rmse,
                            "best_cov_nll": best_cov_nll,
                        },
                    )

            save_ckpt(
                network,
                optimizer,
                scheduler,
                epoch_i,
                best_vel_rmse,
                conf,
                save_best=False,
                extra_state={"best_cov_nll": best_cov_nll},
            )

    if args.log:
        if is_stage2:
            upload_best_ckpt_artifact(conf, best_loss)
        else:
            upload_best_ckpt_artifact(conf, best_vel_rmse)
        wandb.finish()
