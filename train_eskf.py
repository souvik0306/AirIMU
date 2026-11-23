"""Train AirIMU covariance and selected GRUs through full-flight ESKF replay."""

import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from ekf import DifferentiableESKF
from ekf.data import apply_sensor_metadata, read_dataset_flights
from ekf.experiment import filter_metadata, load_airimu, save_checkpoint, file_sha256
from ekf.runner import FlightTrainer
from ekf.gradients import clipping_groups, clipping_policy
from ekf.stability import LossGrowthGuard, ValidationEarlyStopping


def log_flight(output_dir, record):
    print(json.dumps(record), flush=True)
    with (output_dir / "flights.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def log_epoch(output_dir, record):
    """Save structured metrics and print readable per-axis epoch statistics."""
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    print(f"epoch={record['epoch']} train_xy_mse={record['train_xy_mse']:.6g} "
          f"validation_xy_mse={record['validation_xy_mse']:.6g} "
          f"best_validation_xy_mse={record['best_validation_xy_mse']:.6g} "
          f"epoch_seconds={record['epoch_seconds']:.3f}", flush=True)
    for phase in ("train", "validation"):
        covariance = record[f"{phase}_covariance"]
        for sensor in ("acc_cov", "gyro_cov"):
            statistics = covariance[sensor]
            values = {
                name: "[" + ", ".join(f"{value:.6e}" for value in statistics[key]) + "]"
                for name, key in (("min", "min"), ("avg", "mean"), ("max", "max"))
            }
            print(f"epoch={record['epoch']} phase={phase} sensor={sensor} axes=x,y,z "
                  f"samples={covariance['samples']} min={values['min']} "
                  f"avg={values['avg']} max={values['max']}", flush=True)


def log_batch(output_dir, record):
    with (output_dir / 'batches.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')


def restore_adapter_optimizer(optimizer, checkpoint):
    """Restore Adam moments/steps without silently changing the requested LR."""
    if checkpoint.get('ekf_metadata',{}).get('configuration') != 'Adapter':
        raise ValueError('Resume requires an adapter-training checkpoint.')
    saved=checkpoint['optimizer_state_dict']
    current=[(group.get('name'),group['lr'],len(group['params'])) for group in optimizer.param_groups]
    previous=[(group.get('name'),group['lr'],len(group['params'])) for group in saved['param_groups']]
    if current != previous:
        raise ValueError(f'Resume optimizer groups/LRs differ: {previous} != {current}')
    if not saved['state'] or len(saved['state']) != sum(len(g['params']) for g in saved['param_groups']):
        raise ValueError('Resume requires Adam state for every adapter parameter.')
    steps={int(state['step']) for state in saved['state'].values()}
    if len(steps)!=1 or next(iter(steps))<1:
        raise ValueError('Inconsistent or empty Adam step counters.')
    optimizer.load_state_dict(saved)
    loaded=optimizer.state_dict()
    for key,state in saved['state'].items():
        for name,value in state.items():
            actual=loaded['state'][key][name]
            if torch.is_tensor(value):
                if not bool(torch.isfinite(value).all()) or not torch.equal(value.cpu(),actual.cpu()):
                    raise ValueError(f'Adam state differs or is non-finite: {key}/{name}')
            elif actual != value:
                raise ValueError(f'Adam state differs: {key}/{name}')
    return next(iter(steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing AirIMU network HOCON config")
    parser.add_argument("--ckpt", required=True, help="AirIMU checkpoint (covariance heads are retained)")
    parser.add_argument("--dataset-config", required=True, help="TLab HOCON dataset split config")
    parser.add_argument(
        "--sensor-metadata-root", required=True,
        help="Root containing the finalized initial_sensor_bias_*.csv tables",
    )
    parser.add_argument("--output", required=True, help="Experiment directory for ESKF checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ekf-device", help="Filter device; defaults to CPU with replay workers, otherwise --device")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument('--resume',action='store_true',
                        help='Adapter only: restore Adam and continue epoch numbering; --epochs is additional epochs')
    parser.add_argument('--evaluate-initial', action='store_true',
                        help='Evaluate the untouched checkpoint on validation before the first update')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--configuration", choices=("A", "B", "C", "Adapter"), default="B",
                        help="A: heads/gains; B: plus GRU2; C: full CNN/GRU1/GRU2 encoder")
    parser.add_argument("--configuration-c-evidence", nargs=2, metavar=("A_RUN", "B_RUN"),
                        help="Optional legacy matched A/B validation evidence check")
    parser.add_argument("--bias-lr", type=float, default=1e-5)
    parser.add_argument('--initialize-adapter', action='store_true',
                        help='Explicitly migrate a parent checkpoint by adding only zero-output adapter tensors')
    parser.add_argument('--adapter-lr', type=float, default=1e-4)
    parser.add_argument('--adapter-weight', type=float, default=0.,
                        help='Penalty on per-flight mean squared log covariance adjustment')
    parser.add_argument('--orientation-smoothing-window',type=float,default=0.,
                        help='Calibration-only SO(3) local quadratic window in seconds; 0 retains legacy references')
    parser.add_argument('--milestone-epochs',type=int,nargs='*',default=[])
    parser.add_argument('--milestone-report-root',type=Path,
                        help='Prepared stronger-regularization run for detailed validation-only milestone reports')
    parser.add_argument("--gru2-lr", type=float, default=1e-6)
    parser.add_argument("--gru1-lr", type=float, default=5e-7)
    parser.add_argument('--cnn-lr', type=float, default=3e-7)
    parser.add_argument("--gyro-weight", type=float, default=1e-4,
                        help="Weak Vicon gyro calibration regularizer; tune on validation")
    parser.add_argument("--smoothness-weight", type=float, default=0.,
                        help="Log gyro variance smoothness; introduce after calibration/GRU comparisons")
    parser.add_argument("--calibration-window", type=float, default=.05, help="Centered SO(3) secant window in seconds")
    parser.add_argument("--orientation-std", type=float, default=.001, help="Vicon orientation noise std (rad), tune on validation")
    parser.add_argument("--calibration-max-step-deg", type=float, default=45.,
                        help="Reject reference windows touching larger adjacent Vicon rotation steps")
    parser.add_argument("--calibration-max-rate", type=float, default=20.,
                        help="Reject reference windows touching larger adjacent Vicon rates (rad/s)")
    parser.add_argument("--correction-weight", type=float, default=0., help="Optional weak normalized correction consistency")
    parser.add_argument("--gradient-clip", type=float, default=1.)
    parser.add_argument('--gradient-clipping', choices=('global','separate'), default='global',
                        help='Separate: independently clip each trainable encoder module and the combined heads/gains')
    parser.add_argument('--gru2-gradient-clip', type=float, default=5.)
    parser.add_argument('--head-gradient-clip', type=float, default=1.)
    parser.add_argument('--cnn-gradient-clip', type=float, default=1.)
    parser.add_argument('--gru1-gradient-clip', type=float, default=1.)
    parser.add_argument('--early-stopping-patience',type=int,default=0,
                        help='Consecutive validation epochs without meaningful MSE progress; 0 disables')
    parser.add_argument('--early-stopping-min-delta',type=float,default=1e-4,
                        help='Absolute validation MSE improvement needed to reset patience')
    parser.add_argument('--require-encoder-gradients',action='store_true',
                        help='Require finite, present, nonzero gradients for each trainable encoder group')
    parser.add_argument('--max-relative-update', type=float,
                        help='Stop if Adam update L2 / max(parameter group L2, 1) exceeds this limit')
    parser.add_argument('--loss-growth-factor', type=float,
                        help='Stop after sustained epoch MSE growth above this multiple of the best')
    parser.add_argument('--loss-growth-patience', type=int, default=2)
    parser.add_argument("--allow-reduced-split", action="store_true", help="Allow small smoke fixtures instead of 82/23/12 flights")
    parser.add_argument("--gravity", type=float, nargs=3, default=(0., 0., -9.80665))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--replay-workers", type=int, default=0,
                        help="Independent CPU replay processes; 0 replays in the main process")
    parser.add_argument("--flight-batch-size", type=int, default=1,
                        help="Complete flights averaged per Adam step; use >= replay-workers for parallel work")
    parser.add_argument("--feature-cache-mb", type=float, default=512,
                        help="CPU cache budget for frozen encoder features; 0 disables caching")
    parser.add_argument("--replay-backend", choices=("eager", "compiled"), default="compiled",
                        help="Compiled aligned TLab replay preserves full-flight gradients and Float64")
    parser.add_argument("--height-updates", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse clean Vicon z before and after cutoff at every estimate, std=0.1 m (default: enabled)")
    args = parser.parse_args()
    if not math.isfinite(args.orientation_smoothing_window) or args.orientation_smoothing_window<0:
        parser.error('Orientation smoothing window must be finite and nonnegative.')
    if any(epoch<1 for epoch in args.milestone_epochs):parser.error('Milestone epochs must be positive.')
    if args.milestone_report_root and args.configuration!='Adapter':parser.error('Milestone reports require Adapter.')
    for name in ("gyro_weight", "smoothness_weight", "orientation_std", "correction_weight", "adapter_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative.")
    for name in ("bias_lr", "gru2_lr", "gru1_lr", "gradient_clip", "calibration_window",
                 'gru2_gradient_clip','head_gradient_clip','cnn_gradient_clip','gru1_gradient_clip','cnn_lr','adapter_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive.")
    if args.gradient_clipping == 'separate' and args.configuration not in ('B','C'):
        parser.error('Separate encoder/head clipping is supported for Configurations B and C.')
    if (args.initialize_adapter or args.adapter_weight) and args.configuration != 'Adapter':
        parser.error('Adapter migration and penalty require --configuration Adapter.')
    if args.resume and (args.configuration!='Adapter' or args.initialize_adapter or not args.evaluate_initial):
        parser.error('Resume requires Adapter, --evaluate-initial, and an existing adapter checkpoint.')
    if args.max_relative_update is not None and (not math.isfinite(args.max_relative_update) or args.max_relative_update <= 0):
        parser.error('max-relative-update must be finite and positive.')
    try:
        stability = LossGrowthGuard(args.loss_growth_factor,args.loss_growth_patience)
        early_stopping = ValidationEarlyStopping(args.early_stopping_patience,args.early_stopping_min_delta)
    except ValueError as error:
        parser.error(str(error))
    if (not math.isfinite(args.calibration_max_step_deg) or not 0 < args.calibration_max_step_deg <= 180
            or not math.isfinite(args.calibration_max_rate) or args.calibration_max_rate <= 0):
        parser.error('Use a calibration step limit in (0, 180] degrees and a positive rate limit.')
    if args.configuration_c_evidence:
        from ekf.experiment import validate_gru1_evidence
        validate_gru1_evidence(*args.configuration_c_evidence)
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.cpu_threads < 1:
        parser.error("epochs, lr, and cpu-threads must be positive.")
    if (args.replay_workers < 0 or args.flight_batch_size < 1 or not math.isfinite(args.feature_cache_mb)
            or args.feature_cache_mb < 0):
        parser.error("Use nonnegative workers/cache budget and a positive batch size.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    flights, cutoffs = apply_sensor_metadata(
        read_dataset_flights(args.dataset_config), args.sensor_metadata_root
    )
    training = [record for record in flights if record["split"] == "train"]
    validation = [record for record in flights if record["split"] == "validation"]
    if not training or not validation:
        parser.error("Dataset config must provide disjoint train and validation flights.")
    counts = {split: sum(record["split"] == split for record in flights) for split in ("train", "validation", "test")}
    if not args.allow_reduced_split and counts != {"train": 82, "validation": 23, "test": 12}:
        parser.error(f"Expected 82/23/12 train/validation/test flights, got {counts}.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ekf_device = args.ekf_device or ("cpu" if args.replay_workers else args.device)
    if args.replay_workers and torch.device(ekf_device).type != "cpu":
        parser.error("--replay-workers requires --ekf-device cpu (AirIMU can stay on CUDA).")
    ekf = DifferentiableESKF(gravity=args.gravity, height_updates=args.height_updates).to(ekf_device)
    wrapper = load_airimu(args.config, args.ckpt, ekf, args.device, configuration=args.configuration,
                         initialize_adapter=args.initialize_adapter)
    if wrapper.network.use_cov_limits:
        parser.error("Use an uncapped covariance config (use_cov_limits=False) for joint training.")
    covariance_policy = {
        "use_cov_limits": wrapper.network.use_cov_limits,
        "acc_variance_min": float(wrapper.network.acc_cov_min),
        "gyro_variance_min": float(wrapper.network.gyro_cov_min),
        "acc_variance_max": (float(wrapper.network.acc_cov_max)
                             if wrapper.network.use_cov_limits else None),
        "gyro_variance_max": (float(wrapper.network.gyro_cov_max)
                              if wrapper.network.use_cov_limits else None),
    }
    groups = wrapper.optimizer_groups(head_lr=args.lr, bias_lr=args.bias_lr,
                                      gru2_lr=args.gru2_lr, gru1_lr=args.gru1_lr, cnn_lr=args.cnn_lr,
                                      adapter_lr=args.adapter_lr)
    frozen_parent = ({name:value.detach().cpu().clone() for name,value in wrapper.network.state_dict().items()
                      if not name.startswith('raw_cov_adapter.')} if args.configuration == 'Adapter' else None)
    optimizer = torch.optim.Adam(groups)
    start_epoch=0;restored_steps=0
    if args.resume:
        checkpoint=torch.load(args.ckpt,map_location='cpu',weights_only=True)
        start_epoch=int(checkpoint['epoch'])
        if start_epoch<1:parser.error('Resume requires a previously trained checkpoint.')
        previous_metadata=checkpoint['ekf_metadata']
        for key,value in (('adapter_weight',args.adapter_weight),('correction_weight',args.correction_weight),
                          ('seed',args.seed),('flight_batch_size',args.flight_batch_size)):
            if previous_metadata.get(key)!=value:parser.error(f'Resume changes {key}.')
        restored_steps=restore_adapter_optimizer(optimizer,checkpoint)
        # Legacy checkpoints did not save Python shuffle state. Reconstruct
        # the exact permutation/RNG sequence from the same seed and split.
        for _ in range(start_epoch):rng.shuffle(training)
        del checkpoint
    clip_policy = clipping_policy(clipping_groups(groups, mode=args.gradient_clipping,
        global_limit=args.gradient_clip, gru2_limit=args.gru2_gradient_clip,
        head_limit=args.head_gradient_clip,cnn_limit=args.cnn_gradient_clip,
        gru1_limit=args.gru1_gradient_clip),args.gradient_clipping)
    loss_options = {"gyro_weight": args.gyro_weight, "smoothness_weight": args.smoothness_weight,
                    "calibration_window": args.calibration_window, "orientation_std": args.orientation_std,
                    "calibration_max_step_deg":args.calibration_max_step_deg,
                    "calibration_max_rate":args.calibration_max_rate,
                    "gyro_floor": float(wrapper.network.gyro_cov_min)}
    if args.orientation_smoothing_window:
        loss_options['orientation_smoothing_window']=args.orientation_smoothing_window
    if args.resume:
        if previous_metadata['loss_options']!=loss_options:parser.error('Resume changes EKF loss options.')
        for key,value in filter_metadata(ekf).items():
            if previous_metadata.get(key)!=value:parser.error(f'Resume changes EKF aiding: {key}.')
        def identities(records):
            # Source CSV paths move into the new immutable run snapshot.
            values=json.loads(json.dumps(records))
            for row in values:row.get('sensor_metadata',{}).pop('source_csv',None)
            return sorted(values,key=lambda row:row['id'])
        current_flights=[{**record,'cutoff_timestamp':cutoffs[record['id']]} for record in training+validation]
        if identities(previous_metadata['flights'])!=identities(current_flights):
            parser.error('Resume changes train/validation identities, biases or cutoffs.')
        print(f'Restored Adam: epoch={start_epoch}, optimizer_steps={restored_steps}; LR unchanged.',flush=True)
    metadata = {"gravity": list(args.gravity), "observation_types": list(ekf.observation_types),
                "timestamp_units": "seconds, original shared reference",
                "evaluation_start_offset": 5., "evaluation_duration": 20.,
                "config": str(Path(args.config).resolve()),
                "initial_checkpoint": str(Path(args.ckpt).resolve()),
                "initial_checkpoint_sha256": file_sha256(args.ckpt),
                "flights": [{**record, "cutoff_timestamp": cutoffs[record["id"]]}
                            for record in training + validation], "seed": args.seed,
                "network_device": args.device, "ekf_device": ekf_device,
                "return_covariance_during_training": False,
                "replay_workers": args.replay_workers, "worker_threads": 1,
                "flight_batch_size": args.flight_batch_size, "feature_cache_mb": args.feature_cache_mb,
                "replay_backend": args.replay_backend,
                "configuration": args.configuration, "loss_options": loss_options,
                'configuration_definition':{'A':'heads_and_bias_gains','B':'plus_gru2',
                    'C':'full_encoder_cnn_gru1_gru2','Adapter':'frozen_parent_residual_covariance_adapter'}[args.configuration],
                'adapter_weight':args.adapter_weight, 'initialize_adapter':args.initialize_adapter,
                'milestone_epochs':args.milestone_epochs,
                'resume':{'enabled':args.resume,'starting_epoch':start_epoch,'restored_adam_steps':restored_steps,
                          'additional_epochs':args.epochs,'maximum_total_epochs':start_epoch+args.epochs,
                          'shuffle_state':'reconstructed from seed and prior in-place permutations' if args.resume else 'seed'},
                'early_stopping':{'patience':args.early_stopping_patience,
                    'min_delta':args.early_stopping_min_delta,'metric':'validation_xy_mse','mode':'min'},
                'require_encoder_gradients':args.require_encoder_gradients,
                "evaluate_initial":args.evaluate_initial,
                "gyro_reference_quality_policy":"vicon_so3_quality_v1",
                "gradient_diagnostics":"batch-mean parameter gradients; gyro weighted and unweighted; before clipping",
                "correction_weight": args.correction_weight,
                "gradient_clip": args.gradient_clip if args.gradient_clipping == 'global' else None,
                'gradient_clipping':clip_policy,
                'stability_policy':{'max_relative_update':args.max_relative_update,
                    'update_normalization':'update_l2 / max(parameter_group_l2_before, 1); unit floor for zero bias gains',
                    'loss_growth_factor':args.loss_growth_factor,'loss_growth_patience':args.loss_growth_patience,
                    'nonfinite_loss_gradient_update':'stop immediately'},
                "optimizer_groups": [{"name": group["name"], "lr": group["lr"]} for group in groups],
                "split_counts": counts,
                "effective_feature_cache_mb": args.feature_cache_mb if wrapper.cacheable_features else 0.,
                "motion_thresholds": {"quiet_rate_rad_s": .1, "quiet_accel_rad_s2": 1.,
                                      "rapid_rate_rad_s": 1., "rapid_accel_rad_s2": 10.},
                "covariance_policy": covariance_policy,
                "optimizer_objective": "mean_flight(velocity_mse + gyro_weight*gyro_nll + smoothness_weight*log_variance_smoothness) + correction_weight*consistency",
                **filter_metadata(ekf)}
    if args.configuration == 'Adapter':
        metadata['optimizer_objective'] += ' + adapter_weight*mean_flight(mean_time_axis(delta**2))'
    with (output_dir / "eskf_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    best_validation = math.inf
    print(f"Network device: {args.device}; EKF device: {ekf_device}; "
          f"flights per epoch: {len(training)} train + {len(validation)} validation", flush=True)
    print(f"Initial biases: flight-specific CSV values from {Path(args.sensor_metadata_root).resolve()}; "
          "refined during Vicon aiding, retained at cutoff; raw IMU is unchanged.", flush=True)
    print("Covariance logs contain per-axis variances in x,y,z order, before the optimizer step.", flush=True)
    print(f"Replay workers: {args.replay_workers}; threads/worker: 1; "
          f"flights/Adam step: {args.flight_batch_size}; feature cache: {args.feature_cache_mb:g} MiB", flush=True)
    print(f"Replay backend: {args.replay_backend}", flush=True)
    print(f"Covariance policy: {json.dumps(covariance_policy)}", flush=True)
    print(f"Gradient clipping policy: {json.dumps(clip_policy)}", flush=True)
    print(f"Stability policy: {json.dumps(metadata['stability_policy'])}", flush=True)
    print(f"Filter aiding policy: {json.dumps(filter_metadata(ekf))}", flush=True)
    with FlightTrainer(wrapper, optimizer, workers=args.replay_workers, batch_size=args.flight_batch_size,
                       feature_cache_mb=args.feature_cache_mb, replay_backend=args.replay_backend,
                       loss_options=loss_options, gradient_clip=args.gradient_clip,
                       correction_weight=args.correction_weight, gradient_clipping=args.gradient_clipping,
                       gru2_gradient_clip=args.gru2_gradient_clip, head_gradient_clip=args.head_gradient_clip,
                       max_relative_update=args.max_relative_update,cnn_gradient_clip=args.cnn_gradient_clip,
                       gru1_gradient_clip=args.gru1_gradient_clip,
                       require_encoder_gradients=args.require_encoder_gradients,
                       adapter_weight=args.adapter_weight) as trainer:
        if args.evaluate_initial:
            started = perf_counter()
            before = {name:value.detach().cpu().clone() for name,value in wrapper.network.state_dict().items()}
            initial_records = []
            def log_initial(record):
                initial_records.append(record)
                print(json.dumps(record),flush=True)
                with (output_dir/'initial_validation_flights.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(record)+'\n')
            initial_losses, initial_covariance = trainer.run_split(validation,cutoffs,
                epoch=start_epoch,phase='validation',epoch_start=started,log=log_initial)
            if trainer.optimizer_steps or any(not torch.equal(value.detach().cpu(),before[name])
                                              for name,value in wrapper.network.state_dict().items()):
                raise RuntimeError('Initial validation changed model weights or performed an optimizer step.')
            initial = {'epoch':start_epoch,'validation_xy_mse':sum(initial_losses)/len(initial_losses),
                'validation_xy_rmse':math.sqrt(sum(initial_losses)/len(initial_losses)),
                'validation_covariance':initial_covariance,
                'gyro_calibration_loss':sum(r['gyro_calibration_loss'] for r in initial_records)/len(initial_records),
                'reference_quality':trainer.last_split_diagnostics['gyro_reference_quality'],
                'optimizer_steps':0,'weights_unchanged':True,'validation_seconds':perf_counter()-started}
            (output_dir/'initial_validation.json').write_text(json.dumps(initial,indent=2)+'\n')
            stability.observe('validation',initial['validation_xy_mse'])
            early_stopping.observe(initial['validation_xy_mse'])
            if args.configuration == 'Adapter':
                # The unchanged B Strong checkpoint can win if no adaptation
                # improves validation. Never force selection of a regression.
                best_validation = initial['validation_xy_mse']
                save_checkpoint(output_dir/'best_model.ckpt',wrapper,optimizer,start_epoch,best_validation,metadata)
            print(f"epoch={start_epoch} validation_xy_mse={initial['validation_xy_mse']:.9g} "
                  f"validation_xy_rmse={initial['validation_xy_rmse']:.9g} optimizer_steps=0",flush=True)
            del before,initial_records
        for epoch in range(start_epoch,start_epoch+args.epochs):
            epoch_start = perf_counter()
            rng.shuffle(training)
            previous_steps = trainer.optimizer_steps
            train_losses, train_covariance = trainer.run_split(
                training, cutoffs, epoch=epoch+1, phase="train", epoch_start=epoch_start,
                log=lambda record: log_flight(output_dir, record),
                log_batch=lambda record: log_batch(output_dir, record))
            train_diagnostics = trainer.last_split_diagnostics
            training_finished = perf_counter()
            validation_losses, validation_covariance = trainer.run_split(
                validation, cutoffs, epoch=epoch+1, phase="validation", epoch_start=epoch_start,
                log=lambda record: log_flight(output_dir, record),
                log_batch=lambda record: log_batch(output_dir, record))
            validation_diagnostics = trainer.last_split_diagnostics
            validation_finished = perf_counter()
            if frozen_parent is not None:
                current = wrapper.network.state_dict()
                if any(not torch.equal(current[name].detach().cpu(),value) for name,value in frozen_parent.items()):
                    raise RuntimeError('Frozen parent weights or buffers changed during adapter training.')
            validation_mse = sum(validation_losses)/len(validation_losses)
            if not math.isfinite(validation_mse):
                raise RuntimeError("Validation loss is non-finite.")
            stability.observe('train',sum(train_losses)/len(train_losses))
            stability.observe('validation',validation_mse)
            should_stop = early_stopping.observe(validation_mse)
            if validation_mse < best_validation:
                best_validation = validation_mse
                save_checkpoint(output_dir / "best_model.ckpt", wrapper, optimizer, epoch+1,
                                best_validation, metadata)
            save_checkpoint(output_dir / "newest.ckpt", wrapper, optimizer, epoch+1, best_validation, metadata)
            record = {"epoch": epoch+1, "train_xy_mse": sum(train_losses)/len(train_losses),
                      "validation_xy_mse": validation_mse, "best_validation_xy_mse": best_validation,
                      "train_xy_rmse": math.sqrt(sum(train_losses)/len(train_losses)),
                      "validation_xy_rmse": math.sqrt(validation_mse),
                      "train_covariance": train_covariance, "validation_covariance": validation_covariance,
                      "optimizer_steps": trainer.optimizer_steps-previous_steps,
                      "feature_cache_hits": trainer.cache_hits, "feature_cache_misses": trainer.cache_misses,
                      "train_gradients":train_diagnostics['gradients'],
                      **({'frozen_parent_weights_and_buffers_unchanged':True} if frozen_parent is not None else {}),
                      "train_reference_quality":train_diagnostics['gyro_reference_quality'],
                      "validation_reference_quality":validation_diagnostics['gyro_reference_quality'],
                      'early_stopping_bad_epochs':early_stopping.bad_epochs,
                      'early_stopping_meaningful_best_mse':early_stopping.meaningful_best,
                      'early_stopping_triggered':should_stop,
                      "train_seconds": training_finished-epoch_start,
                      "validation_seconds": validation_finished-training_finished,
                      "checkpoint_seconds": perf_counter()-validation_finished,
                      "epoch_seconds": perf_counter()-epoch_start}
            log_epoch(output_dir, record)
            if epoch+1 in args.milestone_epochs:
                milestones=output_dir/'milestones';milestones.mkdir(exist_ok=True)
                path=milestones/f'epoch_{epoch+1:03d}.ckpt'
                save_checkpoint(path,wrapper,optimizer,epoch+1,validation_mse,metadata)
                if args.milestone_report_root:
                    from ekf.scripts.strong_regularization import milestone
                    milestone(args.milestone_report_root.resolve(),epoch+1,args.device)
            if should_stop:
                print(f'Early stopping at epoch={epoch+1}: no meaningful validation MSE improvement '
                      f'for {args.early_stopping_patience} epochs.',flush=True)
                break
        summary = {'epochs_completed':epoch+1,'maximum_epochs':start_epoch+args.epochs,
            'additional_epochs_completed':epoch+1-start_epoch,'starting_epoch':start_epoch,
            'restored_adam_steps':restored_steps,'total_optimizer_steps':restored_steps+trainer.optimizer_steps,
            'early_stopped':should_stop,'early_stopping':metadata['early_stopping'],
            'best_validation_xy_mse':best_validation,'optimizer_steps':trainer.optimizer_steps,
            'checkpoint_selection_split':'validation','held_out_eval_used':False}
        (output_dir/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == "__main__":
    main()
