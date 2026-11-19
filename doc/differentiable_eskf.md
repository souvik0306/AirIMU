# Differentiable ESKF for covariance-only AirIMU training

`ekf.DifferentiableESKF` implements the fixed Float64 15-dimensional error-state
filter. `ekf.AirIMUESKF` connects the existing CodeNet covariance heads to a
complete flight replay. The existing `train.py`, `train_motion_finetune.py`,
dataset transformations, and correction-based training paths are unchanged.

For a plain-language explanation of the training bottleneck, the changes that
made replay faster, and measured A40 results, see
[the EKF performance document](ekf_training_performance.md).

Training now enables height aiding before and after cutoff by default: clean Vicon Z is used
at every EKF estimate timestamp, with a 0.1 m standard deviation (variance 0.01).
The completed jobs 1586065 and 1589404 predate this correction and did not use
height. Their saved checkpoints, exports and results have not been changed.
The updated CPU suite passed 51 tests, including pre-cutoff height effects and
post-cutoff covariance-head gradients; three CUDA tests require a GPU.
Job 1591704 benchmarks the new behavior for two epochs from the October Stage 2
best checkpoint, with frozen inputs under
`experiments/20261119_140000_SGT_height_estimate_2epoch`. Its CUDA checks and
real-flight post-cutoff gradient comparison run before training; CPU checks
alone do not establish its GPU correctness or epoch time.

## Filter and frame conventions

- Nominal state: position, velocity, Hamilton **wxyz** quaternion, accelerometer
  bias, gyroscope bias. The quaternion rotates **body to navigation**.
- Error indexing: position `0:3`, velocity `3:6`, right attitude error `6:9`,
  accelerometer bias `9:12`, gyroscope bias `12:15`.
- Right attitude error: `R_true = R_nominal Exp(delta_theta)`; injection is
  `q_new = q_old * Exp(delta_theta)`. The covariance reset uses the SO(3) right
  Jacobian of the injected correction, including its dependence on that correction.
- Default navigation is z-up, with gravity `[0, 0, -9.80665]` m/s². Pass a gravity
  vector for another navigation convention or measured gravity magnitude.
- Inputs are the original body-frame specific force (m/s²) and angular rate
  (rad/s). Do not subtract gravity, apply learned corrections, or subtract a fixed
  bias from these arrays. The EKF subtracts its evolving bias estimates internally.
- All IMU, pose, velocity, and bias inputs must already use matching body and
  navigation conventions. No automatic frame conversion is performed. PyPose
  quaternion arrays elsewhere in the repository use **xyzw** and must be reordered.

The discrete transition is the first derivative of the implemented nominal
integration. In particular, acceleration uses the **prior** attitude, and both
position and velocity contain attitude and accelerometer-bias sensitivity.
Attitude propagation uses `Exp(-omega_corrected * dt)` in its error transition
and `-J_right(omega_corrected * dt) * dt` for gyro-bias sensitivity.

AirIMU supplies six **sample variances**, ordered `[acc_x, acc_y, acc_z, gyro_x,
gyro_y, gyro_z]`. They are neither standard deviations nor continuous noise
densities. IMU increment covariance scales as `variance * dt²`; fixed bias random
walk covariance scales as `density² * dt`. The noise map also accounts for the
position contribution from acceleration integration. Each segment between IMU
boundaries and Vicon timestamps uses its own integration interval and the held
sample's variance. This segment noise model is shared by training and evaluation;
it does not maintain correlations of sample noise across split segments.

The filter has **no parameters**. Fixed priors are 0.10 m position, 0.20 m/s
velocity, 5° attitude, 0.10 m/s² accelerometer bias, and 0.020 rad/s gyro bias
(standard deviations). Bias random-walk densities are 0.002 m/s²/√s and
0.0002 rad/s/√s. Fixed Vicon standard deviations are 0.01 m, 0.05 m/s, and 2°.
Innovation solves use Cholesky with `1e-9` jitter; updates use Joseph form.

## Vicon aiding and cutoff

Before cutoff, enabled Vicon channels are **horizontal position, horizontal
velocity, yaw, and height**. Position updates use XY, with a separate scalar
height update for Z. At and after cutoff, XY position, velocity and yaw updates
are disabled, while height continues. Height uses clean aligned Vicon Z in the same translated
navigation frame as the initialized filter; no artificial noise is added.
Height standard deviation is 0.1 m, so `height_R=[[0.01]]`. The measurement
Jacobian selects only position-z; cross-covariances allow its correction to
affect velocity, attitude and biases as well. Yaw uses a wrapped residual and a
Jacobian in right attitude-error coordinates.

The runner consumes chronological `ViconEvent(timestamp, observation_type,
observation)` entries, with one event per enabled channel per original observation.
It splits IMU propagation at events inside an IMU interval. Each event is fused
once at its timestamp, using nominal propagation and Jacobians from the same
prior state. Duplicate channel/timestamp pairs and out-of-order events are rejected.

Position/velocity/yaw events require `timestamp < cutoff_timestamp`; height
events are accepted before and after cutoff. The loader supplies one height
event per estimate step after initialization, including an interpolated terminal
height if replay ends between original samples. At cutoff there is no state
initialization, covariance replacement, or graph truncation. The output
contains one row per IMU timestamp, including the initial state:

```python
{
    "position": ...,             # [N, 3]
    "velocity": ...,             # [N, 3]
    "quaternion": ...,           # [N, 4], wxyz
    "accelerometer_bias": ...,   # [N, 3]
    "gyroscope_bias": ...,       # [N, 3]
    "covariance": ...,           # [N, 15, 15]
}
```

The loss is the mean of the squared horizontal vector error (sum x/y, then mean
over timestamps) on the inclusive interval `[cutoff+5, cutoff+25]`. Ground truth
must cover that interval. Covariance predictions **before** cutoff can affect
post-cutoff velocity through aided corrections to velocity, attitude, and bias.
Covariance predictions **at or after** cutoff also affect the Kalman gains of
height updates and can therefore affect horizontal velocity through covariance
coupling. Regression tests verify nonzero post-cutoff gradients, agreement with
finite differences and gradients reaching real covariance heads with pre-cutoff
predictions detached. A fully decoupled or zero-innovation example need not have
a nonzero horizontal gradient. Without height aiding or another post-cutoff
measurement, the direct velocity-loss gradient to post-cutoff variances remains
zero, even though covariance is still propagated.

## Sensor metadata and flight files

The preferred input is the finalized `initial_sensor_biases` directory. Each
`initial_sensor_bias_*.csv` table must contain:

```text
bag_file,ulg_file,group,flight_number,cutoff_rosbag_s,cutoff_ros_epoch_s,
bias_samples,accel_bias_x_mps2,accel_bias_y_mps2,accel_bias_z_mps2,
gyro_bias_x_radps,gyro_bias_y_radps,gyro_bias_z_radps
```

The loader matches `group` and `flight_number` to the dataset flight id, maps the July
dataset's `UN` group to `AI`, initializes the EKF bias state from the six bias
columns, and uses `cutoff_ros_epoch_s` directly in the IMU timestamp reference.
`cutoff_rosbag_s` is retained as provenance and is never guessed or rebased.

The ESKF reads the existing TLab HOCON dataset config directly. Its `train`,
`test`, and `eval` sections become the ESKF training, validation, and test splits.
Each selected `data_root/data_drive` directory requires:

| File | CSV columns |
| --- | --- |
| `imu.csv` | `time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z` |
| `data.csv` | `time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_w,quat_x,quat_y,quat_z` |

All timestamps are seconds in the **same reference**. The standard aligned
`data.csv` supplies Vicon pose and velocity for initialization, pre-cutoff aiding,
height aiding throughout replay, and the post-cutoff velocity target.

The loader reuses the standard TLab loader and initializes position, velocity,
and attitude from its aligned first selected row. Initial accelerometer and
gyroscope biases come from `initial_sensor_biases`. No bias is pre-subtracted from IMU.

Flights must cover the complete evaluation interval and contain aiding after
initialization. The loader ends replay at exactly `cutoff+25`, inserting a terminal
output timestamp inside the last held IMU interval if necessary. Vicon after
cutoff supplies Z-only height aiding when enabled; horizontal velocity remains
the loss target and post-cutoff XY position, XY velocity and yaw are never fused.

## Run training and evaluation

Use the existing PyTorch environment with the repository requirements installed:

```bash
python train_eskf.py \
  --config configs/exp/TLab/finetune_motion_body_stage2.conf \
  --ckpt path/to/matching_airimu_checkpoint.ckpt \
  --dataset-config configs/datasets/TLab/tlab_body.conf \
  --sensor-metadata-root initial_sensor_biases \
  --output experiments/eskf_run --device cuda:0 --epochs 10
```

`AirIMUESKF` freezes the backbone and correction heads, extracts fixed features,
and trains the loaded covariance heads. Untouched raw IMU goes into the ESKF.
Nine repeated leading samples provide CNN context without GT attitude. The
network runs in Float32; conversion to Float64 preserves covariance gradients.
Each flight retains its complete aided-prefix graph. The CLI defaults to one
optimizer step per flight; validation horizontal MSE selects the best checkpoint.

Outputs include `best_model.ckpt`, `newest.ckpt`, `metrics.jsonl`, and
`eskf_metadata.json`, with cutoff/channel/frame settings recorded for review.
The checkpoint's `model_state_dict` can be loaded by the existing AirIMU tools.

Training also writes `flights.jsonl` and prints each flight's start, index, loss,
and replay timings. Batch logs time GPU heads, head backward and Adam. Both training
and validation report `acc_cov` and `gyro_cov` mean/min/max arrays in x/y/z order.
These are predicted **per-sample variances**, not standard deviations or EKF state
covariances. Training statistics describe the predictions used for that flight,
before its optimizer update. Epoch summaries combine means weighted by the number
of IMU samples and retain the minimum/maximum across all flights in each split.

Replay validates complete input arrays up front, transfers Vicon observations and
integration intervals in batches, and computes fixed observed yaw once. During
training and validation, `return_covariance=False` omits covariance history only.
State covariance is propagated through every replayed interval, including after
cutoff, using the six predicted IMU variances and fixed bias random walk. Height
updates and the full velocity-loss gradient remain connected through cutoff.
The default Python interface and the evaluator still return complete covariance
trajectories.

`--device` selects the AirIMU network device. The optional `--ekf-device cpu` keeps
the small Float64 filter operations on CPU while the network can remain on GPU;
the variance transfer preserves autograd. Without this override, parallel replay
uses CPU and serial replay uses `--device`.
The shell/PBS training workflows expose the filter choice as `EKF_DEVICE`.

For several CPU cores plus one GPU, the trainer can batch covariance prediction
on the GPU while independent CPU processes replay complete flights:

```bash
python train_eskf.py \
  --config configs/exp/TLab/4th_Nov_Scratch_stage2.conf \
  --ckpt path/to/stage2/best_model.ckpt \
  --dataset-config configs/datasets/TLab/tlab_body.conf \
  --sensor-metadata-root initial_sensor_biases --output experiments/eskf_parallel \
  --device cuda:0 --ekf-device cpu --replay-workers 4 --flight-batch-size 4 \
  --cpu-threads 1 --feature-cache-mb 512
```

Workers use one PyTorch thread each and return the exact full-flight
`d(loss)/d(variance)`. The main process applies these gradients to the GPU heads
and runs Adam. Validation reuses the workers without building an autograd graph.

`--flight-batch-size` is the number of complete-flight losses **averaged per Adam
step**. Larger batches change update frequency from the original one-step-per-flight
training; they do not truncate flights, omit IMU/aiding samples, or approximate
the gradient. Batch size 1 and `--replay-workers 0` retain the original update
frequency. Use a batch size at least as large as the worker count to use all
workers in a batch.

Frozen CNN/GRU features and detached correction features are cached on CPU under
`--feature-cache-mb` (default 512 MiB); 0 disables caching. Trainable covariance
heads are recomputed after Adam updates. The next batch's fixed features are
prepared while workers replay the current batch. Logs report worker CPU time,
peak memory, replay time, head timings, and cache size.

The existing 36-core/250-GB PBS workflow defaults to four replay workers and four
flights per batch. Override `EKF_REPLAY_WORKERS`, `EKF_FLIGHT_BATCH_SIZE`,
`EKF_CPU_THREADS`, and `EKF_FEATURE_CACHE_MB` as needed.
Each simultaneous replay retains its own full autograd graph, so scale worker
count using observed worker peak memory as well as available CPU cores.
`AIRIMU_CKPT=/path/to/stage2/best_model.ckpt` skips the completed AirIMU stages.
The direct shell workflow keeps zero replay workers by default for smaller hosts.

Evaluate learned covariance or a fixed-noise baseline:

```bash
python -m evaluation.evaluate_eskf \
  --config configs/exp/TLab/finetune_motion_body_stage2.conf \
  --ckpt experiments/eskf_run/best_model.ckpt \
  --dataset-config configs/datasets/TLab/tlab_body.conf \
  --sensor-metadata-root initial_sensor_biases \
  --output experiments/eskf_run/evaluation --split test

python -m evaluation.evaluate_eskf \
  --accel-variance 0.001 --gyro-variance 0.001 \
  --dataset-config configs/datasets/TLab/tlab_body.conf \
  --sensor-metadata-root initial_sensor_biases \
  --output experiments/eskf_fixed/evaluation --split test
```

Both commands accept `--gravity GX GY GZ`. Set it to the selected flight navigation
convention and gravity magnitude, with matching recorded poses and velocities.
Fixed baseline variances are configurable per-sample values.

Evaluation saves full state/bias/covariance trajectories and reports horizontal
EKF–Vicon error on the evaluation interval.

`train_eskf.py` defaults to `--height-updates`; `--no-height-updates` explicitly
selects the old no-height aiding policy. Covariance propagation continues in
either mode. `evaluation.evaluate_eskf` defaults to the policy saved in the
checkpoint, treating older checkpoints without that metadata as no-height.
Pass `--height-updates` explicitly when comparing an original Stage 2 model with
a height-trained model under the new filter policy. Fixed-noise evaluation
defaults to height aiding. The low-level constructor keeps legacy defaults;
use `DifferentiableESKF(height_updates=True)` with explicit height events.
Metadata and per-flight logs record the height policy and number of updates.

It also generates velocity and covariance time-series plots under `plots/`,
with a browsable `plots/index.html`. Every flight has separate x/y/z figures
and a combined overview. Panels show EKF and Vicon velocity, predicted IMU
acceleration and gyro variances, and EKF velocity-state variance. The velocity
plot includes a ±2-standard-deviation band from the EKF covariance. A dashed
line marks cutoff and shading marks the scored cutoff+5 to cutoff+25 interval.
These figures use the actual saved EKF trajectory, rather than the open-loop
AirIMU correction trajectory produced by `evaluation/evaluate_state.py`.

Regenerate plots from an existing evaluation without rerunning the filter:

```bash
python -m evaluation.plot_eskf --evaluation experiments/eskf_run/evaluation
```

## Python interface and acceptance checks

```python
import torch
from ekf import DifferentiableESKF, horizontal_velocity_loss

ekf = DifferentiableESKF(height_updates=True).to(device)
state, P = ekf.initialize(vicon_initial_position, vicon_initial_velocity,
                          vicon_initial_quaternion_wxyz)
imu_variance = torch.full((len(imu_time), 6), .001,
                          dtype=torch.float64, device=device, requires_grad=True)
result = ekf.run_sequence(imu, imu_time, imu_variance, vicon_events,
                           cutoff, state, P)
mask = (imu_time >= cutoff + 5) & (imu_time <= cutoff + 25)
loss = horizontal_velocity_loss(result["velocity"], vicon_velocity_gt, mask)
loss.backward()
assert imu_variance.grad is not None
assert torch.isfinite(imu_variance.grad).all()
print(imu_variance.grad[imu_time < cutoff].norm())
print(imu_variance.grad[imu_time >= cutoff].norm())  # height updates can provide a gradient
```

`imu` is `[N,6]` ordered acceleration then gyro, or `{"acc": acc, "gyro": gyro}`.
The function returns the dictionary above, rather than the incomplete tuple
example in the original requirements. Variance may have N rows or N−1 rows; the
last IMU sample has no propagation interval.
Here `vicon_events` must include height observations before and after cutoff. The data
loader generates these automatically at the estimate timestamps.

```bash
python -m unittest discover -s tests -v
```

The suite covers initialization, gravity, quaternion and bias propagation,
covariance symmetry/PSD and units, finite-difference transition/reset/yaw
Jacobians, event splitting and cutoff continuity, autograd/finite-difference
agreement, pre/post-cutoff sensitivity, real CodeNet covariance-head gradients,
frozen weights/buffers, aligned TLab loading, and sensor-metadata matching.
It includes a full 200 Hz replay through cutoff+25 seconds with backpropagation.

## Compiled replay and A40 performance benchmark

The launch scripts select `--replay-backend compiled`. It compiles an entire
prediction plus the four sequential pre-cutoff updates (XY position, XY velocity,
yaw, height), and a prediction plus scalar height update after cutoff, including their backward
pass. Repeated geometry is shared, diagonal process noise avoids a dense matrix
multiply, and variance rows are unbound once before the loop. Full-flight
Float64 gradients, update order, aiding rate, cutoff, and the loss remain intact.
Covariance propagation no longer switches to nominal-only propagation after
cutoff. The asynchronous reference implementation remains available with
`--replay-backend eager` (or `EKF_REPLAY_BACKEND=eager` in the launch scripts).
Compiled replay requires the aligned TLab position/velocity_xy/yaw schedule; it
rejects other schedules explicitly.

CPU replay workers cache up to 16 prepared flights each, independently of the
bounded frozen-feature cache. Training graphs are never cached. GPU Cholesky
status is checked at the end of each replay rather than synchronizing after
every measurement update. Compilation is a one-time cost per kernel variant;
first-epoch timing includes worker compilation and cache loading.

To compare the compiled CPU and A40 paths, check a complete real flight against
the reference loss and variance gradient, and train **two complete epochs**:

```bash
qsub airimu_ekf_benchmark.pbs
```

The benchmark uses the free GPU queue and an existing Stage 2 checkpoint. It
now exercises height aiding and checks the real-flight post-cutoff variance
gradient. The published timings for completed job 1586065 describe the older
no-height version and must be remeasured for the current behavior. It
requests 8 CPUs, 64 GB RAM and one GPU. The learning rate, seed and four-flight
Adam batches match the earlier EKF run. The completed job 1586065 used the
expanded October dataset: 82 training, 23 validation and 12 held-out flights,
whereas the earlier 31-minute run used 51 training and 14 validation flights.
The epoch-time comparison therefore uses different flight inventories; the
same-flight replay comparison isolates the backend change. `AIRIMU_CKPT`, `EKF_CONFIG`,
`EKF_DATASET_CONFIG`, `SENSOR_METADATA_ROOT`, and `EKF_RUN_ROOT` can override inputs.
It selects GPU replay only when its measured single-flight throughput exceeds
four parallel CPU workers; otherwise the neural network stays on the A40 and
the filter uses four CPU workers. The synthetic comparison estimates throughput;
`training/metrics.jsonl` contains the measured whole-epoch times.
CPU and CUDA parity checks run before benchmarking and cover both backward
gradients and validation with gradients disabled.

By default the benchmark starts from
`experiments/20261116_151808_SGT_bias_no_cov_limits/scratch/stage2/ckpt/best_model.ckpt`.
It uses `configs/exp/TLab/ekf_no_cov_limits.conf` to retain that model's saved
`use_cov_limits=False` setting. It does not rerun Stage 1 or Stage 2.

Each flight uses its own initial accelerometer and gyro biases from the matching
sensor-metadata CSV row, in m/s² and rad/s. They initialize the EKF bias states;
the raw IMU supplied to the network and filter is unchanged. The filter subtracts
its current bias during propagation, refines the bias during Vicon aiding, and
retains it through cutoff without reinitialization. Vicon XY position, XY velocity
and yaw are fused only before cutoff. Clean Vicon Z supplies a height update
at every estimate step both before and after cutoff, with measurement standard deviation
0.1 m. Covariance propagation continues through the replay. Horizontal velocity
ground truth from cutoff+5 to cutoff+25 seconds
supervises covariance-head weights through the complete replay. This objective
tunes covariance for velocity accuracy; it does not directly supervise absolute
variance calibration. Training, validation and held-out test flights are
disjoint; the held-out flights do not enter training or model selection. The
selected dataset snapshot and metadata record the actual flight inventory.

Results are written to `experiments/<timestamp>_job<id>_ekf_benchmark/`:

- `replay_benchmark.json`: compilation, warmed timings and real-flight parity.
- `job.log`: backend checks and training progress.
- `gpu_usage.csv`: utilization, memory and power sampled every ten seconds.
- `training/metrics.jsonl` and `training/flights.jsonl`: epoch and flight timings.
- `training/best_model.ckpt` and `training/newest.ckpt`: new benchmark weights.

Every epoch prints readable `min`, `avg`, and `max` predicted IMU variances for
accelerometer and gyroscope x/y/z axes, separately for training and validation.
The average weights each IMU sample equally across flights of different lengths.
The same values are saved as `min`, `mean`, and `max` inside `train_covariance`
and `validation_covariance` in `training/metrics.jsonl`. Epoch losses and elapsed
time are printed alongside them. Terminal output is captured in `job.log` and
the PBS output log.

The PBS launcher uses `python3 -s scripts/container_python.py` to give the
container's matched PyTorch/Triton packages priority over user-installed packages.
User dependencies such as PyPose and HOCON remain available after the container
packages. This avoids the incompatible Triton `AttrsDescriptor` error observed
when compiling with the default mixed environment.
