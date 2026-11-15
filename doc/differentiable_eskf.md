# Differentiable ESKF for covariance-only AirIMU training

`ekf.DifferentiableESKF` implements the fixed Float64 15-dimensional error-state
filter. `ekf.AirIMUESKF` connects the existing CodeNet covariance heads to a
complete flight replay. The existing `train.py`, `train_motion_finetune.py`,
dataset transformations, and correction-based training paths are unchanged.

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

The default enabled channels match the supplied flight configuration:
**horizontal position, horizontal velocity, and yaw**. Position z is ignored;
height fusion is disabled. Yaw accepts a scalar angle or quaternion and uses a
wrapped residual and a Jacobian in right attitude-error coordinates.

The runner consumes chronological `ViconEvent(timestamp, observation_type,
observation)` entries, with one event per enabled channel per original observation.
It splits IMU propagation at events inside an IMU interval. Each event is fused
once at its timestamp, using nominal propagation and Jacobians from the same
prior state. Duplicate channel/timestamp pairs and out-of-order events are rejected.

Only events with `timestamp < cutoff_timestamp` are fused. At cutoff there is no
state initialization, covariance replacement, or graph truncation. The output
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
Covariance predictions **at or after** cutoff change covariance only: their
gradient from the primary velocity loss is exactly zero. Shared AirIMU weights
are trained through their pre-cutoff outputs.

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
and the post-cutoff velocity target.

The loader initializes position, velocity, and attitude from pre-cutoff Vicon at
the first IMU timestamp. It uses linear interpolation for position/velocity and
shortest-arc SLERP for attitude, with no extrapolation. Initial accelerometer and
gyroscope biases come from `initial_sensor_biases`. No bias is pre-subtracted from IMU.

Flights must cover the complete evaluation interval and contain aiding after
initialization. The loader ends replay at exactly `cutoff+25`, inserting a terminal
output timestamp inside the last held IMU interval if necessary. Vicon after
cutoff is used only as ground truth.

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

`AirIMUESKF` calls `network.inference()` for covariance only. It freezes the
backbone and correction heads using a local covariance-only policy, retains loaded
covariance-head weights, and passes untouched raw IMU into the ESKF. Nine repeated
leading raw samples provide CNN context without requiring GT attitude. The
network runs in Float32 and the differentiable conversion to Float64 preserves
gradients into its uncertainty heads. Each flight is replayed from initialization
with a complete graph; one optimizer step is taken per flight. Validation runs in
evaluation mode, and the best checkpoint is selected by validation horizontal MSE.
Long aided prefixes consume more memory because their graph is retained.

Outputs include `best_model.ckpt`, `newest.ckpt`, `metrics.jsonl`, and
`eskf_metadata.json`, with cutoff/channel/frame settings recorded for review.
The checkpoint's `model_state_dict` can be loaded by the existing AirIMU tools.

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

## Python interface and acceptance checks

```python
import torch
from ekf import DifferentiableESKF, run_ekf_sequence, horizontal_velocity_loss

ekf = DifferentiableESKF().to(device)
state, P = ekf.initialize(vicon_initial_position, vicon_initial_velocity,
                          vicon_initial_quaternion_wxyz)
imu_variance = torch.full((len(imu_time), 6), .001,
                          dtype=torch.float64, device=device, requires_grad=True)
result = run_ekf_sequence(imu, imu_time, imu_variance, vicon_events,
                          cutoff, state, P, ekf=ekf)
mask = (imu_time >= cutoff + 5) & (imu_time <= cutoff + 25)
loss = horizontal_velocity_loss(result["velocity"], vicon_velocity_gt, mask)
loss.backward()
assert imu_variance.grad is not None
assert torch.isfinite(imu_variance.grad).all()
print(imu_variance.grad[imu_time < cutoff].norm())
print(imu_variance.grad[imu_time >= cutoff].norm())  # exactly zero
```

`imu` is `[N,6]` ordered acceleration then gyro, or `{"acc": acc, "gyro": gyro}`.
The function returns the dictionary above, rather than the incomplete tuple
example in the original requirements. Variance may have N rows or N−1 rows; the
last IMU sample has no propagation interval.

```bash
python -m unittest discover -s tests -v
```

The suite covers initialization, gravity, quaternion and bias propagation,
covariance symmetry/PSD and units, finite-difference transition/reset/yaw
Jacobians, event splitting and cutoff continuity, autograd/finite-difference
agreement, pre/post-cutoff sensitivity, real CodeNet covariance-head gradients,
frozen weights/buffers, input interpolation, and sensor-metadata matching.
It includes a full 200 Hz replay through cutoff+25 seconds with backpropagation.
