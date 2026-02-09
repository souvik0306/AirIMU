import os
import torch
import pypose as pp
from utils import CPU_Unpickler
from datasets import SeqDataset

exp_path = 'experiments/EuRoC/codenet/'
net_result_path = os.path.join(exp_path, 'net_output.pickle')

print("="*80)
print("LOADING MODEL OUTPUT")
print("="*80)

with open(net_result_path, 'rb') as handle:
    inference_state_load = CPU_Unpickler(handle).load()

data_name = list(inference_state_load.keys())[0]
inference_state = inference_state_load[data_name]

print(f"\nSequence: {data_name}")

has_cov = 'acc_cov' in inference_state and 'gyro_cov' in inference_state
print(f"Covariance available: {has_cov}")

if has_cov:
    gyro_cov = inference_state['gyro_cov'][0]
    acc_cov = inference_state['acc_cov'][0]
    print(f"  gyro_cov: {gyro_cov.shape}")
    print(f"  acc_cov: {acc_cov.shape}")

print("\n" + "="*80)
print("LOADING RAW IMU DATA")
print("="*80)

data_conf = type('obj', (object,), {
    'data_root': '/content/AirIMU/EuRoC-Dataset',
    'name': 'Euroc'
})()

dataset = SeqDataset(
    data_conf.data_root, 
    data_name, 
    'cpu', 
    name='Euroc',
    duration=200,
    step_size=200,
    drop_last=False,
    conf={}
)

n_samples = 3
sample_data = dataset[0]

if 'imu' in sample_data:
    acc_raw = sample_data['imu'][:n_samples, :3]
    gyro_raw = sample_data['imu'][:n_samples, 3:]
else:
    acc_raw = sample_data['acc'][:n_samples]
    gyro_raw = sample_data['gyro'][:n_samples]

dt = sample_data['dt'][:n_samples+1]
if dt.dim() > 1:
    dt = dt.squeeze(-1)

print(f"\nFirst {n_samples} samples:")
print(f"\nAccelerometer [m/s^2]:")
for i in range(n_samples):
    print(f"  [{acc_raw[i, 0]:8.4f}, {acc_raw[i, 1]:8.4f}, {acc_raw[i, 2]:8.4f}]")

print(f"\nGyroscope [rad/s]:")
for i in range(n_samples):
    print(f"  [{gyro_raw[i, 0]:8.4f}, {gyro_raw[i, 1]:8.4f}, {gyro_raw[i, 2]:8.4f}]")

print(f"\nTime deltas [s]: {dt[1:n_samples+1].numpy()}")

if has_cov:
    gyro_cov_samples = gyro_cov[:n_samples]
    acc_cov_samples = acc_cov[:n_samples]
    
    print(f"\nModel Covariances:")
    print(f"Gyro cov:")
    for i in range(n_samples):
        print(f"  [{gyro_cov_samples[i, 0]:.6e}, {gyro_cov_samples[i, 1]:.6e}, {gyro_cov_samples[i, 2]:.6e}]")
    
    print(f"\nAcc cov:")
    for i in range(n_samples):
        print(f"  [{acc_cov_samples[i, 0]:.6e}, {acc_cov_samples[i, 1]:.6e}, {acc_cov_samples[i, 2]:.6e}]")

print("\n" + "="*80)
print("IMU PREINTEGRATION")
print("="*80)

init_pos = torch.zeros(3, dtype=torch.float64)
init_rot = pp.identity_SO3(1, dtype=torch.float64)
init_vel = torch.zeros(3, dtype=torch.float64)
gravity = 9.81007

gyro_batch = gyro_raw.unsqueeze(0).double()
acc_batch = acc_raw.unsqueeze(0).double()
dt_batch = dt[1:].unsqueeze(0).unsqueeze(-1).double()

print("\n" + "-"*80)
print("WITHOUT Model Covariances")
print("-"*80)

integrator_no_cov = pp.module.IMUPreintegrator(
    init_pos, init_rot, init_vel,
    gravity=gravity,
    reset=True,
    prop_cov=False
).double()

result_no_cov = integrator_no_cov(dt_batch, gyro_batch, acc_batch)

print(f"\nIntegrated State:")
print(f"  Position: [{result_no_cov['pos'][0, -1, 0]:.6f}, {result_no_cov['pos'][0, -1, 1]:.6f}, {result_no_cov['pos'][0, -1, 2]:.6f}]")
print(f"  Rotation: {result_no_cov['rot'][0, -1].numpy()}")
print(f"  Velocity: [{result_no_cov['vel'][0, -1, 0]:.6f}, {result_no_cov['vel'][0, -1, 1]:.6f}, {result_no_cov['vel'][0, -1, 2]:.6f}]")

if has_cov:
    print("\n" + "-"*80)
    print("WITH Model Covariances")
    print("-"*80)
    
    integrator_with_cov = pp.module.IMUPreintegrator(
        init_pos, init_rot, init_vel,
        gravity=gravity,
        reset=False,
        prop_cov=True
    ).double()
    
    gyro_cov_batch = gyro_cov_samples.unsqueeze(0).double()
    acc_cov_batch = acc_cov_samples.unsqueeze(0).double()
    
    result_with_cov = integrator_with_cov(
        dt_batch, gyro_batch, acc_batch, 
        rot=None, 
        gyro_cov=gyro_cov_batch, 
        acc_cov=acc_cov_batch
    )
    
    print(f"\nIntegrated State:")
    print(f"  Position: [{result_with_cov['pos'][0, -1, 0]:.6f}, {result_with_cov['pos'][0, -1, 1]:.6f}, {result_with_cov['pos'][0, -1, 2]:.6f}]")
    print(f"  Rotation: {result_with_cov['rot'][0, -1].numpy()}")
    print(f"  Velocity: [{result_with_cov['vel'][0, -1, 0]:.6f}, {result_with_cov['vel'][0, -1, 1]:.6f}, {result_with_cov['vel'][0, -1, 2]:.6f}]")
    
    if 'cov' in result_with_cov and result_with_cov['cov'] is not None:
        print(f"\nUncertainty (covariance diagonal):")
        final_cov = result_with_cov['cov'][0]
        diag = final_cov.diagonal().numpy()
        print(f"  Rotation: [{diag[0]:.6e}, {diag[1]:.6e}, {diag[2]:.6e}]")
        print(f"  Velocity: [{diag[3]:.6e}, {diag[4]:.6e}, {diag[5]:.6e}]")
        print(f"  Position: [{diag[6]:.6e}, {diag[7]:.6e}, {diag[8]:.6e}]")
    
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    pos_diff = (result_with_cov['pos'][0, -1] - result_no_cov['pos'][0, -1]).numpy()
    vel_diff = (result_with_cov['vel'][0, -1] - result_no_cov['vel'][0, -1]).numpy()
    
    print(f"\nTrajectory Difference (WITH cov - WITHOUT cov):")
    print(f"  Position: [{pos_diff[0]:.6e}, {pos_diff[1]:.6e}, {pos_diff[2]:.6e}]")
    print(f"  Velocity: [{vel_diff[0]:.6e}, {vel_diff[1]:.6e}, {vel_diff[2]:.6e}]")

print("\n" + "="*80)
