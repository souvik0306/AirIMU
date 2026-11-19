"""Compare replay backends on the allocated node before a training benchmark."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import resource
import torch
from ekf import DifferentiableESKF, ViconEvent
from ekf.data import apply_sensor_metadata, prepare_flight, read_dataset_flights
from ekf.experiment import filter_metadata, load_airimu
from ekf.parallel import replay_velocity


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def synthetic():
    ekf = DifferentiableESKF(height_updates=True)
    state, P = ekf.initialize([0., 0., 0.], [.2, -.1, 0.], [1., 0., 0., 0.])
    n = 501
    times = torch.arange(n, dtype=torch.float64)*.005
    imu = torch.tensor([.1, -.04, 9.80665, .005, -.008, .01], dtype=torch.float64).repeat(n, 1)
    events = []
    for t in times[1:250].tolist():
        events.extend((ViconEvent(t, 'position', torch.tensor([.2*t, -.1*t], dtype=torch.float64)),
                       ViconEvent(t, 'velocity_xy', torch.tensor([.2, -.1], dtype=torch.float64)),
                       ViconEvent(t, 'yaw', torch.tensor(.01*t, dtype=torch.float64))))
    events.extend(ViconEvent(t, 'height', torch.zeros(1, dtype=torch.float64))
                  for t in times[1:].tolist())
    events.sort(key=lambda event: float(event.timestamp))
    return dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:], imu_time=times,
                vicon_events=events, cutoff_timestamp=float(times[250]), initial_state=state,
                initial_covariance=P)


def micro(device):
    from ekf.compiled import velocity_sequence
    flight = synthetic()
    ekf = DifferentiableESKF(height_updates=True).to(device)
    rows = []
    for iteration in range(2):
        variance = torch.tensor([.005]*3+[1e-5]*3, dtype=torch.float64,
                               device=device).repeat(501, 1).requires_grad_()
        sync(device); started = perf_counter()
        result = velocity_sequence(ekf, flight, variance)
        sync(device); forwarded = perf_counter()
        loss = (result[-50:, :2]-result.new_tensor([.2, -.1])).square().sum(-1).mean()
        gradient, = torch.autograd.grad(loss, variance)
        sync(device); finished = perf_counter()
        row = dict(device=device, iteration=iteration, forward=forwarded-started,
                   backward=finished-forwarded, total=finished-started, loss=float(loss.detach()))
        print(json.dumps({'microbenchmark':row}), flush=True)
        rows.append(row)
        if not torch.isfinite(gradient).all(): raise RuntimeError('Non-finite benchmark gradient')
        del result, gradient, loss, variance
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset-config', required=True)
    parser.add_argument('--sensor-metadata-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
                      'cpu_affinity':len(__import__('os').sched_getaffinity(0))}), flush=True)
    cpu = micro('cpu')
    gpu = micro('cuda:0')
    # Four flights/Adam step preserves the existing optimizer batching. The
    # compiler uses far less RAM, so CPU workers need a much smaller allocation.
    workers, batch_size = 4, 4
    use_cuda = gpu[-1]['total'] < cpu[-1]['total']/workers
    runtime = dict(replay_backend='compiled', replay_workers=0 if use_cuda else workers,
                   flight_batch_size=batch_size, ekf_device='cuda:0' if use_cuda else 'cpu')
    print(json.dumps({'selected_runtime':runtime}), flush=True)
    flights, cutoffs = apply_sensor_metadata(read_dataset_flights(args.dataset_config),
                                             args.sensor_metadata_root)
    record = next(r for r in flights if r['split']=='train')
    ekf = DifferentiableESKF(height_updates=True)
    wrapper = load_airimu(args.config, args.ckpt, ekf, 'cuda:0')
    if wrapper.network.use_cov_limits:
        raise ValueError('This benchmark requires covariance minimums only: use_cov_limits must be False.')
    covariance_policy = dict(acc_variance_min=float(wrapper.network.acc_cov_min),
                             gyro_variance_min=float(wrapper.network.gyro_cov_min),
                             variance_max=None)
    print(json.dumps({'covariance_policy':covariance_policy}), flush=True)
    flight = prepare_flight(record, cutoffs[record['id']], ekf)
    with torch.no_grad():
        variance = wrapper.predict_variance(flight['raw_acceleration'], flight['raw_gyroscope'])
    comparisons = []
    reference = None
    for backend in ('eager', 'compiled'):
        started = perf_counter()
        result = replay_velocity(ekf, flight, variance, backward=True, backend=backend)
        row = dict(backend=backend, wall_seconds=perf_counter()-started, loss=result['loss'],
                   timing_seconds=result['timing_seconds'],
                   peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        post = flight['imu_time'] >= flight['cutoff_timestamp']
        row['post_cutoff_variance_gradient_norm'] = float(result['gradient'][post].norm())
        if row['post_cutoff_variance_gradient_norm'] <= 0:
            raise RuntimeError('Height-aided real-flight loss has no post-cutoff variance gradient.')
        if reference is None: reference=result
        else:
            torch.testing.assert_close(torch.tensor(result['loss'],dtype=torch.float64),
                                       torch.tensor(reference['loss'],dtype=torch.float64),rtol=1e-7,atol=1e-8)
            torch.testing.assert_close(result['gradient'],reference['gradient'],rtol=1e-5,atol=1e-7)
            row['max_gradient_error']=float((result['gradient']-reference['gradient']).abs().max())
        comparisons.append(row)
        print(json.dumps({'full_flight':record['id'],**row}),flush=True)
    report=dict(runtime=runtime, cpu_micro=cpu, gpu_micro=gpu, full_flight=comparisons,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                covariance_policy=covariance_policy, filter_policy=filter_metadata(ekf))
    (output/'replay_benchmark.json').write_text(json.dumps(report,indent=2))
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2))


if __name__ == '__main__': main()
