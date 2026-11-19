# Differentiable EKF

| Location | Purpose |
| --- | --- |
| `eskf.py` | State, covariance propagation and measurement updates |
| `compiled.py` | Compiled prediction and measurement steps |
| `training.py`, `runner.py`, `parallel.py` | Covariance-head training and replay workers |
| `data.py`, `experiment.py` | Flight preparation and experiment metadata |
| `tests/` | EKF filter, data, height and gradient regression checks |
| `scripts/prepare_ekf_run.py` | Freeze and validate Stage 2 inputs for an EKF run |
| `benchmarks/benchmark_eskf.py` | Replay timing and CPU/CUDA comparison |

The root `tests/` folder contains general model/dataset checks, rather than EKF
checks. Shared container and Stage 1/2 preparation tools live under
[`jobs/utils/`](../jobs/utils/). PBS and shell launchers and scheduler logs are
grouped under [`jobs/`](../jobs/README.md), with dedicated EKF subfolders.

Run from the repository root, using the configured Python environment:

```bash
python -m unittest discover -s ekf/tests -v
python -m ekf.scripts.prepare_ekf_run --help
python -m ekf.benchmarks.benchmark_eskf --help
```

For the project container, the shared runner supplies the repository imports
and dependency compatibility used by the PBS jobs:

```bash
python3 -s jobs/utils/container_python.py -m unittest discover -s ekf/tests -v
python3 -s jobs/utils/container_python.py -m ekf.scripts.prepare_ekf_run --help
```

The public training and evaluation entry points remain `train_eskf.py` and
`python -m evaluation.evaluate_eskf`, alongside the other training/evaluation
entry points. See the [integration guide](../doc/differentiable_eskf.md) for filter
behavior and the [performance document](../doc/ekf_training_performance.md) for
timing evidence.
