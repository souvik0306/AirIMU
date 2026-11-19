# Job launchers and logs

Jobs are grouped by workflow. EKF launchers and scheduler logs are under `ekf/`
subfolders here; the EKF implementation, tests and preparation tools live in
[`../ekf/`](../ekf/README.md).

```text
jobs/
  pbs/
    ekf/          EKF benchmark and training
    stage12/      Stage 1/2 training and export
    evaluation/   Existing-model inference and state evaluation
  local/
    ekf/          Direct EKF launcher
    stage12/      Direct Stage 1/2 launchers
  logs/
    ekf/          EKF scheduler logs
    stage12/      Stage 1/2 scheduler logs
    evaluation/   Evaluation scheduler logs
  setup/          Environment setup shell scripts
  utils/          Shared container runner and Stage 1/2 preparation
  submit.sh       Submission helper
```

Submit by filename; the helper finds its workflow, submits from the repository
root, and puts scheduler stdout/stderr in the matching log folder. All PBS
launchers retain the `auto_free` queue.

```bash
./jobs/submit.sh airimu_ekf_benchmark.pbs -l walltime=02:00:00
./jobs/submit.sh airimu_ekf_train_export.pbs -v EKF_RUN_ROOT=/absolute/path/to/prepared/run
./jobs/submit.sh airimu_stage12_export.pbs
```

The helper also accepts a path such as `ekf/airimu_ekf_benchmark.pbs` or an
absolute PBS path. Use the helper's absolute path when calling from outside the
repository. PBS scripts outside the workflow folders use `logs/misc/`.

Direct submission is supported **from the repository root**:

```bash
qsub jobs/pbs/ekf/airimu_ekf_benchmark.pbs
```

PBS uses `<job-id>.OU` for directory-only stdout destinations; older moved logs
retain their original filenames. Current launchers join stderr into stdout.
Experiment `job.log`, checkpoints and results remain under `experiments/`.

Local launchers resolve the repository root from their own location:

```bash
./jobs/setup/setup_airimu_ens.sh
./jobs/local/stage12/airimu_scratch.sh
./jobs/local/stage12/airimu_finetune.sh
AIRIMU_CKPT=/absolute/path/to/stage2.ckpt ./jobs/local/ekf/airimu_joint_ekf.sh
```

`setup/setup_airimu.sh` is the HPC/Apptainer setup variant.

| PBS launcher under `pbs/` | Purpose |
| --- | --- |
| `ekf/airimu_ekf_benchmark.pbs` | CPU/CUDA checks, replay benchmark and two EKF epochs |
| `ekf/airimu_ekf_train_export.pbs` | Prepared EKF training, inference, evaluations and export |
| `ekf/airimu_joint_ekf.pbs` | General EKF training and learned/fixed-noise evaluation |
| `stage12/airimu_stage12_export.pbs` | Stage 1, Stage 2, inference, evaluation and export |
| `stage12/airimu_scratch.pbs` | Scratch Stage 1/2 training and downstream outputs |
| `stage12/airimu_finetune.pbs` | Stage 1/2 fine-tuning and downstream outputs |
| `evaluation/evaluate_stage2_ekf.pbs` | Existing-model inference and standard state evaluation |

The [relocation manifest](relocation_manifest.json) maps old paths to current
locations. Archived scripts already inside `experiments/` retain their original
contents and paths.
