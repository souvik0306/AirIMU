# NUS Vanda HPC AirIMU Usage Guide

This guide contains the main commands needed to run AirIMU training on the NUS Vanda A40 GPU system.

# Section 1. Login to Vanda

From your local terminal:

```bash
ssh e1349884@vanda.nus.edu.sg
```

After login:

```bash
hostname
```

A login node may appear as:

```text
stdct-login-01
```

Use the login node to prepare code, submit PBS jobs, check job status, and inspect logs.

# Section 2. Enter the AirIMU repository

```bash
cd ~/AirIMU
```

Check the repository:

```bash
git status
```

Pull the latest changes when needed:

```bash
git pull
```

Check the important AirIMU files:

```bash
ls
```

Check the TLab dataset:

```bash
ls ./T-Lab_31st_July_dataset
```

Check the pretrained model:

```bash
ls -lh ./AirIMU_EuRoC/best_model.ckpt
```

# Section 3. Apptainer

Load Apptainer:

```bash
module load apptainer/1.4.1
```

Define the Vanda PyTorch container:

```bash
image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif
```

Check the container:

```bash
ls -lh $image
```

Run a command inside the container:

```bash
apptainer exec -e $image COMMAND
```

Run a GPU command inside a PBS GPU job:

```bash
apptainer exec --nv -e $image COMMAND
```

The `--nv` option exposes the NVIDIA GPU to the container.

# Section 4. W&B

W&B login only needs to be done once unless the credentials change.

Load Apptainer and define the image:

```bash
module load apptainer/1.4.1
image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif
```

Login:

```bash
apptainer exec -e $image python3 -m wandb login
```

Check W&B status:

```bash
apptainer exec -e $image python3 -m wandb status
```

The W&B credentials are stored in your home directory and can be used by later PBS jobs.

# Section 5. AirIMU PBS training file

From:

```bash
cd ~/AirIMU
```

use this PBS file:

```bash
#!/bin/bash

#PBS -q auto_free
#PBS -j oe
#PBS -N airimu_finetune
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=36:mem=250gb:ngpus=1

set -e

cd $PBS_O_WORKDIR

module load apptainer/1.4.1

image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif

echo "Working directory:"
pwd

echo "Compute node:"
hostname

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "Starting AirIMU finetuning"

apptainer exec --nv -e $image \
python3 -u train_motion_finetune.py \
--config configs/exp/TLab/finetune_motion_body.conf \
--device cuda:0

echo "AirIMU finetuning finished"
```

Save this as:

```text
airimu_finetune.pbs
```

Important PBS lines:

```bash
#PBS -q auto_free
```

Uses the confirmed free queue.

```bash
#PBS -j oe
```

Combines normal output and errors into one log file.

```bash
#PBS -N airimu_finetune
```

Sets the job name.

```bash
#PBS -l walltime=24:00:00
```

Sets the maximum runtime.

```bash
#PBS -l select=1:ncpus=36:mem=250gb:ngpus=1
```

Requests 36 CPU cores, 250 GB RAM, and one GPU.

```bash
cd $PBS_O_WORKDIR
```

Runs the job from the directory where `qsub` was executed.

# Section 6. Submit AirIMU training

Always submit from the AirIMU repository:

```bash
cd ~/AirIMU
```

Submit:

```bash
qsub airimu_finetune.pbs
```

PBS returns a job ID similar to:

```text
1301505.stdct-mgmt-02
```

# Section 7. Check job status

Show active jobs:

```bash
qstat
```

Typical states:

```text
Q    Queued
R    Running
F    Finished
```

Show active and finished jobs:

```bash
qstat -x
```

Check one job:

```bash
qstat -x JOB_ID
```

Example:

```bash
qstat -x 1301505
```

Show detailed information:

```bash
qstat -fx JOB_ID
```

Example:

```bash
qstat -fx 1301505.stdct-mgmt-02
```

Useful information includes the compute node, resources, job state, runtime, and exit status.

# Section 8. Live AirIMU logs

From:

```bash
cd ~/AirIMU
```

Watch the newest AirIMU log live:

```bash
tail -f $(ls -t airimu_finetune.o* | head -1)
```

This shows new output as training runs.

To stop watching the log:

```text
Ctrl C
```

This only stops `tail`.

The PBS training job continues running.

Confirm:

```bash
qstat
```

Return to the live log later:

```bash
tail -f $(ls -t airimu_finetune.o* | head -1)
```

Read the final log after the job finishes:

```bash
cat $(ls -t airimu_finetune.o* | head -1)
```

List AirIMU logs:

```bash
ls -ltr airimu_finetune.o*
```

# Section 9. Check the assigned GPU

The PBS script already prints GPU information.

You can also include:

```bash
nvidia-smi
```

For a shorter output:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

The confirmed Vanda GPU is:

```text
NVIDIA A40
```

# Section 10. Stop a job

Check active jobs:

```bash
qstat
```

Stop a job:

```bash
qdel JOB_ID
```

Example:

```bash
qdel 1301505
```

Check again:

```bash
qstat
```

# Section 11. Check job resource usage

For a completed job:

```bash
qstat -fx JOB_ID | grep -E "Resource_List|resources_used"
```

Example:

```bash
qstat -fx 1301505.stdct-mgmt-02 | grep -E "Resource_List|resources_used"
```

This shows requested and used CPU, RAM, GPU, and walltime information.

# Section 12. Useful NUS HPC commands

Show NUS GPU instructions:

```bash
hpc gpu
```

Show PBS help:

```bash
hpc pbs help
```

Show queues:

```bash
qstat -Q
```

Check project allocation:

```bash
hpc project
```

Check cluster status:

```bash
hpc gstat
```

# Section 13. Normal AirIMU workflow

Login:

```bash
ssh e1349884@vanda.nus.edu.sg
```

Enter AirIMU:

```bash
cd ~/AirIMU
```

Update code when needed:

```bash
git pull
```

Submit training:

```bash
qsub airimu_finetune.pbs
```

Check the job:

```bash
qstat
```

Watch live logs:

```bash
tail -f $(ls -t airimu_finetune.o* | head -1)
```

Stop watching logs with:

```text
Ctrl C
```

Check job state later:

```bash
qstat -x
```

Read the final log:

```bash
cat $(ls -t airimu_finetune.o* | head -1)
```

Check the corresponding W&B run for training curves and recorded metrics.

# Section 14. Quick command reference

```bash
ssh e1349884@vanda.nus.edu.sg
cd ~/AirIMU
git pull
qsub airimu_finetune.pbs
qstat
qstat -x
qstat -fx JOB_ID
tail -f $(ls -t airimu_finetune.o* | head -1)
qdel JOB_ID
```

Apptainer:

```bash
module load apptainer/1.4.1
image=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif
```

W&B:

```bash
apptainer exec -e $image python3 -m wandb login
```

GPU check:

```bash
nvidia-smi
```

The confirmed PBS resource request is:

```bash
#PBS -l select=1:ncpus=36:mem=250gb:ngpus=1
```