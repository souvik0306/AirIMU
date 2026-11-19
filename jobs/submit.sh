#!/usr/bin/env bash
# Usage: jobs/submit.sh JOB.pbs [qsub options...]
set -euo pipefail

if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: jobs/submit.sh JOB.pbs [qsub options...]'
    echo 'Example: jobs/submit.sh airimu_ekf_benchmark.pbs -l walltime=02:00:00'
    if [[ $# -eq 0 ]]; then exit 2; else exit 0; fi
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
job_file=$1
shift
if [[ -f "$job_file" ]]; then
    job_file=$(realpath -- "$job_file")
elif [[ -f "$repo_root/jobs/pbs/$job_file" ]]; then
    job_file=$(realpath -- "$repo_root/jobs/pbs/$job_file")
elif [[ "$job_file" != */* ]]; then
    shopt -s nullglob
    matches=("$repo_root"/jobs/pbs/*/"$job_file")
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "ERROR: Expected one PBS script named $job_file; use its path if ambiguous." >&2
        exit 2
    fi
    job_file=${matches[0]}
else
    echo "ERROR: PBS script not found: $job_file" >&2
    exit 2
fi
if [[ "$job_file" != *.pbs ]]; then
    echo "ERROR: Expected a .pbs script: $job_file" >&2
    exit 2
fi

cd -- "$repo_root"
case "$job_file" in
    "$repo_root"/jobs/pbs/ekf/*.pbs) category=ekf ;;
    "$repo_root"/jobs/pbs/stage12/*.pbs) category=stage12 ;;
    "$repo_root"/jobs/pbs/evaluation/*.pbs) category=evaluation ;;
    *) category=misc ;;
esac
log_dir="$repo_root/jobs/logs/$category"
mkdir -p -- "$log_dir"
exec qsub -o "$log_dir/" -e "$log_dir/" "$@" "$job_file"
