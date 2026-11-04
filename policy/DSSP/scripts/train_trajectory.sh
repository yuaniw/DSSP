#!/bin/bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_name=$1
task_name=$2
setting=$3
expert_data_num=$4
run_label=$5
seed=$6
gpu_id=$7
extra_overrides=("${@:8}")
exp_name="${task_name}-${config_name}-${run_label}"
run_dir="$POLICY_ROOT/data/outputs/${exp_name}_seed${seed}"
export HYDRA_FULL_ERROR=1
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
fi
cd "$POLICY_ROOT"
exec "${PYTHON:-python}" -u src/train_dssp_trajectory.py \
    --config-name="${config_name%.yaml}" \
    "task_name=$task_name" "setting=$setting" "expert_data_num=$expert_data_num" \
    "training.seed=$seed" training.device=cuda:0 training.debug=false training.resume=false \
    checkpoint.save_ckpt=true "logging.mode=${WANDB_MODE:-disabled}" \
    "exp_name=$exp_name" "hydra.run.dir=$run_dir" \
    "task.dataset.zarr_path=$POLICY_ROOT/data/${task_name}-${setting}-${expert_data_num}.zarr" \
    "${extra_overrides[@]}"
