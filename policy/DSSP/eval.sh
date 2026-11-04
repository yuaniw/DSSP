#!/bin/bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$POLICY_ROOT/../.." && pwd)"
# Usage: eval.sh TASK TASK_CONFIG CKPT_SETTING DEMOS SEED GPU [EVAL_OVERRIDES...]
task_name=$1
task_config=$2
ckpt_setting=$3
expert_data_num=$4
seed=$5
gpu_id=$6
export HYDRA_FULL_ERROR=1
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
fi
cd "$REPO_ROOT"
exec "${PYTHON:-python}" -u script/eval_policy.py \
    --config policy/DSSP/deploy_policy.yml --overrides \
    --policy_name DSSP --config_name dssp \
    --task_name "$task_name" --task_config "$task_config" --ckpt_setting "$ckpt_setting" \
    --expert_data_num "$expert_data_num" --seed "$seed" \
    --mamba_version Mamba1 --checkpoint_num 1000 --use_rgb False \
    --use_cached_mamba_eval True --eval_video_log True --eval_episodes 100 \
    "${@:7}"
