#!/bin/bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Usage: train.sh TASK SETTING DEMOS SEED GPU [HYDRA_OVERRIDES...]
bash "$POLICY_ROOT/scripts/train_trajectory.sh" dssp "$1" "$2" "$3" paper "$4" "$5" "${@:6}"
