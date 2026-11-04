#!/bin/bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$POLICY_ROOT"
exec "${PYTHON:-python}" scripts/process_data.py "$1" "$2" "$3"
