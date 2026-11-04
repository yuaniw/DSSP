#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Create and activate policy/DSSP/environment.yaml before running this script.
# Its dependencies replace the upstream requirements.txt installation.
echo "Applying RoboTwin simulator patches ..."
bash script/patch_simulator.sh

echo "Installing PyTorch3D ..."
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"

echo "Installing CuRobo ..."
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
python -m pip install --no-build-isolation -e envs/curobo

echo "RoboTwin simulator installation complete. Download assets with bash script/_download_assets.sh."
