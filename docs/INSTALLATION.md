# Installation

Use Linux with an NVIDIA GPU and the CUDA 12.1 toolkit (`nvcc`). Run the commands below from the DSSP repository root.

## 1. Create the Conda environment

```bash
conda env create -f policy/DSSP/environment.yaml
conda activate RoboTwin-DSSP
export PYTHONNOUSERSITE=1
```

## 2. Set up RoboTwin

Follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for GPU drivers and Vulkan. Then run the simulator installation and asset download steps in this checkout:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4

bash script/_install.sh
bash script/_download_assets.sh
```

Adjust `CUDA_HOME` if your toolkit is installed elsewhere. The installer uses the Conda environment above, installs the compatible PyTorch3D and CuRobo versions, and applies the required simulator patches. The asset script downloads the assets and configures their paths.

This checkout includes RoboTwin revision [ea25643](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea25643210c3036e57eb44b4cd4b59b8dc1e98d7/README.md), which uses `script/` rather than `scripts/`. Use the commands above in the active `RoboTwin-DSSP` environment; a separate RoboTwin checkout or XPolicyLab installation is not needed.

## 3. Install DSSP

```bash
python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.2.2 --no-build-isolation
python -m pip install -e policy/DSSP/src
python -m pip check
```

Continue with [data preparation, training, and evaluation](../README.md#2-demonstrations).
