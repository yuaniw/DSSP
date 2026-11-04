# Installing DSSP

DSSP uses Linux, an NVIDIA GPU, Python 3.10, and PyTorch 2.4.1 with CUDA 12.1. Run the commands below from the repository root.

## 1. Set up RoboTwin

Follow the [official RoboTwin installation and asset instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for SAPIEN, rendering, motion planning, CuRobo, and simulator assets. Keep the required simulator patches described there. This release uses RoboTwin revision `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`.

Use the environment below for DSSP and install the simulator components into the same environment. If you use RoboTwin's `script/_install.sh`, apply the DSSP dependency versions afterwards: that script installs its own versions, including Hugging Face Hub.

## 2. Create the DSSP environment

[policy/DSSP/environment.yaml](../policy/DSSP/environment.yaml) specifies the Python dependencies and the CUDA 12.1 PyTorch wheel index, including the MPLib and TOPPRA packages used by RoboTwin.

```bash
conda env create -f policy/DSSP/environment.yaml
conda activate RoboTwin-DSSP
```

## 3. Install CUDA extensions

Install Mamba and causal-conv1d after PyTorch. Building these extensions from source requires a CUDA toolkit compatible with the installed PyTorch build.

```bash
python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.2.2 --no-build-isolation
```

Install PyTorch3D following its [official installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), using a build compatible with PyTorch 2.4.1 and CUDA 12.1. The dependency version is `0.7.9`; the suffix in `0.7.9+pt2.4.1cu121` identifies the PyTorch/CUDA build and is not a standalone PyPI requirement.

## 4. Install the DSSP package

```bash
python -m pip install -e policy/DSSP/src
python -m pip check
```

FFmpeg is required for evaluation videos and is included in the Conda environment file. Verify rendering using RoboTwin's rendering test after downloading the simulator assets.

The launchers use the active Python interpreter. Set `PYTHON=/absolute/path/to/python` to select a different interpreter when needed.

## 5. Prepare data and run DSSP

Follow the [usage guide](../policy/DSSP/README.md) to prepare demonstrations, train a model, and evaluate its checkpoint. To run the preprocessing and argument-parser tests:

```bash
python -m unittest discover -s tests -v
```

These tests do not require CUDA or simulator assets.
