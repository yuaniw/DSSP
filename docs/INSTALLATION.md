# Installing DSSP

DSSP uses Linux, an NVIDIA GPU, Python 3.10, and PyTorch 2.4.1 with CUDA 12.1. Building the CUDA extensions also requires the CUDA 12.1 toolkit (`nvcc`) and a compatible C++ compiler. Run the commands below from the repository root.

## 1. Set up RoboTwin

Follow the [official RoboTwin installation and asset instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for SAPIEN, rendering, motion planning, CuRobo, and simulator assets. Keep the required simulator patches described there. This release uses RoboTwin revision `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`.

Use the environment below for DSSP and install the simulator components into the same environment. RoboTwin's `script/_install.sh` installs different dependency versions and unpinned PyTorch3D/CuRobo sources; use the pinned installation steps below for those packages. Apply the SAPIEN UTF-8 file-reading and MPLib planner patches from the RoboTwin setup instructions to this environment.

## 2. Create the DSSP environment

[policy/DSSP/environment.yaml](../policy/DSSP/environment.yaml) installs Python and FFmpeg through Conda and the Python packages through pip, including MPLib, TOPPRA, and the CUDA 12.1 PyTorch wheels. Setuptools is pinned because SAPIEN 3.0.0b1 imports `pkg_resources`; NumPy-dependent planner packages are pinned to versions compatible with NumPy 1.23.5.

```bash
conda env create -f policy/DSSP/environment.yaml
conda activate RoboTwin-DSSP
export PYTHONNOUSERSITE=1
```

Set `PYTHONNOUSERSITE=1` in training and evaluation jobs as well if your account has packages installed under `~/.local`; this keeps them from shadowing the environment's pinned dependencies.

## 3. Install CUDA extensions

Install Mamba and causal-conv1d after PyTorch. Building these extensions from source requires a CUDA toolkit compatible with the installed PyTorch build.

Set `CUDA_HOME` to the toolkit directory on your machine and verify the compiler:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

```bash
python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.2.2 --no-build-isolation
```

Build PyTorch3D v0.7.9 from its pinned source revision. See its [installation instructions](https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/INSTALL.md) for compiler requirements.

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"
```

Install CuRobo v0.7.8 for RoboTwin's motion planner:

```bash
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
python -m pip install --no-build-isolation -e envs/curobo
```

CuRobo v0.7.8 corresponds to revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`. If `envs/curobo` already exists, use that revision before installing. Build on a node with the target GPU visible. For an RTX 4090, `TORCH_CUDA_ARCH_LIST=8.9` selects its architecture; use the appropriate architecture for other GPUs. `MAX_JOBS=4` limits compilation parallelism.

## 4. Install the DSSP package

```bash
python -m pip install -e policy/DSSP/src
python -m pip check
```

Evaluation and demonstration videos use the FFmpeg binary bundled with `imageio-ffmpeg`, which includes the required `libx264` encoder. This avoids depending on the codec selection in the system or Conda FFmpeg build. To select your own compatible binary, set `IMAGEIO_FFMPEG_EXE=/absolute/path/to/ffmpeg`. Verify rendering using RoboTwin's rendering test after downloading the simulator assets.

The launchers use the active Python interpreter. Set `PYTHON=/absolute/path/to/python` to select a different interpreter when needed.

Check that all compiled dependencies import from the active environment:

```bash
python -c "import torch, mamba_ssm, causal_conv1d, pytorch3d.ops, sapien, mplib, curobo; print('Dependencies imported')"
```

## 5. Prepare data and run DSSP

Follow the [usage guide](../policy/DSSP/README.md) to prepare demonstrations, train a model, and evaluate its checkpoint. To run the release tests:

```bash
python -m unittest discover -s tests -v
```

These tests do not require CUDA or simulator assets.

## Optional language-generation tools

DSSP training and evaluation use the provided task instructions and do not require an external language-model API. If you use RoboTwin's optional description or code-generation utilities, install their API client packages and configure your own credentials through environment variables:

- Azure description generation: `AZURE_API_KEY`, `AZURE_INFERENCE_ENDPOINT`, and `AZURE_INFERENCE_MODEL`.
- OpenAI code generation: `OPENAI_API_KEY`; the client uses its standard endpoint configuration.
- DeepSeek code generation: `DEEPSEEK_API_KEY`.
- Moonshot image analysis: `KIMI_API_KEY`.

Those optional utilities send the supplied prompts or images to the configured service. Keep credentials and generated run metadata outside Git.
