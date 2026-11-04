# Installing DSSP

DSSP uses Linux, an NVIDIA GPU, Python 3.10, and PyTorch 2.4.1 with CUDA 12.1. This guide covers the simulator, motion planner, assets, and DSSP dependencies in this repository. Run the commands below from the DSSP repository root, in order.

The simulator is based on RoboTwin revision `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`. For provenance, see the [README at that revision](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea25643210c3036e57eb44b4cd4b59b8dc1e98d7/README.md) and its [installation script](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea25643210c3036e57eb44b4cd4b59b8dc1e98d7/script/_install.sh). The [official documentation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) is an additional reference; the steps needed for this release are included below.

## 1. System prerequisites

Use a Linux machine with Conda, an NVIDIA driver compatible with CUDA 12.1, and the CUDA 12.1 toolkit, including `nvcc`. SAPIEN also requires the NVIDIA Vulkan driver for rendering. On Ubuntu 22.04, install the host utilities and graphics runtime libraries:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git unzip libvulkan1 vulkan-tools libgl1 libegl1
nvidia-smi
```

On a managed cluster, use the administrator-provided driver, toolkit, and system packages. When starting a container, expose the GPU and NVIDIA graphics/Vulkan libraries, for example with Docker options `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. The CUDA version displayed by `nvidia-smi` describes driver support; check the installed compiler with `nvcc --version` in step 3.

Use this DSSP checkout throughout setup. The steps below replace `bash script/_install.sh`, whose inherited dependency versions and unpinned source installs differ from DSSP's environment.

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
export MAX_JOBS=4
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Build on a node with the target GPU visible. For an RTX 4090, set `export TORCH_CUDA_ARCH_LIST=8.9` before compiling; use the appropriate architecture for other GPUs. `MAX_JOBS=4` limits compilation parallelism.

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

CuRobo v0.7.8 corresponds to revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`. If `envs/curobo` already exists, use that revision before installing.

## 4. Apply the RoboTwin simulator patches

The pinned RoboTwin revision reads URDF/SRDF files as UTF-8 and changes MPLib's screw-planning termination condition. Apply these two upstream patches to the active environment after installing SAPIEN 3.0.0b1 and MPLib 0.2.1:

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sysconfig

assert version("sapien") == "3.0.0b1"
assert version("mplib") == "0.2.1"
site = Path(sysconfig.get_paths()["purelib"])
patches = {
    "sapien/wrapper/urdf_loader.py": [
        ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
        ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
    ],
    "mplib/planner.py": [
        ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
         "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
    ],
}
for relative, replacements in patches.items():
    path = site / relative
    text = path.read_text()
    for before, after in replacements:
        if after in text:
            continue  # Allow rerunning setup in an already patched environment.
        if text.count(before) != 1:
            raise RuntimeError(f"Unexpected source in {path}; check the package version")
        text = text.replace(before, after)
    path.write_text(text)
    print(f"Patched: {path}")
PY
```

These patches modify only the active environment's installed packages. Reapply them if those packages are reinstalled. Keep SAPIEN's SRDF filename expression as `urdf_file[:-4] + "srdf"`; the dot is already retained by the slice.

## 5. Download and configure simulator assets

The asset downloader included in this revision fetches `background_texture.zip`, `embodiments.zip`, and `objects.zip` from the `TianxingChen/RoboTwin2.0` Hugging Face dataset. Download and extract them into this checkout:

```bash
(
  set -e
  cd assets
  python _download.py
  unzip -n background_texture.zip
  unzip -n embodiments.zip
  unzip -n objects.zip
)
python script/update_embodiment_config_path.py
```

The resulting layout must be:

```text
assets/
  background_texture/
  embodiments/
  objects/
```

The path configuration command generates robot configuration files from the supplied `*_tmp.yml` templates. Run it from the repository root and rerun it after moving the checkout. When reusing existing assets, copy or link these three directories into `assets/`; the command writes into `assets/embodiments`, so use a separate copy of that directory if another checkout needs its own paths.

The asset dataset is versioned separately from the RoboTwin code. The inherited downloader uses the dataset's default revision; the code revision above does not pin an asset snapshot. Assets and generated path configurations are excluded from Git. Demonstration data is prepared separately in step 8.

## 6. Install the DSSP package

```bash
python -m pip install -e policy/DSSP/src
python -m pip check
```

Evaluation and demonstration videos use the FFmpeg binary bundled with `imageio-ffmpeg`, which includes the required `libx264` encoder. This avoids depending on the codec selection in the system or Conda FFmpeg build. To select your own compatible binary, set `IMAGEIO_FFMPEG_EXE=/absolute/path/to/ffmpeg`.

The launchers use the active Python interpreter. Set `PYTHON=/absolute/path/to/python` to select a different interpreter when needed.

Check that all compiled dependencies import from the active environment:

```bash
python -c "import torch, mamba_ssm, causal_conv1d, pytorch3d.ops, sapien, mplib, curobo; print('Dependencies imported')"
```

## 7. Verify GPU rendering and video encoding

Run this check on a GPU node with the environment activated. It renders a camera frame without opening a window and raises an error if rendering fails:

```bash
python - <<'PY'
import numpy as np
import sapien.core as sapien
import torch

assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir("rt")
scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_ground(0)
camera = scene.add_camera("setup_check", 64, 64, 1.0, 0.1, 10.0)
camera.set_pose(sapien.Pose([0, 0, 1]))
scene.update_render()
camera.take_picture()
rgba = camera.get_picture("Color")
assert rgba.shape == (64, 64, 4) and np.isfinite(rgba).all()
print("CUDA and headless SAPIEN rendering passed")
PY

python -m unittest discover -s tests -v
```

The release tests include encoding and decoding an H.264 video; they do not require CUDA or simulator assets. If the render check fails while CUDA works, run `vulkaninfo --summary` and check that it lists the NVIDIA GPU. Install or expose the NVIDIA Vulkan driver in the host/container before continuing. A CPU Vulkan device alone is insufficient for this setup.

## 8. Prepare data, train, and evaluate

For a complete example, the included `demo_clean` configuration collects 50 demonstrations with point clouds and joint states:

```bash
bash collect_data.sh stack_bowls_two demo_clean 0
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0
```

Run these commands sequentially: evaluation uses the `1000.ckpt` produced by training. If you already have compatible demonstrations in `data/stack_bowls_two/demo_clean/data/`, skip collection. If the processed Zarr dataset already exists, also skip preprocessing, which refuses to overwrite it. See the local [usage guide](../policy/DSSP/README.md) for data layout, shorter training runs, checkpoint overrides, videos, and Slurm usage.

## Optional language-generation tools

DSSP training and evaluation use the provided task instructions and do not require an external language-model API. If you use RoboTwin's optional description or code-generation utilities, install their API client packages and configure your own credentials through environment variables:

- Azure description generation: `AZURE_API_KEY`, `AZURE_INFERENCE_ENDPOINT`, and `AZURE_INFERENCE_MODEL`.
- OpenAI code generation: `OPENAI_API_KEY`; the client uses its standard endpoint configuration.
- DeepSeek code generation: `DEEPSEEK_API_KEY`.
- Moonshot image analysis: `KIMI_API_KEY`.

Those optional utilities send the supplied prompts or images to the configured service. Keep credentials and generated run metadata outside Git.
