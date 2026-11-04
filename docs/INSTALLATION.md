# Installation

**Requirements:** Linux, Conda, an NVIDIA GPU with CUDA 12.1 and Vulkan support, and the CUDA 12.1 toolkit (including `nvcc`). Run all commands from the DSSP repository root with the GPU available.

## 1. Create the environment

Install the system dependencies on Ubuntu if needed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git unzip libvulkan1 libgl1 libegl1
```

```bash
conda env create -f policy/DSSP/environment.yaml
conda activate RoboTwin-DSSP
export PYTHONNOUSERSITE=1
```

## 2. Install CUDA extensions and DSSP

Set `CUDA_HOME` to your CUDA 12.1 installation:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4

python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.2.2 --no-build-isolation
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"

git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
python -m pip install --no-build-isolation -e envs/curobo
python -m pip install -e policy/DSSP/src
python -m pip check
```

## 3. Apply the required simulator patches

Run once in the activated environment to apply RoboTwin's SAPIEN and MPLib patches:

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

## 4. Prepare simulator assets

Download the RoboTwin assets and configure their paths:

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

The assets must be under `assets/background_texture`, `assets/embodiments`, and `assets/objects`. Rerun the path configuration command if you move the repository.

## 5. Train and evaluate

Example for `stack_bowls_two` with 50 demonstrations:

```bash
bash collect_data.sh stack_bowls_two demo_clean 0
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0
```

Skip collection if you already have the demonstrations; skip preprocessing if the Zarr dataset exists. Training runs for 1,000 epochs. Evaluation uses the resulting `1000.ckpt` for 100 episodes and saves videos.

See the [usage guide](../policy/DSSP/README.md) for data layout and command options.

This setup uses RoboTwin revision [ea25643](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea25643210c3036e57eb44b4cd4b59b8dc1e98d7/README.md). Use the steps above instead of `script/_install.sh` to keep the DSSP dependency versions.
