<h1 align="center">DSSP: Diffusion State Space Policy<br>with Full-History Encoding</h1>

<p align="center">
  <a href="https://yuaniw.github.io/DSSP-Page/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2605.14598">Paper</a> ·
  <a href="https://yuaniw.github.io/DSSP-Page/#demo">Video</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Zhiyuan Guan<sup>1</sup>, Jianshu Hu<sup>1</sup>, Han Fang<sup>1</sup>,
  Yunpeng Jiang<sup>1</sup>, Yize Huang<sup>2</sup>, Shujia Li<sup>1</sup>,
  Xiao Li<sup>2</sup>, Yutong Ban<sup>1,*</sup><br>
  <sup>1</sup>Global College, Shanghai Jiao Tong University<br>
  <sup>2</sup>School of Mechanical Engineering, Shanghai Jiao Tong University<br>
  <sup>*</sup>Corresponding author
</p>

DSSP conditions robot actions on the full observation history. A compact state-space memory captures task progress, while recent observations provide local control information for diffusion-based action generation.

This repository contains the **DSSP implementation for RoboTwin 2.0**, including trajectory training, cached inference, and evaluation with video recording. The other benchmarks and real-robot systems described in the paper are outside this release.

<p align="center">
  <img src="https://yuaniw.github.io/DSSP-Page/static/images/dssp-teaser.jpg" alt="History-dependent robot manipulation with DSSP" width="900">
</p>

## Method

<p align="center">
  <img src="https://yuaniw.github.io/DSSP-Page/static/images/dssp-method.png" alt="DSSP architecture: history encoding, dynamics supervision, and action denoising" width="1000">
</p>

1. **Full-history encoding.** A causal Mamba encoder summarizes the observation sequence into a context token.
2. **Dynamics-aware learning.** An auxiliary prediction objective encourages this context to retain information about future state evolution.
3. **Hierarchical action conditioning.** A state-space denoiser uses the history context and recent observation tokens, with diffusion timestep modulation through AdaLN.

See the [paper](https://arxiv.org/abs/2605.14598) for the formulation and the [implementation guide](policy/DSSP/README.md#implementation-map) for the corresponding code.

## Results

RoboTwin 2.0 results from Table 1, as reported on the [project page](https://yuaniw.github.io/DSSP-Page/#results). Success rates are percentages (↑); parameter counts are shown in the last column (↓).

| Method | Observation | Short (18) | Mid (15) | Long (17) | Average | Parameters |
|---|---|---:|---:|---:|---:|---:|
| ACT | RGB | 29.94 | 26.40 | 32.47 | 29.74 | 80.0M |
| DP | RGB | 31.28 | 26.00 | 26.41 | 28.04 | 96.8M |
| π₀ | RGB | 44.56 | 44.07 | 50.47 | 46.42 | 3.3B |
| RDT | RGB | 36.72 | 32.47 | 33.94 | 34.50 | 1.2B |
| SeedPolicy | RGB | 43.39 | 36.53 | 47.59 | 42.76 | 147.3M |
| DP3 | Point cloud | 59.83 | 52.53 | 52.76 | 55.24 | 264.4M |
| FlowPolicy | Point cloud | 54.89 | 37.40 | 29.59 | 41.04 | 264.4M |
| MambaPolicy | Point cloud | 61.67 | 56.27 | 58.88 | 59.10 | 47.9M |
| Diff-Control | Point cloud | 64.00 | 54.67 | 58.82 | 59.44 | 264.4M |
| MTIL | Point cloud | **66.22** | 54.73 | 56.18 | 59.36 | 60.5M |
| CycleManip | Point cloud | 57.78 | 49.27 | 54.47 | 54.10 | 279.2M |
| **DSSP (ours)** | Point cloud | 64.78 | **57.33** | **64.06** | **62.30** | **44.3M** |

Bold numeric entries indicate the best result in each column. Short, Mid, and Long contain 18, 15, and 17 tasks, respectively. The benchmark uses 50 tasks and 100 evaluation episodes per task under `demo_clean`; observation modalities differ across methods. These are manuscript results; the release checkout has separately been validated on Stack Bowls Two. See [reproduction notes](docs/REPRODUCIBILITY.md) for that validation and its scope.

## Getting started

### 1. Environment and simulator assets

Follow the official [RoboTwin installation and asset setup](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for the simulator, rendering dependencies, robot embodiments, and objects. Use this checkout for DSSP training and evaluation.

DSSP-specific dependency versions are recorded in [policy/DSSP/environment.yaml](policy/DSSP/environment.yaml). The tested stack uses Python 3.10, PyTorch 2.4.1 with CUDA 12.1, Mamba-SSM 2.2.2, and causal-conv1d 1.4.0. See the [environment notes](policy/DSSP/README.md#environment) before installing CUDA-dependent packages. A fresh installation from the environment file has not yet been validated.

Once those dependencies are installed, register DSSP in the active environment:

```bash
# Run from the root of this repository.
python -m pip install -e policy/DSSP/src
```

### 2. Demonstrations

Prepare demonstrations using the official [RoboTwin data collection guide](https://robotwin-platform.github.io/doc/usage/collect-data.html) and [task configuration guide](https://robotwin-platform.github.io/doc/usage/configurations.html). DSSP requires point clouds and joint-state observations.

For the example below, place 50 raw demonstrations in `data/stack_bowls_two/demo_clean/data/`, then preprocess once:

```bash
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
```

This creates `policy/DSSP/data/stack_bowls_two-demo_clean-50.zarr`. Skip preprocessing if you already have the processed dataset. Dataset files, assets, and checkpoints are not bundled with the code; public DSSP checkpoint downloads are not yet listed. See [data layout](policy/DSSP/README.md#data-layout).

### 3. Train

```bash
# Arguments: task, setting, demonstration count, training seed, GPU index.
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
```

The [DSSP configuration](policy/DSSP/src/dssp/config/dssp.yaml) uses full-trajectory training, XYZ point clouds, Mamba1, EMA, and 1,000 epochs. It combines one history context token with three recent observation tokens, an eight-step action horizon, six executed actions per chunk, and a dynamics loss weight of 0.05.

Checkpoints are written to `policy/DSSP/checkpoints/<task>-<setting>-<demos>_<seed>_dssp_Mamba1/`.

### 4. Evaluate and record videos

```bash
# Arguments: task, evaluation setting, checkpoint setting, demos, seed, GPU.
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0
```

The default evaluates checkpoint `1000.ckpt` for 100 episodes with cached inference and video recording enabled. Results and `episode<N>.mp4` videos are written under `eval_result/<task>/DSSP/`.

For a single-episode check:

```bash
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0 --eval_episodes 1
```

See the [usage guide](policy/DSSP/README.md) for configuration overrides, explicit checkpoint paths, output layout, and Slurm usage.

## Repository structure

```text
policy/DSSP/
  src/dssp/
    config/dssp.yaml           # Selected DSSP configuration
    policy/dssp.py             # DSSP policy and auxiliary objective
    policy/dssp_base.py        # Shared diffusion-policy implementation
    model/diffusion/           # History encoder, denoiser, and dynamics head
  src/train_dssp_trajectory.py # Trajectory training entry point
  deploy_policy.py             # RoboTwin observation/action adapter
  train.sh / eval.sh           # Training and evaluation launchers
  README.md                   # Detailed usage and implementation map
envs/                         # RoboTwin simulation tasks
task_config/                  # Simulator and observation settings
docs/REPRODUCIBILITY.md        # Release validation and compatibility notes
```

## Citation

If DSSP is useful for your research, please cite the paper. The entry below follows the [project page citation](https://yuaniw.github.io/DSSP-Page/#bibtex).

```bibtex
@inproceedings{guan2026dssp,
  title={DSSP: Diffusion State Space Policy with Full-History Encoding},
  author={Zhiyuan Guan and Jianshu Hu and Han Fang and Yunpeng Jiang and Yize Huang and Shujia Li and Xiao Li and Yutong Ban},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  eprint={2605.14598},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  doi={10.48550/arXiv.2605.14598},
  url={https://arxiv.org/abs/2605.14598}
}
```

## Acknowledgments and license

DSSP builds on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), and [Mamba](https://github.com/state-spaces/mamba). We thank their authors for making their work available. Upstream project information and citations are preserved in [the RoboTwin archive](docs/UPSTREAM_ROBOTWIN.md).

The upstream [LICENSE](LICENSE) and copyright notice are retained. Refer to the respective upstream projects for their dependency licenses.

For implementation questions, please open an issue in this repository. Paper correspondence: [Yutong Ban](mailto:yban@sjtu.edu.cn).
