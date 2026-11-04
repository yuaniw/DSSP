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

This repository contains the **DSSP implementation for RoboTwin 2.0**, including trajectory training, cached inference, and evaluation with video recording.

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

DSSP achieves **62.30% average success** across 50 RoboTwin 2.0 tasks with **44.3M parameters**. The table below presents the paper's RoboTwin 2.0 comparison (Table 1). Success rates are percentages (↑); parameter counts are shown in the last column (↓).

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

Bold numeric entries indicate the best result in each column. The benchmark uses 50 tasks and 100 evaluation episodes per task under `demo_clean`; observation modalities differ across methods.

## Getting started

Clone the repository:

```bash
git clone https://github.com/yuaniw/DSSP.git
cd DSSP
```

### 1. Environment and simulator assets

Follow the [installation guide](docs/INSTALLATION.md) to set up the environment, required simulator patches, and assets in this checkout.

### 2. Demonstrations

Prepare demonstrations using the official [RoboTwin data collection guide](https://robotwin-platform.github.io/doc/usage/collect-data.html) and [task configuration guide](https://robotwin-platform.github.io/doc/usage/configurations.html). DSSP requires point clouds and joint-state observations.

For the example below, place 50 raw demonstrations in `data/stack_bowls_two/demo_clean/data/`, then preprocess once:

```bash
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
```

This creates `policy/DSSP/data/stack_bowls_two-demo_clean-50.zarr`. Skip preprocessing if you already have the processed dataset. Simulator assets and demonstrations are downloaded or collected separately. See [data layout](policy/DSSP/README.md#data-layout).

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

After training, this command evaluates `1000.ckpt` for 100 episodes with cached inference and video recording enabled. Results and `episode<N>.mp4` videos are written under `eval_result/<task>/DSSP/`. To use a checkpoint stored elsewhere, pass `--checkpoint_path /absolute/path/to/model.ckpt`.

See the [usage guide](policy/DSSP/README.md) for configuration overrides, explicit checkpoint paths, output layout, and Slurm usage.

## Repository structure

```text
policy/DSSP/
  src/dssp/
    config/dssp.yaml           # DSSP configuration
    policy/dssp.py             # DSSP policy and auxiliary objective
    policy/dssp_base.py        # Shared diffusion-policy implementation
    model/diffusion/           # History encoder, denoiser, and dynamics head
  src/train_dssp_trajectory.py # Trajectory training entry point
  deploy_policy.py             # RoboTwin observation/action adapter
  train.sh / eval.sh           # Training and evaluation launchers
  README.md                   # Detailed usage and implementation map
envs/                         # RoboTwin simulation tasks
task_config/                  # Simulator and observation settings
docs/INSTALLATION.md           # Environment and dependency setup
```

## Citation

If you use DSSP in your research, please cite:

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

DSSP builds on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), and [Mamba](https://github.com/state-spaces/mamba). We thank their authors for making their work available. See the [official RoboTwin documentation](https://robotwin-platform.github.io/doc/) for simulator setup and usage.

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for the retained upstream license and copyright notices.

For implementation questions, please open an issue in this repository. Paper correspondence: [Yutong Ban](mailto:yban@sjtu.edu.cn).
