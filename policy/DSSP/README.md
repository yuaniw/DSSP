# DSSP implementation guide

See the [main README](../../README.md) for the paper, method overview, figures, and benchmark results. This guide covers the RoboTwin implementation and its command-line interfaces. Run all commands from the repository root.

## Environment

Start with the official [RoboTwin simulator and asset installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). This checkout is based on RoboTwin revision `ea25643210c3036e57eb44b4cd4b59b8dc1e98d7`; future upstream instructions may describe a different revision.

[environment.yaml](environment.yaml) records the DSSP dependency versions used for validation:

| Component | Tested version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.4.1 + CUDA 12.1 |
| Mamba-SSM | 2.2.2 |
| causal-conv1d | 1.4.0 |
| PyTorch3D | 0.7.9 + PyTorch 2.4.1 / CUDA 12.1 |
| Diffusers | 0.18.2 |
| SAPIEN | 3.0.0b1 |

The environment file is a version reference, not a verified one-command installer. PyTorch/PyTorch3D wheels must match the CUDA stack; Mamba and causal-conv1d may require a compatible build toolchain. A clean installation has not yet been tested. After installing the dependencies:

```bash
python -m pip install -e policy/DSSP/src
```

The launchers use the active `python`. Set `PYTHON` to an interpreter path when required. Install FFmpeg for video recording.

## Data layout

Obtain or collect demonstrations following [RoboTwin data collection](https://robotwin-platform.github.io/doc/usage/collect-data.html). Enable `data_type.pointcloud` and joint-state observations in the [task configuration](../../task_config/demo_clean.yml).

Raw demonstrations:

```text
data/<task>/<setting>/data/
  episode0.hdf5
  episode1.hdf5
  ...
```

Preprocess the desired demonstration count:

```bash
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
```

The converter reads point clouds and joint states and produces:

```text
policy/DSSP/data/<task>-<setting>-<demos>.zarr/
  data/point_cloud
  data/state
  data/action
  meta/episode_ends
```

The preprocessing script replaces an existing output dataset with the same name. Skip this step when reusing a prepared Zarr dataset. Training requires the processed dataset to exist; it does not automatically recollect or preprocess data.

The release code does not bundle simulator assets, demonstrations, or checkpoints. Local validation reused data through filesystem links; those links are not portable to another machine.

## Training

```bash
# train.sh TASK SETTING DEMOS SEED GPU [HYDRA_OVERRIDES...]
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
```

The launcher selects [dssp.yaml](src/dssp/config/dssp.yaml) and the trajectory trainer. Defaults include 1,000 epochs, EMA, 49 training demonstrations plus one validation demonstration for this 50-demo example, and disabled online experiment tracking. Set `WANDB_MODE=online` to enable Weights & Biases.

Hydra overrides follow the GPU argument. For a one-epoch training check without writing checkpoints:

```bash
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0   training.num_epochs=1 checkpoint.save_ckpt=false
```

Checkpoints are saved every 500 epochs by default:

```text
policy/DSSP/checkpoints/<task>-<setting>-<demos>_<seed>_dssp_Mamba1/
  500.ckpt
  1000.ckpt
```

Training outputs, including `train_metrics.json`, are written to `policy/DSSP/data/outputs/<task>-dssp-paper_seed<seed>/`. Use `hydra.run.dir=/absolute/output/path` to select another run directory.

## Evaluation

```bash
# eval.sh TASK TASK_CONFIG CKPT_SETTING DEMOS SEED GPU [EVAL_OVERRIDES...]
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0
```

`TASK_CONFIG` chooses the evaluation environment; `CKPT_SETTING` identifies the demonstration setting used for training. The default uses `1000.ckpt`, 100 episodes, EMA weights, cached Mamba inference, and video recording.

Useful overrides:

| Option | Effect |
|---|---|
| `--eval_episodes 1` | Evaluate one episode. |
| `--checkpoint_num 500` | Select the 500-epoch checkpoint. |
| `--checkpoint_path /absolute/path/1000.ckpt` | Load an explicit checkpoint file. |
| `--eval_video_log False` | Disable video recording. |

Outputs:

```text
eval_result/<task>/DSSP/<task_config>/<ckpt_setting>/dssp_Mamba1/
  seed_<seed>/ckpt_<checkpoint_num>/run_<timestamp>/
    _metrics.json
    _result.txt
    episode0.mp4
    ...
```

The metrics record the success rate, evaluated/successful episode counts, model class, and performance measurements. Video filenames are indexed by episode.

## Slurm

Run the same launchers inside a Slurm job with a GPU allocation and the desired environment activated. They preserve the GPU selected by Slurm; the positional GPU argument controls device selection only outside Slurm. Partition names, resource requests, and environment activation depend on the cluster. Training must finish before evaluating its checkpoint.

## Implementation map

| Component | Implementation |
|---|---|
| Selected configuration | [dssp.yaml](src/dssp/config/dssp.yaml) |
| DSSP policy and trajectory loss | [DSSP](src/dssp/policy/dssp.py) |
| Shared diffusion policy | [DSSPBase](src/dssp/policy/dssp_base.py) |
| History encoder | [DSSPHistoryEncoder](src/dssp/model/diffusion/dssp_history_encoder.py) |
| Action denoiser | [DSSPDenoiser](src/dssp/model/diffusion/dssp_denoiser.py) |
| Dynamics prediction head | [LatentForwardDynamicsMLP](src/dssp/model/diffusion/latent_forward_dynamics.py) |
| Point-cloud encoder | [PointCloudObservationEncoder](src/dssp/model/vision/pointnet_extractor.py) |
| Trajectory dataset | [RobotTrajectoryDataset](src/dssp/dataset/robot_trajectory_dataset.py) |
| Training workspace | [TrainDSSPTrajectoryWorkspace](src/train_dssp_trajectory.py) |
| RoboTwin integration | [deploy_policy.py](deploy_policy.py) and [dssp_policy.py](src/dssp_policy.py) |

The selected policy uses one full-history context token and three recent observation tokens, an eight-step action horizon, six actions per execution chunk, ten DDIM steps, and a cosine dynamics loss weighted by 0.05. The progress loss is disabled. Its inactive head remains in the implementation to preserve compatibility with existing checkpoint keys.

## Checkpoints and release validation

State-dictionary `.ckpt` files from the selected pre-rename implementation load strictly. Full-workspace pickle snapshots remain tied to their original module names. Use `--checkpoint_path` for a checkpoint stored outside the canonical DSSP directory.

See [reproduction and compatibility notes](../../docs/REPRODUCIBILITY.md) for the existing 1,000-epoch training run, 100-episode evaluations, and the DSSP rename checks.
