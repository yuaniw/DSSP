# DSSP usage guide

See the [main README](../../README.md) for the paper, method overview, figures, and benchmark results. This guide covers the RoboTwin implementation and its command-line interfaces. Run all commands from the repository root.

## Environment

Set up the simulator and assets using the official [RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), then install the DSSP dependencies following [INSTALLATION.md](../../docs/INSTALLATION.md).

The main dependency versions are:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.4.1 + CUDA 12.1 |
| Mamba-SSM | 2.2.2 |
| causal-conv1d | 1.4.0 |
| PyTorch3D | 0.7.9 + PyTorch 2.4.1 / CUDA 12.1 |
| Diffusers | 0.18.2 |
| SAPIEN | 3.0.0b1 |

PyTorch3D and the CUDA extensions must match the installed PyTorch/CUDA stack. After installing the dependencies, register the DSSP package:

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

The preprocessing script refuses to overwrite an existing output dataset, including symlinks. Missing or invalid input fails before an output dataset is created. Skip this step when reusing a prepared Zarr dataset. Training requires the processed dataset to exist; it does not automatically recollect or preprocess data.

Download or collect the simulator assets and demonstrations separately. Train a model using the commands below to obtain a checkpoint for evaluation.

## Training

```bash
# train.sh TASK SETTING DEMOS SEED GPU [HYDRA_OVERRIDES...]
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
```

The launcher selects [dssp.yaml](src/dssp/config/dssp.yaml) and the trajectory trainer. Defaults include 1,000 epochs, EMA, 49 training demonstrations plus one validation demonstration for this 50-demo example, and disabled online experiment tracking. Set `WANDB_MODE=online` to enable Weights & Biases.

Pass Hydra overrides after the GPU argument. For example, to run a shorter training schedule and save a checkpoint every 100 epochs:

```bash
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0 training.num_epochs=100 training.checkpoint_every=100
```

Checkpoints are saved every 500 epochs by default:

```text
policy/DSSP/checkpoints/<task>-<setting>-<demos>_<seed>_dssp_Mamba1/
  500.ckpt
  1000.ckpt
```

Training outputs, including `train_metrics.json` and per-epoch loss records in `train_log.jsonl`, are written to `policy/DSSP/data/outputs/<task>-dssp-paper_seed<seed>/`. Use `hydra.run.dir=/absolute/output/path` to select another log directory. Checkpoint paths depend on the task, setting, demonstration count, and training seed; changing the log directory does not change the checkpoint directory.

## Resume training

Resume the latest numbered checkpoint for the same task, setting, and seed:

```bash
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0 training.resume=true
```

Set `training.resume_path=/absolute/path/500.ckpt` to select a particular checkpoint. `training.num_epochs` remains the total target epoch count, including completed epochs. Keep the dataset, seed, batch/accumulation settings, and total training schedule the same when resuming. A requested resume with no checkpoint fails instead of starting a new run.

Trajectory checkpoints include model and optimizer state, training progress, scheduler and EMA progress, and Python/NumPy/PyTorch random states. GPU operations and DataLoader worker state can affect exact numerical replay.

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
    _config.json
    _episodes.jsonl
    _result.txt
    episode0.mp4
    ...
```

The metrics record the success rate, evaluated/successful episode counts, model class, and performance measurements. `_config.json` records the requested evaluation and environment configuration. `_episodes.jsonl` records each accepted scene seed, success flag, and executed action count, so comparisons can be matched by scene. Video filenames are indexed by episode.

## Slurm

Run the same launchers inside a Slurm job with a GPU allocation and the desired environment activated. They preserve the GPU selected by Slurm; the positional GPU argument controls device selection only outside Slurm. Partition names, resource requests, and environment activation depend on the cluster. Training must finish before evaluating its checkpoint.

## Implementation map

| Component | Implementation |
|---|---|
| Configuration | [dssp.yaml](src/dssp/config/dssp.yaml) |
| DSSP policy and trajectory loss | [DSSP](src/dssp/policy/dssp.py) |
| Shared diffusion policy | [DSSPBase](src/dssp/policy/dssp_base.py) |
| History encoder | [DSSPHistoryEncoder](src/dssp/model/diffusion/dssp_history_encoder.py) |
| Action denoiser | [DSSPDenoiser](src/dssp/model/diffusion/dssp_denoiser.py) |
| Dynamics prediction head | [LatentForwardDynamicsMLP](src/dssp/model/diffusion/latent_forward_dynamics.py) |
| Point-cloud encoder | [PointCloudObservationEncoder](src/dssp/model/vision/pointnet_extractor.py) |
| Trajectory dataset | [RobotTrajectoryDataset](src/dssp/dataset/robot_trajectory_dataset.py) |
| Training workspace | [TrainDSSPTrajectoryWorkspace](src/train_dssp_trajectory.py) |
| RoboTwin integration | [deploy_policy.py](deploy_policy.py) and [dssp_policy.py](src/dssp_policy.py) |

DSSP uses one full-history context token and three recent observation tokens, an eight-step action horizon, six actions per execution chunk, ten DDIM steps, and a cosine dynamics loss weighted by 0.05. The default configuration disables the progress auxiliary loss.

## Tests

Run the preprocessing and evaluation-argument tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests do not require CUDA or simulator assets.
