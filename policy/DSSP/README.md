# DSSP on RoboTwin

This guide covers data preparation, training, and evaluation for DSSP on RoboTwin.

See the [main README](../../README.md) for the method and benchmark results. Run all commands below from the repository root.

## 1. Set up the environment

Complete the [installation guide](../../docs/INSTALLATION.md), including the RoboTwin simulator assets and DSSP dependencies. Then activate the environment:

```bash
conda activate RoboTwin-DSSP
export PYTHONNOUSERSITE=1
```

## 2. Prepare demonstrations

Collect demonstrations using the provided task configuration:

```bash
# Arguments: task, data setting, GPU index.
bash collect_data.sh stack_bowls_two demo_clean 0
```

The included [demo_clean configuration](../../task_config/demo_clean.yml) collects 50 demonstrations with point clouds and joint states enabled. See the [RoboTwin data collection guide](https://robotwin-platform.github.io/doc/usage/collect-data.html) for other collection settings.

### Data layout

Skip collection if you already have the demonstrations. Place the 50 episode files in:

```text
data/stack_bowls_two/demo_clean/data/
  episode0.hdf5
  ...
  episode49.hdf5
```

Convert them to the dataset used by DSSP:

```bash
# Arguments: task, data setting, demonstration count.
bash policy/DSSP/process_data.sh stack_bowls_two demo_clean 50
```

This creates `policy/DSSP/data/stack_bowls_two-demo_clean-50.zarr`. Skip this step if you already have this processed dataset; preprocessing does not overwrite existing datasets.

## 3. Train DSSP

```bash
# Arguments: task, data setting, demonstration count, training seed, GPU index.
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0
```

The launcher uses [dssp.yaml](src/dssp/config/dssp.yaml) to train for 1,000 epochs with EMA. For this 50-demonstration example, 49 demonstrations are used for training and one for validation. Online experiment tracking is disabled by default.


Training logs are saved under `policy/DSSP/data/outputs`. Use `train_log.jsonl` to inspect per-epoch losses and `train_metrics.json` to check the completed epoch and optimizer-update counts.

### Resume an interrupted run

```bash
bash policy/DSSP/train.sh stack_bowls_two demo_clean 50 2 0 training.resume=true
```

This resumes from the latest saved checkpoint for the same task, data setting, demonstration count, and seed. Keep the original dataset and training configuration; the default target remains 1,000 total epochs.

## 4. Evaluate the trained policy

After training completes, run evaluation. By default, the launcher loads `1000.ckpt`, saved at epoch 1,000:

```bash
# Arguments: task, evaluation setting, training data setting, demos, seed, GPU.
bash policy/DSSP/eval.sh stack_bowls_two demo_clean demo_clean 50 2 0
```

Evaluation runs for 100 episodes using EMA weights and cached Mamba inference, and saves a video for each episode. The second argument selects the evaluation environment; the third identifies the data setting used to train the checkpoint.

### Read the results

Each evaluation creates a new output directory:

```text
eval_result/stack_bowls_two/DSSP/demo_clean/demo_clean/dssp_Mamba1/
  seed_2/ckpt_1000/run_<timestamp>/
    _metrics.json
    _config.json
    _episodes.jsonl
    _result.txt
    episode0.mp4
    ...
```

- **Success rate:** `_metrics.json` contains `success_rate` as a value between 0 and 1, together with `successful_episodes` and `evaluated_episodes`. Multiply `success_rate` by 100 to report a percentage.
- **Episode outcomes:** `_episodes.jsonl` records scene seeds, success flags, and action counts.
- **Run settings:** `_config.json` records the evaluation and environment configuration.
- **Videos:** `episode<N>.mp4` contains the rollout for each evaluated episode.

For comparisons, use the same task configuration, demonstration count, training schedule, and evaluation episode count, and record the seed. The single-task example above is a starting point; see the main README for the benchmark scope.

## Other tasks and optional settings

To run another task, replace `stack_bowls_two` in the collection, preprocessing, training, and evaluation commands. Task names correspond to the implementations in [`envs/`](../../envs/). Keep the data setting, demonstration count, and seed consistent between training and checkpoint selection.

Training accepts Hydra overrides after the GPU argument. For example, append `training.checkpoint_every=100` to save every 100 epochs, or `training.resume=true training.resume_path=/absolute/path/500.ckpt` to resume a specific checkpoint. Set `WANDB_MODE=online` before training to enable Weights & Biases.

Evaluation accepts these options after the GPU argument:

| Option | Purpose |
|---|---|
| `--checkpoint_num 1000` | Evaluate the checkpoint saved at epoch 1,000 (default). |
| `--checkpoint_path /absolute/path/model.ckpt` | Load a checkpoint stored elsewhere. |
| `--eval_episodes 1` | Run a single episode to check the setup. |
| `--eval_video_log False` | Disable video recording. |

Use the default 100 evaluation episodes when comparing success rates.

## Configuration and method code

The default configuration uses full-history encoding with one history context token and three recent observation tokens, an eight-step action horizon, six executed actions per chunk, ten DDIM denoising steps, and a dynamics loss weight of 0.05.

| Component | Source |
|---|---|
| Training and model configuration | [dssp.yaml](src/dssp/config/dssp.yaml) |
| DSSP policy and training objective | [dssp.py](src/dssp/policy/dssp.py) |
| History encoder | [dssp_history_encoder.py](src/dssp/model/diffusion/dssp_history_encoder.py) |
| Action denoiser | [dssp_denoiser.py](src/dssp/model/diffusion/dssp_denoiser.py) |
| Dynamics prediction head | [latent_forward_dynamics.py](src/dssp/model/diffusion/latent_forward_dynamics.py) |
