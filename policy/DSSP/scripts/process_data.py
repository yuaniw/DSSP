import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr


def load_hdf5(dataset_path):
    with h5py.File(dataset_path, "r") as root:
        return root["/joint_action/vector"][()], root["/pointcloud"][()]


def process_demonstrations(load_dir, save_dir, num_episodes):
    if num_episodes < 1:
        raise ValueError("The demonstration count must be positive.")
    # Existing datasets, including links to research data, must never be replaced.
    if save_dir.exists() or save_dir.is_symlink():
        raise FileExistsError(f"Dataset already exists: {save_dir}")

    states, actions, point_clouds, episode_ends = [], [], [], []
    total_steps = 0
    for episode in range(num_episodes):
        vector, point_cloud = load_hdf5(load_dir / f"episode{episode}.hdf5")
        if vector.ndim != 2 or point_cloud.ndim != 3:
            raise ValueError("Expected joint states [T, D] and point clouds [T, N, C].")
        if len(vector) != len(point_cloud) or len(vector) < 2:
            raise ValueError(f"Episode {episode} requires matching observations and at least two frames.")
        states.append(vector[:-1])
        actions.append(vector[1:])
        point_clouds.append(point_cloud[:-1])
        total_steps += len(vector) - 1
        episode_ends.append(total_steps)

    arrays = {
        "state": np.concatenate(states).astype(np.float32),
        "action": np.concatenate(actions).astype(np.float32),
        "point_cloud": np.concatenate(point_clouds),
    }
    # Exclusive creation also prevents overwriting a dataset created concurrently.
    root = zarr.open_group(str(save_dir), mode="w-")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    for name, values in arrays.items():
        data.create_dataset(
            name, data=values, chunks=(min(100, len(values)),) + values.shape[1:],
            compressor=compressor,
        )
    meta.create_dataset(
        "episode_ends", data=np.asarray(episode_ends, dtype=np.int64),
        compressor=compressor,
    )
    print(f"Saved {num_episodes} episodes ({total_steps} transitions) to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin demonstrations to a DSSP Zarr dataset.")
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    args = parser.parse_args()
    policy_root = Path(__file__).resolve().parents[1]
    repository_root = policy_root.parents[1]
    load_dir = repository_root / "data" / args.task_name / args.task_config / "data"
    save_dir = policy_root / "data" / f"{args.task_name}-{args.task_config}-{args.expert_data_num}.zarr"
    process_demonstrations(load_dir, save_dir, args.expert_data_num)


if __name__ == "__main__":
    main()
