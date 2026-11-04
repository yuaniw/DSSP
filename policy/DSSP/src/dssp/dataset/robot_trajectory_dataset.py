import copy
import os
import sys
from typing import Dict

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, ".."))
sys.path.append(os.path.join(parent_directory, "../.."))

from dssp.common.pytorch_util import dict_apply
from dssp.common.replay_buffer import ReplayBuffer
from dssp.dataset.base_dataset import BaseDataset
from dssp.model.common.normalizer import LinearNormalizer


def _get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def _downsample_mask(mask, max_n, seed=0):
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
    return train_mask


class _RobotReplayBufferWindowDataset(BaseDataset):
    def __init__(
        self,
        replay_buffer,
        horizon,
        pad_before,
        pad_after,
        episode_mask,
        include_prev_action=False,
        task_name=None,
    ):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_pad_before = min(max(self.pad_before, 0), self.horizon - 1)
        self.sequence_pad_after = min(max(self.pad_after, 0), self.horizon - 1)
        self.include_prev_action = include_prev_action
        self.task_name = task_name
        self.train_mask = np.asarray(episode_mask, dtype=bool)
        self.indices = self._build_indices(self.train_mask)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.indices = self._build_indices(val_set.train_mask)
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        if self.include_prev_action:
            data["prev_action"] = self.replay_buffer["action"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.indices)

    def _episode_bounds(self, episode_id):
        start_idx = 0 if episode_id == 0 else int(self.replay_buffer.episode_ends[episode_id - 1])
        end_idx = int(self.replay_buffer.episode_ends[episode_id])
        return start_idx, end_idx

    def _build_indices(self, episode_mask):
        indices = []
        for episode_id in np.nonzero(episode_mask)[0]:
            start_idx, end_idx = self._episode_bounds(int(episode_id))
            episode_length = end_idx - start_idx
            min_start = -self.sequence_pad_before
            max_start = episode_length - self.horizon + self.sequence_pad_after
            for idx in range(min_start, max_start + 1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx + self.horizon, episode_length) + start_idx
                start_offset = buffer_start_idx - (idx + start_idx)
                end_offset = (idx + self.horizon + start_idx) - buffer_end_idx
                sample_start_idx = start_offset
                sample_end_idx = self.horizon - end_offset
                indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
        if len(indices) == 0:
            return np.zeros((0, 4), dtype=np.int64)
        return np.asarray(indices, dtype=np.int64)

    def _get_episode_start_idx(self, buffer_start_idx):
        episode_idx = np.searchsorted(self.replay_buffer.episode_ends, buffer_start_idx, side="right")
        if episode_idx == 0:
            return 0
        return int(self.replay_buffer.episode_ends[episode_idx - 1])

    def _build_prev_action_sequence(self, action, sampler_index):
        prev_action = np.zeros_like(action, dtype=np.float32)
        if action.shape[0] <= 1:
            return prev_action

        prev_action[1:] = action[:-1]
        buffer_start_idx, _buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[sampler_index]
        if sample_end_idx <= sample_start_idx:
            return prev_action

        episode_start_idx = self._get_episode_start_idx(buffer_start_idx)
        if buffer_start_idx > episode_start_idx:
            prev_action[sample_start_idx] = self.replay_buffer["action"][buffer_start_idx - 1].astype(np.float32)
        else:
            prev_action[sample_start_idx] = 0.0

        if sample_start_idx > 0:
            prev_action[:sample_start_idx] = prev_action[sample_start_idx]
        if sample_end_idx < action.shape[0]:
            prev_action[sample_end_idx:] = prev_action[sample_end_idx - 1]
        return prev_action

    def _sample_to_data(self, sample, sampler_index=None):
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }
        if self.include_prev_action:
            assert sampler_index is not None
            data["obs"]["prev_action"] = self._build_prev_action_sequence(
                data["action"],
                sampler_index=sampler_index,
            )
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._sample_sequence(idx)
        data = self._sample_to_data(sample, sampler_index=idx)
        return dict_apply(data, torch.from_numpy)

    def _sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = {}
        for key in ("state", "action", "point_cloud"):
            input_arr = self.replay_buffer[key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
                data = np.zeros(
                    shape=(self.horizon,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.horizon:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


class RobotTrajectoryDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        include_prev_action=False,
    ):
        super().__init__()
        self.task_name = task_name
        self.include_prev_action = include_prev_action
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.sequence_pad_before = min(max(self.pad_before, 0), self.horizon - 1)
        self.sequence_pad_after = min(max(self.pad_after, 0), self.horizon - 1)

        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["state", "action", "point_cloud"])

        val_mask = _get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = _downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask
        train_episode_ids = np.nonzero(self.train_mask)[0].astype(np.int64)
        self.episode_ids = np.asarray(
            [
                episode_id
                for episode_id in train_episode_ids
                if self._episode_window_starts(self._episode_bounds(int(episode_id))[1] - self._episode_bounds(int(episode_id))[0]).size
                > 0
            ],
            dtype=np.int64,
        )

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        if self.include_prev_action:
            data["prev_action"] = self.replay_buffer["action"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def get_validation_dataset(self):
        return _RobotReplayBufferWindowDataset(
            replay_buffer=self.replay_buffer,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            include_prev_action=self.include_prev_action,
            task_name=self.task_name,
        )

    def __len__(self) -> int:
        return len(self.episode_ids)

    def _episode_bounds(self, episode_id):
        start_idx = 0 if episode_id == 0 else int(self.replay_buffer.episode_ends[episode_id - 1])
        end_idx = int(self.replay_buffer.episode_ends[episode_id])
        return start_idx, end_idx

    def _episode_window_starts(self, episode_length):
        min_start = -self.sequence_pad_before
        max_start = episode_length - self.horizon + self.sequence_pad_after
        return np.arange(min_start, max_start + 1, dtype=np.int64)

    def _build_prev_action(self, action):
        prev_action = np.zeros_like(action, dtype=np.float32)
        if action.shape[0] > 1:
            prev_action[1:] = action[:-1]
        return prev_action

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_id = int(self.episode_ids[idx])
        start_idx, end_idx = self._episode_bounds(episode_id)
        action = self.replay_buffer["action"][start_idx:end_idx].astype(np.float32)
        agent_pos = self.replay_buffer["state"][start_idx:end_idx].astype(np.float32)
        point_cloud = self.replay_buffer["point_cloud"][start_idx:end_idx].astype(np.float32)
        window_starts = self._episode_window_starts(len(action))

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": action,
            "window_starts": window_starts,
            "episode_length": np.asarray(len(action), dtype=np.int64),
            "episode_id": np.asarray(episode_id, dtype=np.int64),
        }
        if self.include_prev_action:
            data["obs"]["prev_action"] = self._build_prev_action(action)
        return dict_apply(data, torch.from_numpy)
