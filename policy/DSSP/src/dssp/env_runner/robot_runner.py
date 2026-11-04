import wandb
import numpy as np
import torch
import tqdm

from dssp.policy.base_policy import BasePolicy
from dssp.common.pytorch_util import dict_apply
from dssp.env_runner.base_runner import BaseRunner
import dssp.common.logger_util as logger_util
from termcolor import cprint
import pdb
from queue import deque
from dssp.common.metric_util import ModelRuntimeTracker


class RobotRunner(BaseRunner):

    def __init__(
        self,
        output_dir=None,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        use_cached_mamba_eval=False,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        task_name=None
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.use_cached_mamba_eval = use_cached_mamba_eval

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.obs = deque(maxlen=n_obs_steps + 1)
        self.prev_action = deque(maxlen=n_obs_steps + 1)
        self._obs_since_plan = []
        self._streaming_plan_active = False
        self.env = None
        self.runtime_tracker = ModelRuntimeTracker()

    def _policy_requires_prev_action(self, policy: BasePolicy = None) -> bool:
        return bool(policy is not None and getattr(policy, "requires_prev_action_history", False))

    def _zero_action_like_obs(self, policy: BasePolicy, current_obs):
        action_shape = (policy.action_dim,)
        if current_obs is not None:
            for value in current_obs.values():
                if isinstance(value, np.ndarray):
                    return np.zeros(action_shape, dtype=value.dtype)
                if isinstance(value, torch.Tensor):
                    return torch.zeros(action_shape, dtype=value.dtype)
        return np.zeros(action_shape, dtype=np.float32)

    def _coerce_action_like_reference(self, action, reference):
        if isinstance(reference, np.ndarray):
            return np.asarray(action, dtype=reference.dtype)
        if isinstance(reference, torch.Tensor):
            if isinstance(action, torch.Tensor):
                return action.to(dtype=reference.dtype)
            return torch.as_tensor(action, dtype=reference.dtype)
        raise RuntimeError(f"Unsupported action reference type {type(reference)}")

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()
        self.prev_action.clear()
        self._obs_since_plan = []
        self._streaming_plan_active = False




    def _build_single_obs_input(self, policy: BasePolicy, current_obs):
        np_obs_dict = self._build_obs_sequence([current_obs])
        if self._policy_requires_prev_action(policy):
            if len(self.prev_action) == 0:
                prev_action = self._zero_action_like_obs(policy, current_obs)
            else:
                prev_action = self.prev_action[-1]
            if isinstance(prev_action, np.ndarray):
                np_obs_dict["prev_action"] = np.expand_dims(prev_action, axis=0)
            elif isinstance(prev_action, torch.Tensor):
                np_obs_dict["prev_action"] = prev_action.unsqueeze(0)
            else:
                raise RuntimeError(f"Unsupported action type {type(prev_action)}")
        return self._to_policy_input(policy, np_obs_dict)








    def _build_obs_sequence(self, obs_sequence):
        assert len(obs_sequence) > 0
        result = dict()
        for key in obs_sequence[0].keys():
            values = [obs[key] for obs in obs_sequence]
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values, axis=0)
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                raise RuntimeError(f"Unsupported obs type {type(values[0])}")
        return result

    def _to_policy_input(self, policy: BasePolicy, np_obs_dict):
        device = policy.device
        obs_dict = dict_apply(
            np_obs_dict,
            lambda x: torch.from_numpy(x).to(device=device) if isinstance(x, np.ndarray) else x.to(device=device),
        )
        return {
            key: value.unsqueeze(0)
            for key, value in obs_dict.items()
        }

    def _advance_streaming_obs_chunks(self, policy: BasePolicy):
        if not self.use_cached_mamba_eval or not self._streaming_plan_active:
            return
        if not hasattr(policy, "advance_eval_streaming_state"):
            return

        while len(self._obs_since_plan) >= (2 * self.n_obs_steps):
            chunk = self._obs_since_plan[: self.n_obs_steps]
            np_obs_dict = self._build_obs_sequence(chunk)
            obs_dict_input = self._to_policy_input(policy, np_obs_dict)
            with torch.no_grad():
                policy.advance_eval_streaming_state(obs_dict_input)
            self._obs_since_plan = self._obs_since_plan[self.n_obs_steps :]

    def _advance_streaming_obs_single(self, policy: BasePolicy, current_obs):
        if not hasattr(policy, "advance_eval_streaming_state"):
            return
        obs_dict_input = self._build_single_obs_input(policy, current_obs)
        with torch.no_grad():
            policy.advance_eval_streaming_state(obs_dict_input)

    def _initialize_streaming_eval_state(self, policy: BasePolicy):
        if not hasattr(policy, "advance_eval_streaming_state"):
            return
        if len(self.obs) == 0:
            return
        obs = self.get_n_steps_obs(policy=policy)
        obs_dict_input = self._to_policy_input(policy, dict(obs))
        with torch.no_grad():
            policy.advance_eval_streaming_state(obs_dict_input)

    def update_obs(self, current_obs, policy: BasePolicy = None, executed_action=None):
        self.obs.append(current_obs)
        if self._policy_requires_prev_action(policy):
            if len(self.prev_action) == 0:
                aligned_prev_action = self._zero_action_like_obs(policy, current_obs)
            elif executed_action is not None:
                aligned_prev_action = self._coerce_action_like_reference(executed_action, self.prev_action[-1])
            else:
                aligned_prev_action = self.prev_action[-1]
            self.prev_action.append(aligned_prev_action)
        if self.use_cached_mamba_eval and self._streaming_plan_active:
            if policy is not None and getattr(policy, "streaming_eval_per_obs", False):
                self._advance_streaming_obs_single(policy, current_obs)
            else:
                self._obs_since_plan.append(current_obs)
                if policy is not None:
                    self._advance_streaming_obs_chunks(policy)

    def get_n_steps_obs(self, policy: BasePolicy = None):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)
        if self._policy_requires_prev_action(policy) and len(self.prev_action) > 0:
            result["prev_action"] = self.stack_last_n_obs(self.prev_action, self.n_obs_steps)

        return result

    def get_action(self, policy: BasePolicy, observaton=None, executed_action=None) -> bool:
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.update_obs(observaton, policy=policy, executed_action=executed_action)
        obs = self.get_n_steps_obs(policy=policy)
        obs_dict_input = self._to_policy_input(policy, dict(obs))
        # run policy
        with torch.no_grad():
            action_dict = self.runtime_tracker.record_inference(
                lambda: policy.predict_action(obs_dict_input)
            )

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)
        return action

    def get_action_streaming(self, policy: BasePolicy, observaton=None, executed_action=None):
        if observaton is not None:
            self.update_obs(observaton, policy=policy, executed_action=executed_action)
        if not self._streaming_plan_active and getattr(policy, "streaming_eval_per_obs", False):
            self._initialize_streaming_eval_state(policy)
        obs = self.get_n_steps_obs(policy=policy)
        obs_dict_input = self._to_policy_input(policy, dict(obs))
        with torch.no_grad():
            action_dict = self.runtime_tracker.record_inference(
                lambda: policy.predict_action_streaming(obs_dict_input)
            )

        self._streaming_plan_active = True
        self._obs_since_plan = []

        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        return np_action_dict["action"].squeeze(0)

    def reset_streaming_eval_state(self, policy: BasePolicy = None):
        self.reset_obs()
        if policy is not None and hasattr(policy, "reset_eval_streaming_state"):
            policy.reset_eval_streaming_state()

    def get_runtime_metrics(self):
        return self.runtime_tracker.summary()

    def run(self, policy: BasePolicy):
        pass


if __name__ == "__main__":
    test = RobotRunner("./")
    print("ready")
