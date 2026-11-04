if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
import sys
from train_dssp import TrainDSSPWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("dssp", "config")),
)
def main(cfg):
    workspace = TrainDSSPWorkspace(cfg)
    workspace.eval()


class DSSP:

    def __init__(self, cfg, usr_args) -> None:
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, usr_args)

    def update_obs(self, observation, executed_action=None):
        self.env_runner.update_obs(observation, policy=self.policy, executed_action=executed_action)

    def get_action(self, observation=None):
        if (
            getattr(self.env_runner, "use_cached_mamba_eval", False)
            and hasattr(self.policy, "predict_action_streaming")
            and hasattr(self.policy, "advance_eval_streaming_state")
            and getattr(self.policy, "streaming_eval_enabled", True)
        ):
            action = self.env_runner.get_action_streaming(self.policy, observation)
        else:
            action = self.env_runner.get_action(self.policy, observation)
        return action

    def get_runtime_metrics(self):
        metrics = self.env_runner.get_runtime_metrics()
        total_parameters = sum(parameter.numel() for parameter in self.policy.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.policy.parameters() if parameter.requires_grad
        )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.policy.parameters()
        )
        metrics["parameters"] = {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "storage_bytes": parameter_bytes,
            "storage_mib": parameter_bytes / (1024 ** 2),
        }
        metrics["policy_class"] = (
            f"{self.policy.__class__.__module__}.{self.policy.__class__.__qualname__}"
        )
        return metrics







    def reset(self):
        self.env_runner.reset_streaming_eval_state(self.policy)

    def get_policy_and_runner(self, cfg, usr_args):
        workspace = TrainDSSPWorkspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, usr_args)
        return policy, env_runner


if __name__ == "__main__":
    main()
