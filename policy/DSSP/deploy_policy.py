# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, 'src'))

from dssp_policy import DSSP


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs


def get_model(usr_args):
    config_path = "./src/dssp/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.setting = usr_args["ckpt_setting"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    usr_args["use_cached_mamba_eval"] = _as_bool(usr_args.get("use_cached_mamba_eval", False))
    if "mamba_version" in usr_args and "mamba_version" in cfg.policy:
        cfg.policy.mamba_version = usr_args["mamba_version"]
    OmegaConf.set_struct(cfg, True)

    model = DSSP(cfg, usr_args)
    return model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs, executed_action=action)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    if hasattr(model, "reset"):
        model.reset()
    else:
        model.env_runner.reset_obs()
