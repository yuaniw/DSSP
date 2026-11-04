if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import pathlib

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader

from train_dssp import TrainDSSPWorkspace
from dssp.dataset.base_dataset import BaseDataset
from dssp.model.common.lr_scheduler import get_scheduler
from dssp.model.diffusion.ema_model import EMAModel
from dssp.common.metric_util import (
    RunningStats,
    peak_memory_mb,
    reset_peak_memory_stats,
    synchronize_if_needed,
    write_metrics,
)
from dssp.common.pytorch_util import dict_apply, optimizer_to

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDSSPTrajectoryWorkspace(TrainDSSPWorkspace):
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        train_step_time_ms = RunningStats("train_step_time_ms")
        train_peak_memory_mb = RunningStats("train_peak_memory_mb")
        WANDB = str(cfg.logging.mode).lower() not in {"disabled", "none", "off"}

        if cfg.dataloader.batch_size != 1:
            raise ValueError(
                f"Trajectory training expects dataloader.batch_size=1, got {cfg.dataloader.batch_size}."
            )

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        cfg.logging.name = str(cfg.task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir})

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        for _ in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = []
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    synchronize_if_needed()
                    reset_peak_memory_stats()
                    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    t2 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    if t1 is not None:
                        t1.record()

                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    raw_loss, loss_dict = self.model.compute_loss_trajectory(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(loss_dict)

                    synchronize_if_needed()
                    if t2 is not None:
                        t2.record()
                        torch.cuda.synchronize(device)
                        train_step_time_ms.add(t1.elapsed_time(t2))
                    train_peak_memory_mb.add(peak_memory_mb())

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            step_log["train_loss"] = float(np.mean(train_losses)) if train_losses else None
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            if (self.epoch % cfg.training.val_every) == 0:
                with torch.no_grad():
                    val_losses = []
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, _ = policy.compute_loss(batch)
                            val_losses.append(loss)
                            if cfg.training.max_val_steps is not None and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if val_losses:
                        step_log["val_loss"] = torch.mean(torch.tensor(val_losses)).item()

            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                checkpoint_dir = self._checkpoint_dir(cfg, cfg.training.seed)
                checkpoint_dir = pathlib.Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_path = checkpoint_dir.joinpath(f"{self.epoch + 1}.ckpt")
                self.save_checkpoint(save_path)

            policy.train()
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        train_metrics = {
            "config_name": cfg.name,
            "policy_target": cfg.policy._target_,
            "mamba_version": cfg.policy.get("mamba_version", None),
            "train_step_time_ms": train_step_time_ms.summary(),
            "train_peak_memory_mb": train_peak_memory_mb.summary(),
        }
        write_metrics(pathlib.Path(self.output_dir).joinpath("train_metrics.json"), train_metrics)
        print(f"Training metrics: {train_metrics}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("dssp", "config")),
)
def main(cfg):
    workspace = TrainDSSPTrajectoryWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
