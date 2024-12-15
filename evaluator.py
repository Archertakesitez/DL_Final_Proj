from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # print("Pred shape:", pred.shape)
    # print("Target shape:", target.shape)
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config

        self.model = model
        self.model.eval()

        self.quick_debug = quick_debug

        self.ds = probe_train_ds
        self.val_ds = probe_val_ds

        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        Probes whether the predicted embeddings capture the future locations
        """
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = []
        all_parameters += list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size
        batch_steps = None

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################################
                # TODO: Forward pass through your model
                init_states = batch.states[:, 0:1]  # BS, 1, C, H, W
                # print("Evaluator - init_states shape:", init_states.shape)
                # print("Evaluator - actions shape:", batch.actions.shape)

                pred_encs, _ = model(states=init_states, actions=batch.actions)
                pred_encs = pred_encs.transpose(0, 1)  # BS, T, D --> T, BS, D
                # Make sure pred_encs has shape (T, BS, D) at this point
                ################################################################################

                pred_encs = pred_encs.detach()
                # print("pred_encs shape:", pred_encs.shape)

                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                losses_list = []

                target = getattr(batch, "locations").cuda()
                target = self.normalizer.normalize_location(target)
                # print("target shape before sampling:", target.shape)

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    # we only randomly sample n timesteps to train prober.
                    # we most likely do this to avoid OOM
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )

                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 2)

                    for i in range(bs):
                        indices = torch.randperm(n_steps)[: config.sample_timesteps]
                        sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                        sampled_target_locs[i, :] = target[i, indices]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs.cuda()

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                # print("pred_locs final shape:", pred_locs.shape)
                # print("target final shape:", target.shape)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"normalized pred locations loss {per_probe_loss.item()}")

                losses_list.append(per_probe_loss)
                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(
        self,
        prober,
    ):
        """
        Evaluates on all the different validation datasets
        """
        avg_losses = {}

        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=prefix,
            )

        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        prober,
        val_ds,
        prefix="",
    ):
        quick_debug = self.quick_debug
        config = self.config
        model = self.model
        probing_losses = []
        prober.eval()

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            ################################################################################
            # TODO: Forward pass through your model
            init_states = batch.states[:, 0:1]  # [B, 1, C, H, W]
            pred_encs, _ = model(
                states=init_states, actions=batch.actions
            )  # Get predictions for future steps
            pred_encs = pred_encs.transpose(0, 1)  # [B, T, D] -> [T, B, D]
            # Make sure pred_encs has shape (T, BS, D) at this point
            ################################################################################

            # Process target
            target = getattr(batch, "locations").cuda()
            target = self.normalizer.normalize_location(target)

            # Make predictions using prober
            pred_locs = torch.stack(
                [prober(x) for x in pred_encs], dim=1
            )  # [B, T-1, 2]

            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

            # Visualize every N batches
            if idx % 10 == 0:
                os.makedirs("trajectory_plots", exist_ok=True)
                self.visualize_trajectories(
                    pred_locs=pred_locs,
                    target_locs=target,
                    walls=batch.states[:, 0, 1],
                    save_path=f"trajectory_plots/trajectory_pred_{prefix}_{idx}.png",
                )

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)
        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss

    def visualize_trajectories(
        self,
        pred_locs,
        target_locs,
        walls=None,
        batch_idx=0,
        save_path="trajectory_pred.png",
    ):
        """
        Save trajectory visualization plots to a directory for later downloading from HPC
        """
        # Get multiple trajectories
        num_trajectories = min(4, pred_locs.shape[0])  # Plot up to 4 trajectories

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()

        for i in range(num_trajectories):
            pred_path = pred_locs[i].cpu().numpy()
            true_path = target_locs[i].cpu().numpy()

            # Debug prints for each trajectory
            print(f"\nTrajectory {i}:")
            print(f"Predicted path shape: {pred_path.shape}")
            print(f"True path shape: {true_path.shape}")
            print(
                f"Predicted coordinates min/max: ({pred_path.min():.3f}, {pred_path.max():.3f})"
            )
            print(
                f"True coordinates min/max: ({true_path.min():.3f}, {true_path.max():.3f})"
            )
            print(f"Predicted path points:\n{pred_path}")
            print(f"True path points:\n{true_path}")

            ax = axes[i]

            # Plot walls if provided
            if walls is not None:
                wall_layout = walls[i].cpu().numpy()
                ax.imshow(wall_layout, cmap="gray", alpha=0.3)

            # Plot trajectories with markers
            ax.plot(
                pred_path[:, 0],
                pred_path[:, 1],
                "r.-",
                label="Predicted",
                linewidth=2,
                markersize=5,
            )
            ax.plot(
                true_path[:, 0],
                true_path[:, 1],
                "b.-",
                label="Ground Truth",
                linewidth=2,
                markersize=5,
            )

            # Plot start and end points
            ax.scatter(
                pred_path[0, 0],
                pred_path[0, 1],
                c="red",
                marker="o",
                s=100,
                label="Start (Pred)",
            )
            ax.scatter(
                pred_path[-1, 0],
                pred_path[-1, 1],
                c="red",
                marker="x",
                s=100,
                label="End (Pred)",
            )
            ax.scatter(
                true_path[0, 0],
                true_path[0, 1],
                c="blue",
                marker="o",
                s=100,
                label="Start (True)",
            )
            ax.scatter(
                true_path[-1, 0],
                true_path[-1, 1],
                c="blue",
                marker="x",
                s=100,
                label="End (True)",
            )

            ax.legend()
            ax.set_title(f"Trajectory {i}")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.grid(True)

            # Set axis limits with some padding
            all_x = np.concatenate([pred_path[:, 0], true_path[:, 0]])
            all_y = np.concatenate([pred_path[:, 1], true_path[:, 1]])
            ax.set_xlim([all_x.min() - 1, all_x.max() + 1])
            ax.set_ylim([all_y.min() - 1, all_y.max() + 1])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
