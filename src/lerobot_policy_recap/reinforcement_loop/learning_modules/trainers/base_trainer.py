"""
Base Trainer class with shared functionality for all RL trainers.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator

import torch
from torch.utils.data import DataLoader

from lerobot.utils.constants import OBS_STR
from lerobot.datasets.utils import cycle


class BaseTrainer(ABC):
    """
    Abstract base class for all RL trainers.
    
    Provides common functionality for batch preparation, data loading,
    and training loops. Subclasses must implement setup-specific methods.
    """
    
    def __init__(self, rl: Any):
        """
        Initialize the base trainer.
        
        Args:
            rl: RLObjects instance containing policy, buffers, config, etc.
        """
        self.rl = rl
        self.reward_model = rl.reward_model
        self.last_losses: Dict[str, float] = {}
        self.observation_keys: list = []
        
        # These will be set by subclasses during setup
        self.accelerator = None
        self.device = None
    
    def _setup_accelerator(self):
        """Setup accelerator and device."""
        self.accelerator = self.rl.accelerator
        self.device = self.accelerator.device
    
    def _create_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with standard settings.
        
        Args:
            dataset: The dataset to load from
            batch_size: Batch size for the dataloader
            num_workers: Number of workers (defaults to cfg.num_workers)
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            Configured DataLoader
        """
        if num_workers is None:
            num_workers = self.rl.cfg.num_workers
            
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            pin_memory=self.device.type == "cuda",
            drop_last=drop_last,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    
    def _prepare_policy_and_optimizers(self, *dataloaders):
        """
        Prepare policy, optimizers, and dataloaders with accelerator.
        
        Args:
            *dataloaders: Variable number of dataloaders to prepare
            
        Returns:
            Tuple of prepared dataloaders
        """
        opt_dict = self.rl.policy.get_optimizers()
        
        # Prepare policy, dataloaders, and all optimizers
        prepared = self.accelerator.prepare(
            self.rl.policy,
            *dataloaders,
            *opt_dict.values()
        )
        
        # Reassign prepared policy
        self.rl.policy = prepared[0]
        
        # Extract prepared dataloaders
        num_dataloaders = len(dataloaders)
        prepared_dataloaders = prepared[1:1 + num_dataloaders]
        
        # Reassign prepared optimizers back to the policy
        for i, opt_name in enumerate(opt_dict.keys()):
            setattr(self.rl.policy, opt_name, prepared[1 + num_dataloaders + i])
        
        return prepared_dataloaders if num_dataloaders > 1 else (prepared_dataloaders[0] if num_dataloaders == 1 else None)
    
    def _get_batch(self, iterator: Iterator, dataloader: DataLoader) -> dict:
        """
        Get next batch from iterator, cycling if needed.
        
        Args:
            iterator: The batch iterator
            dataloader: The dataloader to recreate iterator from
            
        Returns:
            Next batch from the iterator
        """
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator)
    
    def _prepare_batch(self, raw_batch: dict) -> dict:
        """
        Prepare batch for policy training: restructure, apply reward model, preprocess.
        
        Raw batch from default collation has observation keys with shape [B, 2, C, H, W]
        where index 0 is current obs and index 1 is next obs. This function splits them
        and processes everything on GPU.
        
        Args:
            raw_batch: Raw batch from dataloader (default collation, already on GPU)
            
        Returns:
            Preprocessed batch ready for policy.training_step()
        """
        batch = {}
        next_obs_batch = {}
        
        # Split observation tensors: [B, 2, C, H, W] -> current [B, C, H, W] and next [B, C, H, W]
        for key in self.observation_keys:
            obs_data = raw_batch[key]  # [B, 2, C, H, W]
            batch[key] = obs_data[:, 0]  # Current obs
            next_obs_batch[key] = obs_data[:, 1]  # Next obs (renamed without "next." for preprocessor)
        
        # Action chunks [B, H, D]
        batch["action"] = raw_batch["action"]
        
        # Rewards [B, H] - apply reward model
        batch["next.reward"] = self.reward_model.get_chunked_rewards(raw_batch["next.reward"])
        
        # Done flags - mark as done if any timestep in the chunk is done
        done_data = raw_batch["next.done"]
        if done_data.dim() == 1:
            done_data = done_data.unsqueeze(1)
        # If extended over time, check if at least one value is done
        if done_data.dim() == 2 and done_data.shape[1] > 1:
            done_data = done_data.any(dim=1).float()
        elif done_data.dim() == 3:
            done_data = done_data.any(dim=1).squeeze(-1).float()
        else:
            done_data = done_data.squeeze(-1).float()
        batch["next.done"] = done_data
        
        # Truncated flags
        if "next.truncated" in raw_batch:
            batch["next.truncated"] = raw_batch["next.truncated"].float()
        else:
            batch["next.truncated"] = torch.zeros_like(batch["next.done"]).float()        
        # Complementary data
        batch["task"] = raw_batch["task"]
        batch["episode_index"] = raw_batch["episode_index"]
        
        # Padding masks
        if "action_is_pad" in raw_batch:
            batch["action_is_pad"] = raw_batch["action_is_pad"]
        if "reward_is_pad" in raw_batch:
            batch["reward_is_pad"] = raw_batch["reward_is_pad"]
        
        # Add complementary data to next_obs_batch for processor compatibility
        next_obs_batch["task"] = raw_batch["task"]
        next_obs_batch["episode_index"] = raw_batch["episode_index"]

        # Pass return-to-go if present (e.g. for Recap policy)
        if "return_to_go" in raw_batch:
            batch["return_to_go"] = raw_batch["return_to_go"]

        # Process current observations and actions
        batch = self.rl.preprocessor(batch)
        
        # Process next observations (keys are already without "next." prefix)
        if next_obs_batch:
            processed_next = self.rl.preprocessor(next_obs_batch)
            # Add back with "next." prefix
            for k, v in processed_next.items():
                if k.startswith(OBS_STR):
                    batch[f"next.{k}"] = v
        
        return batch
    
    def n_training_steps(self, n_steps: int) -> Dict[str, float]:
        """
        Executes multiple training steps in sequence.
        
        Args:
            n_steps: Number of training iterations to perform.
            
        Returns:
            A dictionary of the last losses from the final step.
        """
        for step in range(n_steps):
            global_step = self.rl.step + 1 + step
            self.training_step(global_step)
        
        return self.last_losses
    
    def _log_training_metrics(self, losses: Dict[str, float], step: int, prefix: str = "train"):
        """
        Log training metrics if logging interval is reached.
        
        Args:
            losses: Dictionary of loss values
            step: Current training step
            prefix: Prefix for metric names
        """
        if step % self.rl.cfg.log_every == 0:
            self.rl.logger.log_metrics(losses, step=step, prefix=prefix)
    
    @abstractmethod
    def training_step(self, step: int) -> Dict[str, float]:
        """
        Executes a single update of the policy.
        
        Args:
            step: The current global iteration count.
            
        Returns:
            A dictionary of losses for logging.
        """
        pass
