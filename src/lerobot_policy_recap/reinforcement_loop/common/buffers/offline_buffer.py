import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Sequence, Callable
from lerobot.utils.constants import OBS_IMAGE, ACTION, REWARD, DONE, OBS_STR, TRUNCATED
from lerobot.datasets.utils import flatten_dict, check_delta_timestamps, get_delta_indices
from tqdm import tqdm


class OfflineReplayBuffer:
    """
    Offline Replay Buffer for action chunking with multi-step returns.
    
    This buffer wraps a LeRobotDataset and configures delta_timestamps for RL training.
    It's used as a PyTorch Dataset - the dataloader handles batching with default collation,
    and batch restructuring happens in OfflineTrainer._prepare_batch() on GPU.
    """
    
    def __init__(self, cfg, dataset, horizon: int):
        """
        Args:
            cfg: Configuration object
            dataset: LeRobotDataset instance
            horizon: Number of timesteps for action chunks (H)
        """
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.horizon = horizon
        self._update_dataset_deltas()
    
    def _update_dataset_deltas(self):
        """Update dataset with delta timestamps for temporal queries."""        
        # Build delta timestamps for observation at t and actions from t to t+H-1
        delta_timestamps = {}
        
        # Current observation at t=0 for all camera keys
        # Next observation at t=H-1 for all camera keys
        self.observation_keys = [key for key in self.dataset.meta.features.keys() if key.startswith(OBS_STR)]
        for key in self.observation_keys:
            # For each camera, we want frames at t=0 and t=H-1
            delta_timestamps[key] = [0.0, (self.horizon - 1) / self.dataset.fps]
        
        # Action chunk: from current timestep to horizon-1
        # e.g., if horizon=10, actions at [0, 1, 2, ..., 9] relative to current frame
        action_deltas = [i / self.dataset.fps for i in range(self.horizon)]
        delta_timestamps[ACTION] = action_deltas
        
        # Rewards for each step in the horizon
        delta_timestamps[REWARD] = action_deltas
        
        # Done flags for each step in the horizon (to detect done in between)
        delta_timestamps[DONE] = action_deltas
        
        self.dataset.delta_timestamps = delta_timestamps
            
        check_delta_timestamps(delta_timestamps, self.dataset.fps, self.dataset.tolerance_s)
        self.dataset.delta_indices = get_delta_indices(delta_timestamps, self.dataset.fps)
    
    def __len__(self):
        """Return total number of frames in dataset."""
        return len(self.dataset)

    def compute_return_to_go(self, discount: float):
        """
        Compute Return-to-Go for the entire dataset and store in RAM.
        
        Args:
            discount: Discount factor (gamma)
        """
        # Loading all data to RAM for computation might be heavy, but requested by user
        # We access underlying data from dataset.hf_dataset generally for efficiency
        # But LeRobotDataset abstraction might vary. We try to use batch access if possible.
        # Otherwise fall back to iterating.
        
        print("Computing Return-to-Go for offline buffer...")
        
        # Try to access whole columns if possible (assuming HuggingFace Dataset backend)
        if hasattr(self.dataset, "hf_dataset"):
            rewards = np.array(self.dataset.hf_dataset["next.reward"])
            dones = np.array(self.dataset.hf_dataset["next.done"])
            # Handling episode indices to respect boundaries
            episode_indices = np.array(self.dataset.hf_dataset["episode_index"])
        else:
            # Fallback: iterate (slower)
            print("Warning: iterating dataset to compute return-to-go (slow)")
            rewards = []
            dones = []
            episode_indices = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                rewards.append(item["next.reward"])
                dones.append(item["next.done"])
                episode_indices.append(item["episode_index"])
            rewards = np.array(rewards)
            dones = np.array(dones)
            episode_indices = np.array(episode_indices)

        # Compute RTG
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        # Reverse pass
        for t in tqdm(reversed(range(len(rewards))), total=len(rewards), desc="Computing returns"):
            # Reset if done OR if episode index changes (new episode starts at t+1)
            is_last_step = (t == len(rewards) - 1) or (episode_indices[t] != episode_indices[t+1])
            
            if is_last_step or dones[t]:
                running_return = 0.0
            
            running_return = rewards[t] + discount * running_return
            returns[t] = running_return
            
        self.return_to_go = torch.from_numpy(returns)
        
        # Compute return stats for normalization
        self.return_stats = {
            "mean": float(returns.mean()),
            "std": float(returns.std())
        }
        print(f"Return-to-Go computation finished. Stats: mean={self.return_stats['mean']:.4f}, std={self.return_stats['std']:.4f}, max = {returns.max():.4f}, min = {returns.min():.4f}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single transition - delegates directly to dataset.
        
        The returned dict has observation keys with shape [2, C, H, W] where:
        - index 0 is current observation (t=0)
        - index 1 is next observation (t=H-1)
        
        Args:
            idx: Index to sample
            
        Returns:
            Single sample dictionary from dataset
        """
        item = self.dataset[idx]
        
        if getattr(self, "return_to_go", None) is not None:
            item["return_to_go"] = self.return_to_go[idx]
            
        return item