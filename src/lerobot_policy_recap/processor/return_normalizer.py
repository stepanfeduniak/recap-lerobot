#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Return Normalizer Processor Step for RECAP.

Normalizes `return_to_go` and `next.reward` using return statistics (mean/std).
Both are normalized using the same statistics derived from returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="return_normalizer_processor")
class ReturnNormalizerProcessorStep(ProcessorStep):
    """
    Normalizes return_to_go and next.reward using reward statistics scaled by horizon.
    
    Instead of normalizing by return statistics directly, this normalizer:
    1. Uses reward statistics (mean/std) as the base
    2. Scales by the horizon term 1/(1-discount) to convert reward scale to return scale
    
    This ensures proper normalization where:
    - Rewards are normalized as: (reward - reward_mean) / reward_std
    - Returns are normalized as: (return - reward_mean * horizon) / (reward_std * horizon)
      which simplifies to: (return / horizon - reward_mean) / reward_std
    
    The return_to_go is expected to be in complementary_data to avoid being dropped
    by other processor steps.
    
    Attributes:
        reward_stats: Dictionary with 'mean' and 'std' keys for reward normalization.
        discount: Discount factor for computing horizon = 1/(1-discount).
        eps: Small epsilon for numerical stability.
    """
    
    reward_stats: dict[str, float] | None = None
    discount: float = 0.99
    eps: float = 1e-8
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    
    # Internal tensor stats
    _tensor_stats: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)
    _horizon: float = field(default=100.0, init=False, repr=False)
    
    def __post_init__(self):
        """Convert stats to tensors after initialization."""
        if self.dtype is None:
            self.dtype = torch.float32
        # Compute horizon from discount: H = 1/(1-gamma)
        self._horizon = 1.0 / (1.0 - self.discount + self.eps)
        self._update_tensor_stats()
    
    def _update_tensor_stats(self):
        """Convert reward_stats to tensors on the correct device."""
        if self.reward_stats is None:
            self._tensor_stats = {}
            return
        
        self._tensor_stats = {
            "mean": torch.tensor(self.reward_stats["mean"], device=self.device, dtype=self.dtype),
            "std": torch.tensor(self.reward_stats["std"], device=self.device, dtype=self.dtype),
            "horizon": torch.tensor(self._horizon, device=self.device, dtype=self.dtype),
        }
    
    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ReturnNormalizerProcessorStep":
        """Move stats to the specified device."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._update_tensor_stats()
        return self
    
    def state_dict(self) -> dict[str, Tensor]:
        """Return state dictionary for serialization."""
        if not self._tensor_stats:
            return {}
        return {
            "reward.mean": self._tensor_stats["mean"].cpu(),
            "reward.std": self._tensor_stats["std"].cpu(),
            "horizon": self._tensor_stats["horizon"].cpu(),
        }
    
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dictionary."""
        if "reward.mean" in state and "reward.std" in state:
            self.reward_stats = {
                "mean": state["reward.mean"].item(),
                "std": state["reward.std"].item(),
            }
            if "horizon" in state:
                self._horizon = state["horizon"].item()
            self._update_tensor_stats()
    
    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "reward_stats": self.reward_stats,
            "discount": self.discount,
            "eps": self.eps,
        }
    
    def update_stats(self, reward_stats: dict[str, float]) -> None:
        """Update reward statistics."""
        self.reward_stats = reward_stats
        self._update_tensor_stats()
    
    def _normalize_reward(self, value: Tensor) -> Tensor:
        """Apply reward normalization: (reward - mean) / std."""
        if not self._tensor_stats:
            return value
        
        mean = self._tensor_stats["mean"]
        std = self._tensor_stats["std"]
        
        # Ensure stats are on the same device as input
        if mean.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            mean = self._tensor_stats["mean"]
            std = self._tensor_stats["std"]
        
        return (value - mean) / (std + self.eps)
    
    def _normalize_return(self, value: Tensor) -> Tensor:
        """
        Apply return normalization using reward stats and horizon.
        
        Formula: (return / horizon - reward_mean) / reward_std
        This is equivalent to dividing return by horizon first, then normalizing
        using reward statistics.
        """
        if not self._tensor_stats:
            return value
        
        mean = self._tensor_stats["mean"]
        std = self._tensor_stats["std"]
        horizon = self._tensor_stats["horizon"]
        
        # Ensure stats are on the same device as input
        if mean.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            mean = self._tensor_stats["mean"]
            std = self._tensor_stats["std"]
            horizon = self._tensor_stats["horizon"]
        
        # Divide by horizon first to convert return scale to reward scale
        scaled_return = value / horizon
        # Then normalize using reward statistics
        return (scaled_return - mean) / (std + self.eps)
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Normalize return_to_go and next.reward in the transition.
        
        Looks for return_to_go in complementary_data, normalizes it in place.
        Normalizes next.reward at the top level.
        """
        new_transition = transition.copy()
        
        if not self._tensor_stats:
            return new_transition
        
        # Normalize return_to_go from complementary_data using horizon-scaled normalization
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data is not None and "return_to_go" in comp_data:
            rtg = comp_data["return_to_go"]
            if rtg is not None:
                rtg_tensor = torch.as_tensor(rtg, dtype=self.dtype)
                comp_data["return_to_go"] = self._normalize_return(rtg_tensor)
        
        # Normalize reward at top level using reward normalization
        reward = new_transition.get(TransitionKey.REWARD)
        if reward is not None:
            reward_tensor = torch.as_tensor(reward, dtype=self.dtype)
            new_transition[TransitionKey.REWARD] = self._normalize_reward(reward_tensor)
        
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Return features unchanged."""
        return features


@dataclass
@ProcessorStepRegistry.register(name="return_unnormalizer_processor")
class ReturnUnnormalizerProcessorStep(ProcessorStep):
    """
    Unnormalizes return_to_go and rewards back to original scale.
    
    Inverse of ReturnNormalizerProcessorStep.
    
    For rewards: reward = normalized_reward * reward_std + reward_mean
    For returns: return = (normalized_return * reward_std + reward_mean) * horizon
    """
    
    reward_stats: dict[str, float] | None = None
    discount: float = 0.99
    eps: float = 1e-8
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    
    _tensor_stats: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)
    _horizon: float = field(default=100.0, init=False, repr=False)
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32
        # Compute horizon from discount: H = 1/(1-gamma)
        self._horizon = 1.0 / (1.0 - self.discount + self.eps)
        self._update_tensor_stats()
    
    def _update_tensor_stats(self):
        if self.reward_stats is None:
            self._tensor_stats = {}
            return
        
        self._tensor_stats = {
            "mean": torch.tensor(self.reward_stats["mean"], device=self.device, dtype=self.dtype),
            "std": torch.tensor(self.reward_stats["std"], device=self.device, dtype=self.dtype),
            "horizon": torch.tensor(self._horizon, device=self.device, dtype=self.dtype),
        }
    
    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ReturnUnnormalizerProcessorStep":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._update_tensor_stats()
        return self
    
    def state_dict(self) -> dict[str, Tensor]:
        if not self._tensor_stats:
            return {}
        return {
            "reward.mean": self._tensor_stats["mean"].cpu(),
            "reward.std": self._tensor_stats["std"].cpu(),
            "horizon": self._tensor_stats["horizon"].cpu(),
        }
    
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "reward.mean" in state and "reward.std" in state:
            self.reward_stats = {
                "mean": state["reward.mean"].item(),
                "std": state["reward.std"].item(),
            }
            if "horizon" in state:
                self._horizon = state["horizon"].item()
            self._update_tensor_stats()
    
    def get_config(self) -> dict[str, Any]:
        return {
            "reward_stats": self.reward_stats,
            "discount": self.discount,
            "eps": self.eps,
        }
    
    def update_stats(self, reward_stats: dict[str, float]) -> None:
        self.reward_stats = reward_stats
        self._update_tensor_stats()
    
    def _unnormalize_reward(self, value: Tensor) -> Tensor:
        """Unnormalize reward: reward = normalized * std + mean."""
        if not self._tensor_stats:
            return value
        
        mean = self._tensor_stats["mean"]
        std = self._tensor_stats["std"]
        
        if mean.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            mean = self._tensor_stats["mean"]
            std = self._tensor_stats["std"]
        
        return value * std + mean
    
    def _unnormalize_return(self, value: Tensor) -> Tensor:
        """
        Unnormalize return using reward stats and horizon.
        
        Formula: return = (normalized_return * reward_std + reward_mean) * horizon
        This is the inverse of: normalized = (return / horizon - mean) / std
        """
        if not self._tensor_stats:
            return value
        
        mean = self._tensor_stats["mean"]
        std = self._tensor_stats["std"]
        horizon = self._tensor_stats["horizon"]
        
        if mean.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            mean = self._tensor_stats["mean"]
            std = self._tensor_stats["std"]
            horizon = self._tensor_stats["horizon"]
        
        # First unnormalize using reward statistics
        unnorm_scaled = value * std + mean
        # Then multiply by horizon to convert back to return scale
        return unnorm_scaled * horizon
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        
        if not self._tensor_stats:
            return new_transition
        
        # Unnormalize return_to_go from complementary_data using horizon scaling
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data is not None and "return_to_go" in comp_data:
            rtg = comp_data["return_to_go"]
            if rtg is not None:
                rtg_tensor = torch.as_tensor(rtg, dtype=self.dtype)
                comp_data["return_to_go"] = self._unnormalize_return(rtg_tensor)
        
        # Unnormalize reward using reward statistics
        reward = new_transition.get(TransitionKey.REWARD)
        if reward is not None:
            reward_tensor = torch.as_tensor(reward, dtype=self.dtype)
            new_transition[TransitionKey.REWARD] = self._unnormalize_reward(reward_tensor)
        
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Return features unchanged."""
        return features
