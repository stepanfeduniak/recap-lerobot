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
    Normalizes return_to_go and next.reward using return statistics (max_abs).
    
    Both return_to_go and rewards are normalized using the SAME return statistics,
    meaning rewards will be scaled consistently with returns.
    
    The return_to_go is expected to be in complementary_data to avoid being dropped
    by other processor steps.
    
    Attributes:
        return_stats: Dictionary containing "max", "min" or "max_abs" keys for return normalization.
        eps: Small epsilon for numerical stability.
    """
    
    return_stats: dict[str, float] | None = None
    eps: float = 1e-8
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    
    # Internal tensor stats
    _tensor_stats: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """Convert stats to tensors after initialization."""
        if self.dtype is None:
            self.dtype = torch.float32
        self._update_tensor_stats()
    
    def _update_tensor_stats(self):
        """Convert return_stats to tensors on the correct device."""
        if self.return_stats is None:
            self._tensor_stats = {}
            return
        
        # Determine max_abs from available stats
        # Priority: max_abs > max/min > mean/std (fallback, though mean/std shouldn't be used for this logic)
        max_abs_val = 1.0
        
        if "max_abs" in self.return_stats:
            max_abs_val = self.return_stats["max_abs"]
        elif "max" in self.return_stats and "min" in self.return_stats:
            max_val = abs(self.return_stats["max"])
            min_val = abs(self.return_stats["min"])
            max_abs_val = max(max_val, min_val)
        
        # Avoid division by zero if max_abs is 0 (unlikely but possible)
        if max_abs_val < self.eps:
            max_abs_val = 1.0

        self._tensor_stats = {
            "max_abs": torch.tensor(max_abs_val, device=self.device, dtype=self.dtype),
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
            "return.max_abs": self._tensor_stats["max_abs"].cpu(),
        }
    
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load state dictionary."""
        if "return.max_abs" in state:
            self.return_stats = {
                "max_abs": state["return.max_abs"].item(),
            }
            self._update_tensor_stats()
    
    def get_config(self) -> dict[str, Any]:
        """Return serializable configuration."""
        return {
            "return_stats": self.return_stats,
            "eps": self.eps,
        }
    
    def update_stats(self, return_stats: dict[str, float]) -> None:
        """Update return statistics."""
        self.return_stats = return_stats
        self._update_tensor_stats()
    
    def _normalize(self, value: Tensor) -> Tensor:
        """Apply max_abs normalization: value / max_abs."""
        if not self._tensor_stats:
            return value
        
        max_abs = self._tensor_stats["max_abs"]
        
        # Ensure stats are on the same device as input
        if max_abs.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            max_abs = self._tensor_stats["max_abs"]
        
        return value / (max_abs + self.eps)
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Normalize return_to_go and next.reward in the transition.
        
        Looks for return_to_go in complementary_data, normalizes it in place.
        Normalizes next.reward at the top level.
        """
        new_transition = transition.copy()
        
        if not self._tensor_stats:
            return new_transition
        
        # Normalize return_to_go from complementary_data
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data is not None and "return_to_go" in comp_data:
            rtg = comp_data["return_to_go"]
            if rtg is not None:
                rtg_tensor = torch.as_tensor(rtg, dtype=self.dtype)
                comp_data["return_to_go"] = self._normalize(rtg_tensor)
        
        # Normalize reward at top level (next.reward is typically stored as "reward" key
        # after batch preparation, but we check both patterns)
        reward = new_transition.get(TransitionKey.REWARD)
        if reward is not None:
            reward_tensor = torch.as_tensor(reward, dtype=self.dtype)
            new_transition[TransitionKey.REWARD] = self._normalize(reward_tensor)
        
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
    
    Inverse of ReturnNormalizerProcessorStep: value * max_abs.
    """
    
    return_stats: dict[str, float] | None = None
    eps: float = 1e-8
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    
    _tensor_stats: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32
        self._update_tensor_stats()
    
    def _update_tensor_stats(self):
        if self.return_stats is None:
            self._tensor_stats = {}
            return
        
        # Determine max_abs from available stats
        max_abs_val = 1.0
        
        if "max_abs" in self.return_stats:
            max_abs_val = self.return_stats["max_abs"]
        elif "max" in self.return_stats and "min" in self.return_stats:
            max_val = abs(self.return_stats["max"])
            min_val = abs(self.return_stats["min"])
            max_abs_val = max(max_val, min_val)
            
        if max_abs_val < self.eps:
            max_abs_val = 1.0
        
        self._tensor_stats = {
            "max_abs": torch.tensor(max_abs_val, device=self.device, dtype=self.dtype),
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
            "return.max_abs": self._tensor_stats["max_abs"].cpu(),
        }
    
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "return.max_abs" in state:
            self.return_stats = {
                "max_abs": state["return.max_abs"].item(),
            }
            self._update_tensor_stats()
    
    def get_config(self) -> dict[str, Any]:
        return {
            "return_stats": self.return_stats,
            "eps": self.eps,
        }
    
    def update_stats(self, return_stats: dict[str, float]) -> None:
        self.return_stats = return_stats
        self._update_tensor_stats()
    
    def _unnormalize(self, value: Tensor) -> Tensor:
        """Apply max_abs unnormalization: value * max_abs."""
        if not self._tensor_stats:
            return value
        
        max_abs = self._tensor_stats["max_abs"]
        
        if max_abs.device != value.device:
            self.to(device=value.device, dtype=value.dtype)
            max_abs = self._tensor_stats["max_abs"]
        
        return value * max_abs
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        
        if not self._tensor_stats:
            return new_transition
        
        # Unnormalize return_to_go from complementary_data
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data is not None and "return_to_go" in comp_data:
            rtg = comp_data["return_to_go"]
            if rtg is not None:
                rtg_tensor = torch.as_tensor(rtg, dtype=self.dtype)
                comp_data["return_to_go"] = self._unnormalize(rtg_tensor)
        
        # Unnormalize reward
        reward = new_transition.get(TransitionKey.REWARD)
        if reward is not None:
            reward_tensor = torch.as_tensor(reward, dtype=self.dtype)
            new_transition[TransitionKey.REWARD] = self._unnormalize(reward_tensor)
        
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Return features unchanged."""
        return features
