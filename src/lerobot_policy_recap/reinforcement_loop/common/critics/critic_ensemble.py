"""Critic ensemble networks for Q and V value estimation."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot_policy_recap.reinforcement_loop.common.critics.critics import (
    AdaptiveFusion,
    CriticHead,
    DistributionalCriticHead,
)



class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble for Q-value estimation.
    
    Q(s, a) takes both observations and actions as input.

    Args:
        encoder (nn.Module): encoder for observations (outputs feature dictionary).
        fusion (AdaptiveFusion): fusion layer to process encoder outputs.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: nn.Module,
        fusion: AdaptiveFusion,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def get_optim_params(self) -> list:
        """Return trainable parameters from encoder, fusion, and critic heads."""
        params = []
        # Encoder params (respects freeze settings)
        if hasattr(self.encoder, 'get_optim_params'):
            params.extend(self.encoder.get_optim_params())
        # Fusion params (always trainable)
        if hasattr(self.fusion, 'get_optim_params'):
            params.extend(self.fusion.get_optim_params())
        # Critic head params (always trainable)  
        for critic in self.critics:
            if hasattr(critic, 'get_optim_params'):
                params.extend(critic.get_optim_params())
        return params

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for Q-value estimation.
        
        Args:
            batch: Dictionary containing:
                - Keys starting with OBS_PREFIX (e.g., 'observation.image', 'observation.state')
                - ACTION: action tensor
        
        Returns:
            Q-values tensor of shape [num_critics, batch_size]
        """
        # Extract observations and actions from batch
        device = get_device_from_parameters(self)
        
        # Try to use cached features if available
        features = {}
        prefix = getattr(self, "cache_prefix", "")
        for key in ["image_features", "state_features", "env_features"]:
            cache_key = f"{OBS_PREFIX}cache.{prefix}{key}"
            if cache_key in batch:
                features[key] = batch[cache_key].to(device)
            elif f"{OBS_PREFIX}cache.{key}" in batch:
                # Fallback to generic key if prefixed not found
                features[key] = batch[f"{OBS_PREFIX}cache.{key}"].to(device)
            
        if not features:
            # Fallback to encoder if no cached features found
            observations = {k: v.to(device) for k, v in batch.items() if k.startswith(OBS_PREFIX)}
            features = self.encoder(observations)
            
        actions = batch[ACTION].to(device)
        fused = self.fusion(features)
        inputs = torch.cat([fused, actions], dim=-1)
        
        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class VCriticEnsemble(nn.Module):
    """
    VCriticEnsemble wraps multiple CriticHead modules into an ensemble.
    It estimates V(s), so it only takes observations as input (no actions).

    Args:
        encoder (nn.Module): encoder for observations (outputs feature dictionary).
        fusion (AdaptiveFusion): fusion layer to process encoder outputs.
        ensemble (List[CriticHead]): list of v_critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_v_critics, batch_size) containing V-values.
    """

    def __init__(
        self,
        encoder: nn.Module,
        fusion: AdaptiveFusion,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.init_final = init_final
        self.v_critics = nn.ModuleList(ensemble)

    def get_optim_params(self) -> list:
        """Return trainable parameters from encoder, fusion, and v_critic heads."""
        params = []
        # Encoder params (respects freeze settings)
        if hasattr(self.encoder, 'get_optim_params'):
            params.extend(self.encoder.get_optim_params())
        # Fusion params (always trainable)
        if hasattr(self.fusion, 'get_optim_params'):
            params.extend(self.fusion.get_optim_params())
        # V-critic head params (always trainable)
        for critic in self.v_critics:
            if hasattr(critic, 'get_optim_params'):
                params.extend(critic.get_optim_params())
        return params

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for V-value estimation.
        
        Args:
            batch: Dictionary containing:
                - Keys starting with OBS_PREFIX (e.g., 'observation.image', 'observation.state')
        
        Returns:
            V-values tensor of shape [num_v_critics, batch_size]
        """
        # Extract observations from batch
        device = get_device_from_parameters(self)
        
        # Try to use cached features if available
        features = {}
        prefix = getattr(self, "cache_prefix", "")
        for key in ["image_features", "state_features", "env_features"]:
            cache_key = f"{OBS_PREFIX}cache.{prefix}{key}"
            if cache_key in batch:
                features[key] = batch[cache_key].to(device)
            elif f"{OBS_PREFIX}cache.{key}" in batch:
                # Fallback to generic key if prefixed not found
                features[key] = batch[f"{OBS_PREFIX}cache.{key}"].to(device)
                
        if not features:
            # Fallback to encoder if no cached features found
            observations = {k: v.to(device) for k, v in batch.items() if k.startswith(OBS_PREFIX)}
            features = self.encoder(observations)
            
        inputs = self.fusion(features)

        # Loop through v_critics and collect outputs
        v_values = []
        for v_critic in self.v_critics:
            v_values.append(v_critic(inputs))

        # Stack outputs to match expected shape [num_v_critics, batch_size]
        return torch.stack(v_values, dim=0).squeeze(-1)


class DistributionalVCriticEnsemble(nn.Module):
    """Ensemble of distributional V-critics.
    
    Models Z(s) as a discrete distribution over atoms[cite: 75].
    """

    def __init__(
        self,
        encoder: nn.Module,
        fusion: AdaptiveFusion,
        ensemble: list[DistributionalCriticHead],
        v_min: float,
        v_max: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.v_critics = nn.ModuleList(ensemble)
        
        # Distributional parameters 
        self.num_atoms = ensemble[0].num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("atoms", torch.linspace(v_min, v_max, self.num_atoms))
        self.delta_z = (v_max - v_min) / (self.num_atoms - 1)

    def get_optim_params(self) -> list:
        """Return trainable parameters from encoder, fusion, and v_critic heads."""
        params = []
        # Encoder params (respects freeze settings)
        if hasattr(self.encoder, 'get_optim_params'):
            params.extend(self.encoder.get_optim_params())
        # Fusion params (always trainable)
        if hasattr(self.fusion, 'get_optim_params'):
            params.extend(self.fusion.get_optim_params())
        # V-critic head params (always trainable)
        for critic in self.v_critics:
            if hasattr(critic, 'get_optim_params'):
                params.extend(critic.get_optim_params())
        return params

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns: Probabilities of shape [num_critics, batch_size, num_atoms]"""
        device = get_device_from_parameters(self)
        
        # Feature extraction logic (identical to your Q/V ensembles)
        features = {}
        prefix = getattr(self, "cache_prefix", "")
        for key in ["image_features", "state_features", "env_features"]:
            cache_key = f"{OBS_PREFIX}cache.{prefix}{key}"
            if cache_key in batch:
                features[key] = batch[cache_key].to(device)
                
        if not features:
            observations = {k: v.to(device) for k, v in batch.items() if k.startswith(OBS_PREFIX)}
            features = self.encoder(observations)
            
        inputs = self.fusion(features)

        # Collect distributions from each head [cite: 282]
        logits = torch.stack([critic(inputs) for critic in self.v_critics], dim=0)
        return logits

    def get_expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes the expected value (mean) of the distributions[cite: 265]."""
        # logits: [num_critics, batch_size, num_atoms]
        return torch.sum(F.softmax(logits, dim=-1) * self.atoms, dim=-1)    