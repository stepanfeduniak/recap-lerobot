from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
import torch.nn as nn
import torch
import numpy as np

from lerobot.processor.core import TransitionKey

def make_policy(
    cfg: PreTrainedConfig,
    input_features: dict | None = None,
    output_features: dict | None = None,
) -> PreTrainedPolicy:
    kwargs = {}

    policy_cls = get_policy_class(cfg.type)


    cfg.output_features = output_features
    cfg.input_features = input_features
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy

def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "dsrl-na":
        from lerobot.policies.dsrl.modeling_dsrl_na import DSRLNAPolicy

        return DSRLNAPolicy
    if name == "sac-serl":
        from lerobot.policies.sac.modeling_sac_serl import SACSerlPolicy

        return SACSerlPolicy

def merge_batches(online_batch: dict, offline_batch: dict) -> dict:
    """
    Recursively merges two batches. Tensors are concatenated along dim 0.
    Enforces that both batches share the same structure/keys.
    """
    if not isinstance(online_batch, dict) or not isinstance(offline_batch, dict):
        # If we reached this point and they aren't both dicts, but are tensors, cat them.
        # Otherwise, if types mismatch, the protocol is broken.
        if isinstance(online_batch, torch.Tensor) and isinstance(offline_batch, torch.Tensor):
            if online_batch.ndim != offline_batch.ndim:
                 raise ValueError(
                    f"Dimension mismatch for concatenation: "
                    f"Online {online_batch.shape} vs Offline {offline_batch.shape}"
                )
            return torch.cat([online_batch, offline_batch], dim=0)
        elif isinstance(online_batch, list) and isinstance(offline_batch, list):
            return online_batch + offline_batch
        else:
            raise TypeError(f"Structure mismatch: online is {type(online_batch)}, offline is {type(offline_batch)}")

    # Check if keys match exactly per protocol
    if online_batch.keys() != offline_batch.keys():
        raise KeyError(
            f"Batch structure mismatch.\n"
            f"Online keys: {sorted(online_batch.keys())}\n"
            f"Offline keys: {sorted(offline_batch.keys())}"
        )

    merged_batch = {}
    for key in online_batch:
        merged_batch[key] = merge_batches(online_batch[key], offline_batch[key])
    
    return merged_batch


def make_transition_obs(batch: dict, device: torch.device = "cpu") -> dict:
        
    return batch


def add_temporal_dim_to_obs(obs_dict: dict, n_obs_steps: int = 2) -> dict:
    """Add temporal dimension to observations for diffusion policy.
    
    Diffusion policies expect observations with shape [B, n_obs_steps, ...] but
    RL buffers store single timestep observations with shape [B, ...].
    This function repeats the observation across the temporal dimension.
    
    Args:
        obs_dict: Dictionary of observations with shape [B, ...]
        n_obs_steps: Number of observation steps to repeat (default: 2)
        
    Returns:
        Dictionary with observations of shape [B, n_obs_steps, ...]
    """
    temporal_obs = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            # Add temporal dimension and repeat: [B, ...] -> [B, n_obs_steps, ...]
            temporal_obs[key] = value.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (value.ndim - 1)))
        else:
            temporal_obs[key] = value
    return temporal_obs

def drop_non_observation_keys(batch: dict) -> dict:
    for key in list(batch.keys()):
        if not key.startswith("observation.") and not key.startswith("next_observation."):
            batch.pop(key)
    return batch

def to_cpu(x):
    """Recursively move tensors to CPU (and detach), keep non-tensors unchanged."""
    if torch.is_tensor(x):
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_cpu(v) for v in x]
        return type(x)(t)
    return x


def to_device(x, device):
    """Recursively move tensors to device, keep non-tensors unchanged."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_device(v, device) for v in x]
        return type(x)(t)
    return x
