"""SigLIP Encoder: Standalone SigLIP vision encoder.

This encoder uses only the SigLIP vision tower without Gemma language model.
Much lighter weight than the full Gemma3Encoder, suitable for vision-only tasks.
"""

import copy
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot_policy_recap.reinforcement_loop.common.encoders.encoder_configs import (
    SigLIPEncoderConfig,
    EncoderOutputDims,
)
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch, PI05Policy
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
)


def freeze_(m: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for p in m.parameters():
        p.requires_grad = False


class SigLIPVisionEncoder(nn.Module):
    """
    Standalone SigLIP vision encoder with trainable projector.
    
    SigLIP has no CLS token - uses mean pooling over all patch tokens.
    We apply mean pooling first, then project to output dim.
    
    Outputs: Pooled vision features [B, projection_dim]
    """

    def __init__(
        self,
        config: SigLIPEncoderConfig,
        pi05: PI05Policy,
    ):
        super().__init__()

        self.config = config
        self.pi05_config = pi05.config
        self.input_features = pi05.config.input_features

        # --- Load SigLIP vision tower from PI05 ---
        self.vision_tower = copy.deepcopy(pi05.model.paligemma_with_expert.paligemma.vision_tower)

        self.vision_dim = self.vision_tower.config.hidden_size

        # Freeze backbone if configured
        if config.freeze_siglip:
            freeze_(self.vision_tower)

    @property
    def output_dim(self) -> int:
        """Return output dimension (projection_dim)."""
        return self.vision_dim

    def get_optim_params(self) -> list:
        """Return trainable parameters (projector)."""
        params = []
        # Add SigLIP params if not frozen
        if not self.config.freeze_siglip:
            params.extend(self.vision_tower.parameters())
        return params
    
    def _preprocess_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.pi05_config.image_features if key in batch]
        missing_img_keys = [key for key in self.pi05_config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.pi05_config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # DEBUG: Check input stats
            if not hasattr(self, "_logged_input_stats"):
                print(f"[SigLIP Debug] Input stats for {key}:")
                print(f"  Shape: {img.shape}")
                print(f"  Dtype: {img.dtype}")
                print(f"  Range: [{img.min().item():.3f}, {img.max().item():.3f}]")
                print(f"  Mean: {img.mean().item():.3f}")
                self._logged_input_stats = True
                
                # Check for unexpected range
                if img.max() > 1.1:
                    print(f"[SigLIP Debug] WARNING: Input max > 1.1. Is it [0, 255]?")
                if img.min() < -0.1:
                    print(f"[SigLIP Debug] WARNING: Input min < -0.1. Is it already normalized to [-1, 1]?")

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.pi05_config.image_resolution:
                img = resize_with_pad_torch(img, *self.pi05_config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            
        return images

    def encode_images(self, images: List[Tensor]) -> Tensor:
        """Encode images through SigLIP, mean pool, then project.
        
        SigLIP has no CLS token, so we mean pool all patch tokens per image.
        """
        pooled_features = []
        # Process images
        for img in images:
            image_outputs = self.vision_tower(img)
            # last_hidden_state: [B, num_patches, vision_dim]
            patch_features = image_outputs.last_hidden_state
            
            # Mean pool over patches (SigLIP has no CLS token)
            # Shape: [B, vision_dim]
            pooled = patch_features.mean(dim=1)
            
            pooled_features.append(pooled)

        # Stack and average across cameras
        # Shape: [num_cameras, B, hidden_dim] -> [B, hidden_dim]
        stacked = torch.stack(pooled_features, dim=0)
        # Average over cameras
        output = stacked.mean(dim=0)
        
        return output

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        t0 = time.time()
        
        images = self._preprocess_images(batch)
        t1 = time.time()

        # encode_images now returns pooled output directly [B, hidden_dim]
        pooled = self.encode_images(images)
        t2 = time.time()

        B = pooled.shape[0]
        
        t3 = time.time()
        
        print(f"[SigLIP Time] Prep: {t1-t0:.3f}s | Vision+Pool: {t2-t1:.3f}s")
        
        # Log once
        if not hasattr(self, "_logged_info"):
            print(f"[SigLIPEncoder] Output dim: {pooled.shape[-1]} (Batch size: {B})")
            print(f"[SigLIPEncoder] Using mean pooling (SigLIP has no CLS token)")
            self._logged_info = True

        return pooled.float()


class SigLIPEncoder(nn.Module):
    """
    Standalone SigLIP encoder wrapper.
    
    Outputs dictionary of features for critic consumption.
    """

    def __init__(self, config: SigLIPEncoderConfig, pi05: PI05Policy):
        super().__init__()
        self.config = config
        self.input_features = pi05.config.input_features
        
        self.vision_encoder = SigLIPVisionEncoder(config, pi05=pi05)
        
        # State/env handling
        self.has_env = OBS_ENV_STATE in self.input_features
        self.has_state = OBS_STATE in self.input_features
        self.max_state_dim = 128  # Standard max

    @property
    def output_dims(self) -> EncoderOutputDims:
        """Return dimensions for each feature type."""
        return EncoderOutputDims(
            image_dim=self.vision_encoder.output_dim,
            state_dim=self.max_state_dim if self.has_state else 0,
            env_dim=self.input_features[OBS_ENV_STATE].shape[0] if self.has_env else 0,
        )

    def get_optim_params(self) -> list:
        """Return trainable parameters from vision_encoder."""
        return self.vision_encoder.get_optim_params()

    def forward(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Returns dict with:
            - "image_features": SigLIP pooled output [B, projection_dim]
            - "state_features": Padded state [B, max_state_dim] (if state used)
            - "env_features": Env state [B, env_dim] (if env used)
        """
        features = {}
        
        # Vision features
        features["image_features"] = self.vision_encoder(obs)
        
        # 2. State features (pass-through with padding)
        if self.has_state:
            state = obs[OBS_STATE]
            if state.ndim > 2:  # [B, T, D] -> [B, D]
                state = state[:, -1, :]
            # Pad to max dim
            if state.shape[-1] < self.max_state_dim:
                state = F.pad(state, (0, self.max_state_dim - state.shape[-1]))
            features["state_features"] = state
        
        # 3. Env features (pass-through)
        if self.has_env:
            features["env_features"] = obs[OBS_ENV_STATE]
        
        return features
