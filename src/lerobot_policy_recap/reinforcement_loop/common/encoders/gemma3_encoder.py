"""Gemma-3 Encoder: SigLIP vision + Gemma-3-270m language.

This encoder is fully independent from PI05 and combines:
- google/siglip-so400m-patch14-384 for vision
- google/gemma-3-270m-pt for language embeddings
"""

import copy
import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM

from lerobot_policy_recap.reinforcement_loop.common.encoders.encoder_configs import (
    Gemma3EncoderConfig,
    EncoderOutputDims,
)
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch, PI05Policy
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
)


# Gemma-3 specific constants - these match OBS_LANGUAGE_* + "_gemma3" suffix
# The RecapTokenizerProcessorStep produces these by appending suffix to base keys
OBS_LANGUAGE_TOKENS_GEMMA3 = "observation.language.tokens_gemma3"
OBS_LANGUAGE_ATTENTION_MASK_GEMMA3 = "observation.language.attention_mask_gemma3"


def freeze_(m: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for p in m.parameters():
        p.requires_grad = False


class LightweightResampler(nn.Module):
    """Lightweight cross-attention resampler (~2M params).
    
    Uses learnable queries with a single cross-attention layer (no self-attention).
    Much lighter than full TransformerDecoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_output_tokens: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.output_dim = output_dim
        
        # Learnable query tokens (directly in output_dim space)
        self.queries = nn.Parameter(torch.randn(num_output_tokens, output_dim) * 0.02)
        
        # Project input to output dim for cross-attention
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.input_norm = nn.LayerNorm(output_dim)
        
        # Single cross-attention layer (queries attend to projected input)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(output_dim)
        
        # Small MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.mlp_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, input_dim] - input tokens
            
        Returns:
            [B, num_output_tokens, output_dim] - resampled tokens
        """
        B = x.shape[0]
        
        # Project input
        kv = self.input_norm(self.input_proj(x))  # [B, N, output_dim]
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, output_dim]
        
        # Cross-attention: queries attend to input
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = self.cross_attn_norm(queries + attn_out)
        
        # MLP
        out = self.mlp_norm(queries + self.mlp(queries))
        
        return out


class ConvPoolingProjector(nn.Module):
    """Conv1D-based token reduction (~1.5M params).
    
    Uses strided 1D convolution to reduce token count, then projects to output dim.
    Very efficient and lightweight.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_input_tokens: int = 256,
        num_output_tokens: int = 64,
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.output_dim = output_dim
        
        # Calculate stride to reduce tokens
        # For 256 -> 64, stride = 4
        self.stride = num_input_tokens // num_output_tokens
        
        # Conv1D for token reduction (operates on sequence dimension)
        # kernel_size = stride * 2 - 1 for overlapping receptive field
        kernel_size = self.stride * 2 - 1
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(output_dim)
        
        # Small MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.mlp_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, input_dim] - input tokens
            
        Returns:
            [B, num_output_tokens, output_dim] - reduced tokens
        """
        # Conv1d expects [B, C, L], so transpose
        x = x.transpose(1, 2)  # [B, input_dim, N]
        
        # Apply strided conv
        x = self.conv(x)  # [B, output_dim, N']
        
        # Transpose back
        x = x.transpose(1, 2)  # [B, N', output_dim]
        
        # Ensure exact output size (may differ slightly due to padding)
        if x.shape[1] > self.num_output_tokens:
            x = x[:, :self.num_output_tokens, :]
        elif x.shape[1] < self.num_output_tokens:
            # Pad if needed (shouldn't happen with correct padding)
            pad = torch.zeros(x.shape[0], self.num_output_tokens - x.shape[1], x.shape[2], 
                            device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Normalize and MLP
        x = self.norm(x)
        x = self.mlp_norm(x + self.mlp(x))
        
        return x


class Gemma3ValueVLMEncoder(nn.Module):
    """
    Frozen SigLIP + Frozen Gemma-3, trainable projector + <VAL> token (+ optional soft prompts).
    Outputs: <VAL> final hidden state [B, D]
    """

    def __init__(
        self,
        config: Gemma3EncoderConfig,
        pi05: PI05Policy,
        input_features: dict
    ):
        super().__init__()

        self.config = config
        self.pi05_config = pi05.config
        self.input_features = input_features

        # --- Load models directly ---
        self.vision_tower = copy.deepcopy(pi05.model.paligemma_with_expert.paligemma.vision_tower)

        model_kwargs = {"torch_dtype": torch.bfloat16}
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("[Gemma3Encoder] Using Flash Attention 2")
            
        gemma_full = AutoModelForCausalLM.from_pretrained(
            config.gemma3_model_name, 
            **model_kwargs
        )
        self.gemma = gemma_full.model

        self.vision_dim = self.vision_tower.config.hidden_size
        self.hidden_dim = self.gemma.config.hidden_size

        # --- Projector (trainable) ---
        if self.config.token_reduction_method == "resampler":
            self.vision_proj = LightweightResampler(
                input_dim=self.vision_dim,
                output_dim=self.hidden_dim,
                num_output_tokens=self.config.num_image_tokens,
                num_heads=8,
            )
        elif self.config.token_reduction_method == "conv":
            self.vision_proj = ConvPoolingProjector(
                input_dim=self.vision_dim,
                output_dim=self.hidden_dim,
                num_input_tokens=256,
                num_output_tokens=self.config.num_image_tokens,
            )
        elif self.config.token_reduction_method == "none":
            self.vision_proj = nn.Sequential(
                nn.Linear(self.vision_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            raise ValueError(f"Unknown token_reduction_method={self.config.token_reduction_method}")

        # --- Optional soft prompt tokens (trainable) ---
        self.num_soft_prompt_tokens = int(self.config.num_soft_prompt_tokens)
        if self.num_soft_prompt_tokens > 0:
            self.soft_prompts = nn.Parameter(torch.randn(self.num_soft_prompt_tokens, self.hidden_dim) * 0.02)
        else:
            self.soft_prompts = None

        # --- <VAL> token (trainable) ---
        self.val_token = nn.Parameter(torch.randn(1, self.hidden_dim) * 0.02)

        # Freeze backbones
        if config.freeze_siglip:
            freeze_(self.vision_tower)
        if config.freeze_gemma:
            freeze_(self.gemma)

        # Cast trainable components to bfloat16 to match frozen backbones
        self.vision_proj.to(dtype=torch.bfloat16)
        if self.soft_prompts is not None:
            self.soft_prompts.data = self.soft_prompts.data.to(dtype=torch.bfloat16)
        self.val_token.data = self.val_token.data.to(dtype=torch.bfloat16)

    @property
    def output_dim(self) -> int:
        """Return output dimension (Gemma hidden size)."""
        return self.hidden_dim

    def get_optim_params(self) -> list:
        """Return trainable parameters (projector, soft prompts, val token)."""
        params = list(self.vision_proj.parameters())
        if self.soft_prompts is not None:
            params.append(self.soft_prompts)
        params.append(self.val_token)
        # Add Gemma params if not frozen
        if not self.config.freeze_gemma:
            params.extend(self.gemma.parameters())
        return params
    
    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

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
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def encode_images(self, images: List[Tensor], img_masks: List[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            image_outputs = self.vision_tower(img)
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.vision_proj(selected_image_feature)
            bsize, num_img_embs = image_features.shape[:2]

            embs.append(image_features)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        return embs, pad_masks, att_masks

    def encode_text(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        tokens = batch[OBS_LANGUAGE_TOKENS_GEMMA3]
        attn = batch.get(OBS_LANGUAGE_ATTENTION_MASK_GEMMA3, None)
        if attn is None:
            attn = torch.ones_like(tokens)

        tokens = tokens.to(next(self.parameters()).device)
        attn = attn.to(tokens.device)

        # embed_tokens is part of frozen gemma; we can do it without no_grad (cheap),
        # but if gemma frozen it doesn't matter either way.
        with torch.no_grad():
            emb = self.gemma.embed_tokens(tokens)  # [B,L,D]
        return emb, attn

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        t0 = time.time()
        device = next(self.parameters()).device
        gemma_dtype = next(self.gemma.parameters()).dtype  # bfloat16
        images, img_masks = self._preprocess_images(batch)
        t1 = time.time()

        img_embs, img_pad_masks, img_att_masks = self.encode_images(images, img_masks)  # [B, I, D]
        t2 = time.time()

        txt_embs, txt_mask = self.encode_text(batch)  # [B, L, D], [B, L]
        t3 = time.time()

        B = txt_embs.shape[0]

        parts = []
        masks = []

        # Soft prompts (prefix)
        if self.soft_prompts is not None:
            sp = self.soft_prompts.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
            parts.append(sp)
            masks.append(torch.ones(B, sp.shape[1], device=device, dtype=txt_mask.dtype))

        # Image tokens
        if img_embs is not None:
            parts.append(img_embs.to(device))
            # Use the actual padding mask from encode_images, cast to match text mask
            masks.append(img_pad_masks.to(device=device, dtype=txt_mask.dtype))

        # Text tokens
        parts.append(txt_embs.to(device))
        masks.append(txt_mask)

        # <VAL> token at the end
        val = self.val_token.unsqueeze(0).expand(B, 1, -1)  # [B,1,D]
        parts.append(val)
        masks.append(torch.ones(B, 1, device=device, dtype=txt_mask.dtype))

        inputs_embeds = torch.cat(parts, dim=1).to(gemma_dtype)
        attention_mask = torch.cat(masks, dim=1)
        t4 = time.time()

        # Log token count once
        if not hasattr(self, "_logged_token_count"):
            print(f"[Gemma3Encoder] Total input tokens: {inputs_embeds.shape[1]} (Batch size: {B})")
            img_tokens = img_embs.shape[1] if img_embs is not None else 0
            txt_tokens = txt_embs.shape[1]
            sp_tokens = self.num_soft_prompt_tokens
            print(f"  Breakdown: Soft Prompts={sp_tokens} | Images={img_tokens} | Text={txt_tokens} | <VAL>=1")
            self._logged_token_count = True

        out = self.gemma(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=False)
        t5 = time.time()
        
        print(f"[VLM Time] Prep: {t1-t0:.3f}s | Vision: {t2-t1:.3f}s | Text: {t3-t2:.3f}s | Assemble: {t4-t3:.3f}s | Gemma: {t5-t4:.3f}s")
        
        h = out.last_hidden_state  # [B, T, D]

        # Return <VAL> hidden state
        h_val = h[:, -1, :].float()
        return h_val


class Gemma3Encoder(nn.Module):
    """
    Gemma-3 based encoder wrapper.
    
    Outputs dictionary of features for critic consumption.
    """

    def __init__(self, config: Gemma3EncoderConfig, pi05: PI05Policy, input_features: dict):
        super().__init__()
        self.config = config
        self.input_features = input_features
        
        self.vl_encoder = Gemma3ValueVLMEncoder(config, pi05=pi05, input_features=input_features)
        
        # State/env handling
        self.has_env = OBS_ENV_STATE in input_features
        self.has_state = OBS_STATE in input_features
        self.max_state_dim = 128  # Standard max

    @property
    def output_dims(self) -> EncoderOutputDims:
        """Return dimensions for each feature type."""
        return EncoderOutputDims(
            image_dim=self.vl_encoder.output_dim,
            state_dim=self.max_state_dim if self.has_state else 0,
            env_dim=self.input_features[OBS_ENV_STATE].shape[0] if self.has_env else 0,
        )

    def get_optim_params(self) -> list:
        """Return trainable parameters from vl_encoder."""
        return self.vl_encoder.get_optim_params()

    def forward(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Returns dict with:
            - "image_features": VLM pooled output [B, gemma_hidden_dim]
            - "state_features": Padded state [B, max_state_dim] (if state used)
            - "env_features": Env state [B, env_dim] (if env used)
        """
        features = {}
        
        # VLM features - freezing is handled by requires_grad on parameters
        # Projector can train while siglip/gemma are frozen
        features["image_features"] = self.vl_encoder(obs)
        
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
