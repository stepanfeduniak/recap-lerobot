"""RECAP PI policy using Gemma3 encoder with PI05."""

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot_policy_recap.reinforcement_loop.common.encoders.gemma3_encoder import Gemma3Encoder
from lerobot_policy_recap.policies.recap.configuration_recap_pi import RECAP_PI_Config
from lerobot_policy_recap.policies.recap.modeling_recap_base import RECAPBasePolicy, OBS_LANGUAGE_TOKENS_POS, OBS_LANGUAGE_ATTENTION_MASK_POS

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)
import types



class RECAP_PI_Policy(RECAPBasePolicy):
    """RECAP policy variant for PI05 with Gemma3 encoder."""
    config_class = RECAP_PI_Config
    name = "recap_pi"

    def __init__(
        self,
        config: RECAP_PI_Config | None = None,
        **kwargs: Unpack[dict],
    ):
        super().__init__(config)
        print(f"Kwargs received in RECAP_PI_Policy: {kwargs.keys()}")
        print("[RECAP_PI_Policy] Initializing RECAP PI policy")
        self.diffusion_policy.model.sample_actions_cfg = types.MethodType(
            sample_actions_cfg,
            self.diffusion_policy.model,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor]:
        if self.cfg_weight == 1:
            return super().predict_action_chunk(batch)
        self.diffusion_policy.eval()

        # Prepare inputs
        images, img_masks = self.diffusion_policy._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        tokens_pos, masks_pos = batch[f"{OBS_LANGUAGE_TOKENS_POS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK_POS}"]

        # Sample actions using the CFG-enabled model
        actions = self.diffusion_policy.model.sample_actions_cfg(
            images, img_masks, tokens, masks, tokens_pos, masks_pos, cfg_weight=self.cfg_weight
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :self.action_chunk_len, :original_action_dim]

        return actions

    def _init_encoders(self):
        """Initialize Gemma3Encoder for v_critic."""
        print("[RECAP_PI_Policy] Initializing Gemma3Encoder for PI05 policy")
        self.encoder_v_critic = Gemma3Encoder(
            config=self.config.gemma3_encoder_config,
            pi05=self.diffusion_policy,
            input_features=self.config.input_features,
        )
        print(f"[RECAP_PI_Policy] Gemma3Encoder output dims: {self.encoder_v_critic.output_dims}")
            
        # Log trainable params in encoder
        trainable = sum(p.numel() for p in self.encoder_v_critic.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder_v_critic.parameters())
        print(f"[RECAP_PI_Policy] Encoder params: {trainable:,} trainable / {total:,} total params")



@torch.no_grad()
def sample_actions_cfg(
    self,
    images,
    img_masks,
    tokens,
    masks,
    tokens_pos,
    masks_pos,
    noise=None,
    num_steps=None,
    cfg_weight=1.0,
    **kwargs,
) -> Tensor:
    """Do a full inference forward with Classifier Free Guidance.
    
    CFG formula: v_cfg = v_neutral + cfg_weight * (v_positive - v_neutral)
    
    This batches the neutral and positive conditioning together for efficiency,
    then splits the results to compute the CFG combination.
    """
    if num_steps is None:
        num_steps = self.config.num_inference_steps

    bsize = tokens.shape[0]
    device = tokens.device

    if noise is None:
        # Sample noise with padded dimension as expected by action_in_proj
        actions_shape = (
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        noise = self.sample_noise(actions_shape, device)

    # Embed neutral (base) conditioning
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Embed positive conditioning
    prefix_embs_pos, prefix_pad_masks_pos, prefix_att_masks_pos = self.embed_prefix(images, img_masks, tokens_pos, masks_pos)
    prefix_att_2d_masks_pos = make_att_2d_masks(prefix_pad_masks_pos, prefix_att_masks_pos)
    prefix_position_ids_pos = torch.cumsum(prefix_pad_masks_pos, dim=1) - 1

    # Concatenate along batch dimension for efficient batched forward pass
    # Shape: [2*bsize, seq_len, hidden_dim]
    batched_prefix_embs = torch.cat([prefix_embs, prefix_embs_pos], dim=0)
    batched_prefix_pad_masks = torch.cat([prefix_pad_masks, prefix_pad_masks_pos], dim=0)
    batched_prefix_att_2d_masks = torch.cat([prefix_att_2d_masks, prefix_att_2d_masks_pos], dim=0)
    batched_prefix_position_ids = torch.cat([prefix_position_ids, prefix_position_ids_pos], dim=0)

    # Prepare 4D attention masks
    batched_prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(batched_prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    # Forward pass with batched inputs
    _, batched_past_key_values = self.paligemma_with_expert.forward(
        attention_mask=batched_prefix_att_2d_masks_4d,
        position_ids=batched_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[batched_prefix_embs, None],
        use_cache=True,
    )

    # Split past_key_values along the batch dimension
    # Each layer's key/value has shape [batch, heads, seq_len, head_dim]
    # We need to split batch dimension in half: [:bsize] for neutral, [bsize:] for positive
    def split_past_key_values(pkv, split_size):
        """Split past_key_values tuple along batch dimension."""
        split_pkv = []
        for layer_kv in pkv:
            # layer_kv is a tuple of (key, value) tensors
            key, value = layer_kv
            key_neutral, key_pos = key[:split_size], key[split_size:]
            value_neutral, value_pos = value[:split_size], value[split_size:]
            split_pkv.append(((key_neutral, value_neutral), (key_pos, value_pos)))
        # Reorganize: list of ((k_n, v_n), (k_p, v_p)) -> (list of (k_n, v_n), list of (k_p, v_p))
        neutral_pkv = tuple((layer[0][0], layer[0][1]) for layer in split_pkv)
        pos_pkv = tuple((layer[1][0], layer[1][1]) for layer in split_pkv)
        return neutral_pkv, pos_pkv

    past_key_values_neutral, past_key_values_pos = split_past_key_values(batched_past_key_values, bsize)

    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)

        # Denoise with neutral conditioning
        def denoise_neutral(input_x_t, current_timestep=expanded_time):
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values_neutral,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        # Denoise with positive conditioning
        def denoise_positive(input_x_t, current_timestep=expanded_time):
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks_pos,
                past_key_values=past_key_values_pos,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        v_t_neutral = denoise_neutral(x_t)
        v_t_pos = denoise_positive(x_t)

        # Classifier Free Guidance combination
        v_cfg = v_t_neutral + cfg_weight * (v_t_pos - v_t_neutral)

        # Euler step
        x_t += dt * v_cfg
        time += dt

    return x_t