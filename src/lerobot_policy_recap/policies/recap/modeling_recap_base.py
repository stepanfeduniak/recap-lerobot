"""Base modeling class for RECAP policies."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot_policy_recap.reinforcement_loop.common.critics import (
    DistributionalCriticHead,
    DistributionalVCriticEnsemble, 
    AdaptiveFusion
)
from lerobot_policy_recap.policies.recap.configuration_recap_pi import RECAP_PI_Config
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_PREFIX
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

from lerobot.processor.core import TransitionKey
from accelerate import Accelerator

OBS_LANGUAGE_TOKENS_POS = "observation.language_tokens_pos"
OBS_LANGUAGE_ATTENTION_MASK_POS = "observation.language_attention_mask_pos"


class RECAPBasePolicy(PreTrainedPolicy, ABC):
    """Base class for RECAP policies.
    
    Contains all shared logic between RECAP variants.
    Subclasses must implement `_init_encoders()` to set up the encoder.
    """
    config_class = RECAP_PI_Config
    name = "recap_base_policy"

    def __init__(
        self,
        config: RECAP_PI_Config | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._load_diffusion_policy()
        print(f"Loaded diffusion policy: {self.diffusion_policy.config.type} from {self.diffusion_policy.config.pretrained_path} on device {get_device_from_parameters(self.diffusion_policy)}")
        
        # Apply torch.compile to diffusion policy if enabled
        if self.config.use_torch_compile_diffusion:
            self.diffusion_policy = torch.compile(self.diffusion_policy)
            print("Applied torch.compile to diffusion policy")
        self.cfg_weight = config.cfg_weight

        # Initialize all components
        self._init_encoders()
        self._init_v_critics()
        self._init_optimizers()
        self.training_mode = self.config.training_mode

        # Check if horizon attribute exists, otherwise use default from config
        self.action_chunk_len = getattr(self.config, 'action_chunk_len', None)
        if self.action_chunk_len is None:
             self.action_chunk_len = getattr(self.diffusion_policy.config, 'action_chunk_len', 1)
        self.action_queue = None
    
    @abstractmethod
    def _init_encoders(self):
        """Initialize encoder for v_critic.
        
        Must be implemented by subclasses to set up self.encoder_v_critic.
        """
        pass

    @torch.no_grad()
    def _value_info(
        self,
        observations: dict[str, Tensor],
    ) -> dict:
        """
        Compute V(s) from the distributional v-critic ensemble and return a compact info dict.
        Returns values on CPU for cheap logging.
        """
        # Get logits from distributional ensemble: [num_critics, batch_size, num_atoms]
        v_logits = self.v_critic_ensemble(observations)
        
        # Convert to scalar expectations: [num_critics, batch_size]
        v_ens = self.v_critic_ensemble.get_expectation(v_logits)

        v_min, _ = torch.min(v_ens, dim=0)   # (B,)
        v_mean = v_ens.mean(dim=0)           # (B,)

        return {
            "v_min": v_min.detach().cpu(),
            "v_mean": v_mean.detach().cpu(),
        }

    def _init_v_critics(self):
        """Build v_critic ensemble with fusion layer.
        
        Creates AdaptiveFusion layer which contains the trainable adaptive layers.
        The encoder outputs raw features, fusion processes them.
        """
        # Create fusion layer for V-critic (trainable)
        self.v_fusion = AdaptiveFusion(
            encoder_output_dims=self.encoder_v_critic.output_dims,
            config=self.config.v_critic_network_kwargs.fusion_config,
        )
        
        # Create critic heads - input is fusion output
        heads = [
            DistributionalCriticHead(
                input_dim=self.v_fusion.output_dim,
                **{k: v for k, v in asdict(self.config.v_critic_network_kwargs).items() if k != 'fusion_config'},
            )
            for _ in range(self.config.num_v_critics)
        ]
        self.v_critic_ensemble = DistributionalVCriticEnsemble(
            encoder=self.encoder_v_critic, 
            fusion=self.v_fusion,
            ensemble=heads,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
        )
        if self.config.use_torch_compile:
            self.v_critic_ensemble = torch.compile(self.v_critic_ensemble)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], return_bn_actions=False) -> tuple[Tensor]:
        self.diffusion_policy.eval()

        # 1. Handle Observation Extraction
        if TransitionKey.OBSERVATION in batch:
            obs = batch[TransitionKey.OBSERVATION]
        else:
            # Filter batch for observation keys (starting with 'observation.image' or 'observation.state')
            obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
            
        # Standard single-sample path
        if hasattr(self.diffusion_policy, "predict_action_chunk"):
            actions = self.diffusion_policy.predict_action_chunk(batch)
        else:
            actions = self.diffusion_policy.select_action(batch)
        
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return actions[:, :self.action_chunk_len, :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor]:   
        if self.action_queue is None:
            self.action_queue = deque(maxlen=self.action_chunk_len)
 
        if len(self.action_queue) == 0:
            full_chunk = self.predict_action_chunk(batch)
            self.action_queue.extend(full_chunk[:, :self.action_chunk_len].transpose(0, 1))

        action = self.action_queue.popleft()

        return action

    def _load_diffusion_policy(self):
        pretrained_path = self.config.diffusion_repo_id
        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        policy_cls = get_policy_class(policy_config.type)
        self.diffusion_policy = policy_cls.from_pretrained(pretrained_path)
        self.diffusion_policy.requires_grad_(False)
        self.diffusion_policy.eval()

    def reset(self):
        self.action_queue = None
        self.diffusion_policy.reset()

    def _init_optimizers(self):
        """Initialize separate optimizers for critic and actor."""
        # V-Critic optimizer
        v_critic_params = self.v_critic_ensemble.get_optim_params()
        self.v_critic_optimizer = torch.optim.Adam(
            v_critic_params,
            lr=self.config.v_critic_lr
        )
        print(f"Initialized value critic optimizer with {sum(p.numel() for p in v_critic_params)} parameters:")

        # V-Critic scheduler
        v_critic_scheduler_cfg = self.config.get_scheduler_preset()
        if v_critic_scheduler_cfg is not None:
            self.v_critic_scheduler = v_critic_scheduler_cfg.build(
                self.v_critic_optimizer,
                self.config.critic_training_steps
            )
            print(f"Initialized value critic scheduler: {type(self.v_critic_scheduler).__name__}")
        else:
            self.v_critic_scheduler = None

        # Diffusion optimizer/scheduler (from diffusion policy presets)
        params = self.diffusion_policy.get_optim_params()
        optimizer_cfg = self.diffusion_policy.config.get_optimizer_preset()
        scheduler_cfg = self.diffusion_policy.config.get_scheduler_preset()
        self.diffusion_optimizer = optimizer_cfg.build(params)
        self.diffusion_scheduler = None
        if scheduler_cfg is not None:
            self.diffusion_scheduler = scheduler_cfg.build(self.diffusion_optimizer, self.config.diffusion_training_steps)
        print(f"Initialized diffusion policy optimizer with {sum(p.numel() for p in params)} parameters.")
        
    def get_optim_params(self) -> dict:
        return {
            "v_critic": self.v_critic_ensemble.get_optim_params(),
            "diffusion": self.diffusion_policy.get_optim_params(),
        }

    def get_optimizers(self) -> dict:
        """Return dictionary of optimizers for accelerator preparation."""
        optimizers = {
            "v_critic_optimizer": self.v_critic_optimizer,
            "diffusion_optimizer": self.diffusion_optimizer,
        }
        return optimizers

    def get_schedulers(self) -> dict:
        """Return dictionary of schedulers."""
        return {
            "v_critic_scheduler": self.v_critic_scheduler,
            "diffusion_scheduler": self.diffusion_scheduler,
        }

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "v_critic"] = "v_critic",
    ) -> dict[str, Tensor]:
        
        if model == "v_critic":
            loss_v_critic = self.compute_loss_v_critic(batch=batch)
            return {"loss_v_critic": loss_v_critic}
        if model == "actor":
            return {"loss_actor": self.compute_loss_actor(batch=batch)}

        raise ValueError(f"Unknown model type: {model}")
    
    def compute_loss_actor(
            self,
            batch: dict[str, Tensor],
        ) -> Tensor:
        """Compute actor loss using advantage-weighted regression.
        
        Args:
            batch: Dictionary containing observations and actions.
        """
        observations = self.chose_instruction(batch)
        actions = batch[ACTION]

        policy_batch = {**observations, ACTION: actions}
        loss_actor, loss_dict = self.diffusion_policy(policy_batch)

        return loss_actor

    def chose_instruction(self, batch: dict) -> dict:
        """
        Choose between standard, positive, and negative instructions based on improvement indicator.
        
        Args:
           batch: Dictionary containing observations and actions.
        """
        if "improvement_indicator" in batch:
            improvement_indicator = batch["improvement_indicator"]
        else:
            improvement_indicator = self.compute_improvement_indicator(batch)
 
        observations = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}

        # Iterate over batch to select instructions
        batch_size = improvement_indicator.shape[0]
        
        # Keys for token and mask
        key_tokens = f"{OBS_PREFIX}language.tokens"
        key_mask = f"{OBS_PREFIX}language.attention_mask"
        
        # 1. Determine which indices keep standard instruction (dropout)
        if self.config.no_advantage_dropout_chance > 0:
            dropout_mask = torch.rand(batch_size, device=improvement_indicator.device) < self.config.no_advantage_dropout_chance
        else:
            dropout_mask = torch.zeros(batch_size, dtype=torch.bool, device=improvement_indicator.device)
            
        # 2. Prepare source tensors
        # Standard
        tokens_std = batch[key_tokens]
        # Check if std mask exists (it might not if not padded? Actually PI processor ensures it)
        mask_std = batch[key_mask]
        
        # Positive
        tokens_pos = batch[f"{key_tokens}_pos"]
        mask_pos = batch[f"{key_mask}_pos"]
        
        # Negative
        tokens_neg = batch[f"{key_tokens}_neg"]
        mask_neg = batch[f"{key_mask}_neg"]
        
        # Initialize output tensors
        out_tokens = tokens_std.clone()
        out_mask = mask_std.clone()
        
        # Selectors
        use_pos = (~dropout_mask) & improvement_indicator
        use_neg = (~dropout_mask) & (~improvement_indicator)
        
        # Apply choices
        # Apply Positive
        out_tokens[use_pos] = tokens_pos[use_pos]
        out_mask[use_pos] = mask_pos[use_pos]
        
        # Apply Negative
        out_tokens[use_neg] = tokens_neg[use_neg]
        out_mask[use_neg] = mask_neg[use_neg]
        
        # Update observations
        observations[key_tokens] = out_tokens
        observations[key_mask] = out_mask
        
        return observations

    def compute_loss_v_critic(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Paper-style distributional V: Monte-Carlo return -> discretized bin -> CE loss.
        No Bellman backup, no projection.
        """
        # Expect: logits shape [num_critics, batch, B]
        v_logits = self.v_critic_ensemble(batch)

        target_return = batch["return_to_go"]  # [batch]

        # map return -> bin index
        tz = target_return.clamp(self.config.v_min, self.config.v_max)
        # delta_z = (v_max - v_min) / (B - 1)
        delta_z = self.v_critic_ensemble.delta_z
        b = torch.round((tz - self.config.v_min) / delta_z).long()
        b = b.clamp(0, self.config.num_atoms - 1)  # B = num_atoms

        # CE expects [batch, B]
        losses = []
        for k in range(v_logits.shape[0]):
            losses.append(F.cross_entropy(v_logits[k], b))
        return torch.stack(losses).mean()

    def compute_advantages(self, batch: dict[str, Tensor]) -> Tensor:
        # Extract observations
        obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
        next_obs = {k.replace("next.", ""): v for k, v in batch.items() if k.startswith("next." + OBS_PREFIX)}
        rewards = batch["next.reward"]

        # Distributional ensemble returns [num_critics, batch, num_atoms]
        logits_t = self.v_critic_ensemble(obs)
        logits_tn = self.v_critic_ensemble(next_obs)
        
        # Get scalar expectations [num_critics, batch]
        # and average over ensemble for a stable estimate
        v_t = self.v_critic_ensemble.get_expectation(logits_t).mean(0)
        v_tn = self.v_critic_ensemble.get_expectation(logits_tn).mean(0)
        
        advantages = self.config.discount * v_tn - v_t + rewards
        return advantages

    def compute_improvement_indicator(self, batch):
        advantages = self.compute_advantages(batch)
        return advantages > self.config.indicator_threshold
    
    def training_step(self, batch: dict, accelerator: Accelerator) -> dict:
        # 1. Update critic
        if self.training_mode == "critic":        
            # Encoder features - freezing is handled by requires_grad on params
            # (SigLIP/Gemma frozen, projector trainable by default)
            obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
            v_features = self.encoder_v_critic(obs)
            for k, v in v_features.items():
                batch[f"observation.cache.{k}"] = v
            # V-Critic Update
            batch[ACTION] = batch[ACTION].reshape(batch[ACTION].shape[0], -1)

            with accelerator.autocast():
                v_critic_loss_dict = self.forward(batch, model="v_critic")
                v_critic_loss = v_critic_loss_dict["loss_v_critic"]
            
            self.v_critic_optimizer.zero_grad()
            accelerator.backward(v_critic_loss)
            v_grad_norm = accelerator.clip_grad_norm_(self.v_critic_ensemble.parameters(), self.config.grad_clip_norm)
            self.v_critic_optimizer.step()
            if self.v_critic_scheduler is not None:
                self.v_critic_scheduler.step()
            return {"v_loss": v_critic_loss.item()}

        else:
            # Actor Update
            self.diffusion_policy.train()
            with accelerator.autocast():
                actor_loss_dict = self.forward(batch, model="actor")
                loss = actor_loss_dict["loss_actor"]

            self.diffusion_optimizer.zero_grad()
            accelerator.backward(loss)
            self.diffusion_optimizer.step()
            if self.diffusion_scheduler:
                self.diffusion_scheduler.step()
            return {"actor_loss": loss.item()}