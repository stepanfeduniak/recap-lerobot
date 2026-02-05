"""Configuration for RECAP PI policy using Gemma3 encoder with PI05."""

from dataclasses import dataclass, field
from abc import ABC

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot_policy_recap.reinforcement_loop.common.critics.critic_configs import DistributionalMLPCriticNetworkConfig
from lerobot_policy_recap.reinforcement_loop.common.encoders.encoder_configs import Gemma3EncoderConfig


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@PreTrainedConfig.register_subclass("recap_pi")
@dataclass
class RECAP_PI_Config(PreTrainedConfig):
    """Configuration class for the RECAP policy with PI05/Gemma3 encoder.
    
    Uses IDENTITY normalization for visual features (PI05 style)
    and QUANTILES for state/action (PI05 style).
    """
    # Normalization mapping for PI05
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Architecture specifics
    device: str = "cuda"
    storage_device: str = "cuda"
    
    # Training parameters
    online_steps: int = 1000000 # Unused
    online_buffer_capacity: int = 100000 # Unused
    offline_buffer_capacity: int = 100000 # Unused
    online_step_before_learning: int = 100 # Unused
    policy_update_freq: int = 1 # Unused

    # Discount factor
    discount: float = 0.99
    # Number of v_critics in the ensemble
    num_v_critics: int = 1
    # Learning rates
    v_critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    # Gradient clipping norm
    grad_clip_norm: float = 10.0

    # Network configuration
    v_critic_network_kwargs: DistributionalMLPCriticNetworkConfig = field(default_factory=DistributionalMLPCriticNetworkConfig)
    
    # PI05-specific: Gemma3 encoder config
    gemma3_encoder_config: Gemma3EncoderConfig = field(default_factory=Gemma3EncoderConfig)
    
    # Optimizations
    use_torch_compile: bool = False
    use_torch_compile_diffusion: bool = False

    # Diffusion parameters
    diffusion_training_steps: int = 100000
    diffusion_repo_id: str = "lerobot/pi05_libero_finetuned"
    training_mode: str = "critic"  # options: "critic", "actor"

    # Critic training parameters
    critic_training_steps: int = 100000 # Unused
    critic_warmup_steps: int = 1000 # Unused

    # Distributional V-Critic parameters
    v_min: float = -1.0
    v_max: float = 0.0
    num_atoms: int = 101

    # Action chunking parameters
    action_chunk_len: int = 10
    horizon: int = 50 # Unused
    only_online_chunks: bool = False # Unused

    # Processor specific parameters
    max_state_dim: int = 32
    tokenizer_max_length: int = 64

    # RECAP specific thresholds and flags
    indicator_threshold: float = 0.1
    min_online_buffer_size: int = 1000
    n_action_steps: int = 10
    no_advantage_dropout_chance: float = 0.3

    # Classifier-free guidance parameters
    cfg_weight: float = 1.0
    use_cfg: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        # Sync distributional parameters to network config
        if self.v_critic_network_kwargs is not None:
            self.v_critic_network_kwargs.num_atoms = self.num_atoms

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
            },
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig | None:
        """Return scheduler config for critics.
        
        Actor training uses the diffusion policy's own scheduler.
        """
        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=self.critic_warmup_steps,
        )

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features
        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        diffusion_policy_config = PreTrainedConfig.from_pretrained(self.diffusion_repo_id)
        print("DELTAS: ", diffusion_policy_config.action_delta_indices)
        return diffusion_policy_config.action_delta_indices

    @property
    def reward_delta_indices(self) -> None:
        return None
    
    @property
    def q_chunking(self) -> bool:
        return self.action_chunk_len > 1