from dataclasses import dataclass, field


@dataclass
class AdaptiveFusionConfig:
    """Configuration for feature fusion/projection layers in critics.
    
    These layers project raw encoder features to a common hidden dimension
    before the critic MLP processes them.
    """
    use_image_features: bool = True
    use_state_features: bool = False
    use_env_features: bool = False

    image_hidden_dim: int = 512
    state_hidden_dim: int = 128
    env_hidden_dim: int = 128
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: str = "Tanh"


@dataclass
class BaseCriticNetworkConfig:
    pass


@dataclass
class MLPCriticNetworkConfig(BaseCriticNetworkConfig):
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512])
    activations: str = "GELU"
    activate_final: bool = True
    final_activation: str | None = None
    dropout_rate: float = 0.1
    # Fusion layer config for processing raw encoder features
    fusion_config: AdaptiveFusionConfig = field(default_factory=AdaptiveFusionConfig)


@dataclass
class DistributionalMLPCriticNetworkConfig(MLPCriticNetworkConfig):
    num_atoms: int = 101