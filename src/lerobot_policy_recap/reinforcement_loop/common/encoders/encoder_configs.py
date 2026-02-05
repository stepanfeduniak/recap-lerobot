from dataclasses import dataclass, field
from typing import Any


@dataclass
class EncoderOutputDims:
    """Output dimensions of the encoder for critic construction.
    
    Each field is None if that feature type is not used by the encoder.
    """
    image_dim: int | None = None  # projection_dim if images used
    state_dim: int | None = None  # state feature dim if state used  
    env_dim: int | None = None    # env feature dim if env used


@dataclass
class BaseEncoderConfig:
    use_images: bool = True
    use_states: bool = True
    use_env_states: bool = False



@dataclass
class Gemma3EncoderConfig(BaseEncoderConfig):
    """Configuration for the Gemma-3 based encoder.
    
    Combines standalone SigLIP vision encoder with Gemma-3-270m text model.
    
    Freeze control:
        - freeze_siglip: Freeze SigLIP vision encoder (default: True)
        - freeze_gemma: Freeze Gemma-3 language model (default: True)
        - freeze_projector: Freeze vision projector MLP (default: False - trainable)
    """
    # Granular freeze control
    freeze_siglip: bool = True
    freeze_gemma: bool = False  
    freeze_projector: bool = False  # Projector should be trainable by default
    use_gradient_checkpointing: bool = False
    
    pool: str = "mean"  # "mean" or "last"
    siglip_model_name: str = "google/siglip-so400m-patch14-384"
    gemma3_model_name: str = "google/gemma-3-270m"
    
    # Image resolution - 224x224 like PI05 produces 256 tokens/image instead of 729
    image_resolution: tuple[int, int] = (224, 224)
    
    # Token reduction config
    # Options: "none" (simple MLP), "resampler" (cross-attention ~2M), "conv" (strided conv ~1.5M)
    token_reduction_method: str = "none"
    num_image_tokens: int = 256  # Number of output tokens after reduction (from 256)
    
    # Soft prompt tokens (optional learnable prefix tokens)
    num_soft_prompt_tokens: int = 0
    
    # Optimization
    use_flash_attention: bool = False


@dataclass
class SigLIPEncoderConfig(BaseEncoderConfig):
    """Configuration for standalone SigLIP vision encoder.
    
    Uses only the SigLIP vision tower without Gemma language model.
    Much lighter weight than full Gemma3Encoder.
    
    SigLIP has no CLS token - uses mean pooling over all patch tokens,
    then projects to output dim via simple MLP.
    
    Freeze control:
        - freeze_siglip: Freeze SigLIP vision encoder (default: True)
    """
    # Granular freeze control
    freeze_siglip: bool = False    
    # Output projection dimension
    projection_dim: int = 896  # Gemma-3-270m hidden size for compatibility
    
    # Image resolution - 224x224 like PI05 produces 256 tokens/image instead of 729
    image_resolution: tuple[int, int] = (224, 224)
