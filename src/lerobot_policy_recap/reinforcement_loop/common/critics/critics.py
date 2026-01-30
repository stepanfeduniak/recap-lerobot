"""Base critic network components."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot_policy_recap.reinforcement_loop.common.encoders.encoder_configs import EncoderOutputDims
from lerobot_policy_recap.reinforcement_loop.common.critics.critic_configs import AdaptiveFusionConfig


def orthogonal_init():
    """Return a function that applies orthogonal initialization."""
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class AdaptiveFusion(nn.Module):
    """Adaptive fusion layer that processes and concatenates feature types.
    
    This layer takes the raw feature dictionary from an encoder and applies
    trainable projection layers to each feature type before concatenating.
    """
    
    def __init__(
        self,
        encoder_output_dims: EncoderOutputDims,
        config: AdaptiveFusionConfig,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict()
        
        total_dim = 0
        if encoder_output_dims.image_dim is not None and config.use_image_features:

            self.layers["image"] = self._make_projection(
                encoder_output_dims.image_dim, config.image_hidden_dim
            )
            total_dim += config.image_hidden_dim
        
        if encoder_output_dims.state_dim is not None and config.use_state_features:
            self.layers["state"] = self._make_projection(
                encoder_output_dims.state_dim, config.state_hidden_dim
            )
            total_dim += config.state_hidden_dim
        
        if encoder_output_dims.env_dim is not None and config.use_env_features:
            self.layers["env"] = self._make_projection(
                encoder_output_dims.env_dim, config.env_hidden_dim
            )
            total_dim += config.env_hidden_dim
        
        self._output_dim = total_dim
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[AdaptiveFusion] Total params: {total_params:,} | Output dim: {total_dim}")
    
    def _make_projection(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """Create a projection layer for a feature type."""
        layers = [nn.Linear(in_dim, out_dim)]
        if self.config.use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(getattr(nn, self.config.activation)())
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate))
        return nn.Sequential(*layers)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of the fused features."""
        return self._output_dim
    
    def get_optim_params(self) -> list:
        """Return parameters for optimization (all trainable)."""
        return list(self.parameters())

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass: project each feature type and concatenate.
        
        Args:
            features: Dictionary with keys like "image_features", "state_features", "env_features"
            
        Returns:
            Concatenated fused features [B, output_dim]
        """
        parts = []
        if "image_features" in features and "image" in self.layers:
            parts.append(self.layers["image"](features["image_features"]))
        if "state_features" in features and "state" in self.layers:
            parts.append(self.layers["state"](features["state_features"]))
        if "env_features" in features and "env" in self.layers:
            parts.append(self.layers["env"](features["env_features"]))
        return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    """A single critic head that outputs a scalar Q-value or V-value."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def get_optim_params(self) -> list:
        """Return parameters for optimization (all trainable)."""
        return list(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class DistributionalCriticHead(nn.Module):
    """A single distributional critic head outputting a probability distribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
        num_atoms: int = 51,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        # Output layer produces logits for num_atoms [cite: 236]
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=num_atoms)
        
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)

    def get_optim_params(self) -> list:
        """Return parameters for optimization (all trainable)."""
        return list(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.output_layer(self.net(x))
        # Return probabilities using Softmax [cite: 238]
        return logits
