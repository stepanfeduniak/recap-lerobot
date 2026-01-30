from dataclasses import asdict, dataclass, field
import abc
import draccus
import torch

@dataclass
class RewardConfig(draccus.ChoiceRegistry, abc.ABC):
    use_numpy: bool = False  # Updated default
    discount_factor: float = None
    max_steps_per_episode: int = 1000

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


class BaseRewardModel(abc.ABC):
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.discount_factor = cfg.discount_factor
    
    @abc.abstractmethod
    def get_rewards(self, reward_tensor: torch.Tensor, success=None, failed=None) -> torch.Tensor:
        """Should return a torch.Tensor of shape (B, T)"""
        raise NotImplementedError

    def get_chunked_rewards(self, reward_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            reward_tensor: Tensor of shape (B, T, 1)
        Returns:
            total_chunk_reward: Tensor of shape (B,)
        """
        # rewards shape: (B, T)
        rewards = self.get_rewards(reward_tensor, **kwargs)
        device = rewards.device
        B, T = rewards.shape
        
        # Create discount vector (T,) on the correct device
        discounts = torch.pow(
            self.discount_factor, 
            torch.arange(T, device=device, dtype=torch.float32)
        )
        
        # Element-wise multiply (B, T) * (T,) and sum across T (dim=1)
        total_chunk_reward = torch.sum(rewards * discounts, dim=1)
        
        return total_chunk_reward