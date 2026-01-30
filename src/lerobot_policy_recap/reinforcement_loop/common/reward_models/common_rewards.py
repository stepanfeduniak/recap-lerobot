
from lerobot_policy_recap.reinforcement_loop.common.reward_models.base_reward import BaseRewardModel, RewardConfig
from dataclasses import dataclass
import torch



@RewardConfig.register_subclass("original")
@dataclass
class OriginalRewardConfig(RewardConfig):
    pass


class OriginalRewardModel(BaseRewardModel):
    def get_rewards(self, reward_tensor, success=None, failed=None):
        # reward_tensor is (B, T, 1), return (B, T)
        return reward_tensor.squeeze(-1).to(torch.float32)


@RewardConfig.register_subclass("sparse")
@dataclass
class SparseRewardConfig(RewardConfig):
    pass


class SparseRewardModel(BaseRewardModel):
    def get_rewards(self, reward_tensor, success=None, failed=None):
        B, T, _ = reward_tensor.shape
        device = reward_tensor.device
        rewards = torch.zeros((B, T), device=device, dtype=torch.float32)
        
        if success is not None:
            # success should be a boolean tensor of shape (B,)
            if not isinstance(success, torch.Tensor):
                success = torch.tensor(success, device=device)
            rewards[success, -1] = 1.0
            
        return rewards


@RewardConfig.register_subclass("pi06+")
@dataclass
class PI06PlusRewardConfig(RewardConfig):
    pass


class PI06PlusRewardModel(BaseRewardModel):
    def get_rewards(self, reward_tensor, success=None, failed=None):
        B, T, _ = reward_tensor.shape
        device = reward_tensor.device
        # Fill with constant penalty
        rewards = torch.full((B, T), -1.0 / self.cfg.max_steps_per_episode, device=device)
        
        if success is not None:
            if not isinstance(success, torch.Tensor):
                success = torch.tensor(success, device=device)
            rewards[success, -1] += 1.0
            
        return rewards
