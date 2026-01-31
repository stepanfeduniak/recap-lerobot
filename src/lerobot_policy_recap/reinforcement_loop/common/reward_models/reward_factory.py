
from lerobot_policy_recap.reinforcement_loop.common.reward_models.base_reward import RewardConfig, BaseRewardModel
from lerobot_policy_recap.configs.rl_train import RLTrainPipelineConfig
import numpy as np
import torch
from lerobot_policy_recap.reinforcement_loop.common.reward_models.common_rewards import (
    OriginalRewardModel,
    SparseRewardModel,
)

def get_reward_model_class(name: str) -> type[BaseRewardModel]:
    if name == "original":
        return OriginalRewardModel
    elif name == "sparse":
        return SparseRewardModel
    else:
        raise ValueError(f"Unknown reward style: {name}")


def make_reward_model(cfg: RLTrainPipelineConfig) -> BaseRewardModel:
   
    reward_cls = get_reward_model_class(cfg.reward.type)
    reward_cfg= cfg.reward
    reward_cfg.max_steps_per_episode = cfg.max_steps_per_episode
    reward_cfg.discount_factor = cfg.policy.discount
    return reward_cls(reward_cfg)
