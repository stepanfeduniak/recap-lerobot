"""
Modular Logger Module for Reinforcement Learning.

This module provides a flexible logging system with support for multiple backends
(WandB, console, etc.) and a clean interface for RL training pipelines.
"""

from lerobot_policy_recap.reinforcement_loop.common.logger.base import BaseLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.console_backend import ConsoleLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.wandb_backend import WandbLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.composite import CompositeLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.factory import make_logger

__all__ = [
    "BaseLogger",
    "ConsoleLogger", 
    "WandbLogger",
    "CompositeLogger",
    "make_logger",
]
