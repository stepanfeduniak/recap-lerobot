"""
Logger Factory.

Provides a simple interface for creating loggers based on configuration.
"""

import logging
from typing import Any

from lerobot_policy_recap.reinforcement_loop.common.logger.base import BaseLogger, NoOpLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.console_backend import ConsoleLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.wandb_backend import WandbLogger
from lerobot_policy_recap.reinforcement_loop.common.logger.composite import CompositeLogger


def make_logger(cfg: Any) -> BaseLogger:
    """
    Create a logger based on the configuration.
    
    Args:
        cfg: Training configuration object. Expected to have:
            - cfg.wandb.enable: bool - Whether to enable WandB logging
            - cfg.wandb.project: str - WandB project name
            - cfg.log_every: int - Console logging frequency (optional)
    
    Returns:
        A logger instance implementing the BaseLogger protocol.
        Returns NoOpLogger if all logging is disabled.
    """
    loggers: list[BaseLogger] = []
    
    # Check WandB configuration
    wandb_enabled = (
        hasattr(cfg, "wandb") 
        and getattr(cfg.wandb, "enable", False) 
        and getattr(cfg.wandb, "project", None)
    )
    
    # Check console configuration
    console_enabled = getattr(cfg, "console_logging", True)  # Default to True
    log_freq = getattr(cfg, "log_every", 100)
    
    # Create WandB logger if enabled
    if wandb_enabled:
        try:
            wandb_logger = WandbLogger(cfg)
            loggers.append(wandb_logger)
            logging.info("WandB logger initialized.")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB logger: {e}")
    
    # Create console logger if enabled
    if console_enabled:
        console_logger = ConsoleLogger(log_freq=log_freq, verbose=True)
        loggers.append(console_logger)
    
    # Return appropriate logger
    if len(loggers) == 0:
        logging.info("All logging disabled, using NoOpLogger.")
        return NoOpLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
