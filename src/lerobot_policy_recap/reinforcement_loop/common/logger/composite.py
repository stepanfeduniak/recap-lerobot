"""
Composite Logger.

Aggregates multiple logger backends to allow simultaneous logging
to different destinations (e.g., WandB + Console).
"""

from pathlib import Path
from typing import Any

from lerobot_policy_recap.reinforcement_loop.common.logger.base import BaseLogger


class CompositeLogger:
    """
    Logger that delegates to multiple backend loggers.
    
    Useful for logging to both WandB and console simultaneously,
    or for adding custom logging backends without modifying existing code.
    """
    
    def __init__(self, loggers: list[BaseLogger]):
        """
        Args:
            loggers: List of logger backends to delegate to.
        """
        self.loggers = loggers
    
    def log_metrics(
        self, 
        metrics: dict[str, float | int | str], 
        step: int, 
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Log metrics to all backends."""
        for logger in self.loggers:
            # Pass kwargs for backends that support extra args (like custom_step_key)
            if hasattr(logger, "log_metrics"):
                try:
                    logger.log_metrics(metrics, step, prefix, **kwargs)
                except TypeError:
                    # Backend doesn't support kwargs, call without them
                    logger.log_metrics(metrics, step, prefix)
    
    def log_video(
        self, 
        video_path: str | Path, 
        step: int, 
        prefix: str = "eval"
    ) -> None:
        """Log video to all backends."""
        for logger in self.loggers:
            if hasattr(logger, "log_video"):
                logger.log_video(video_path, step, prefix)
    
    def log_artifact(
        self, 
        path: Path, 
        artifact_type: str = "model", 
        name: str | None = None
    ) -> None:
        """Log artifact to all backends."""
        for logger in self.loggers:
            if hasattr(logger, "log_artifact"):
                logger.log_artifact(path, artifact_type, name)

    def finish(self) -> None:
        """Finish all backends."""
        for logger in self.loggers:
            if hasattr(logger, "finish"):
                logger.finish()
