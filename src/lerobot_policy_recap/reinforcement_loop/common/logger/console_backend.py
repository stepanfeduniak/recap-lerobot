"""
Console Logger Backend.

Provides rich terminal output for metrics, useful for debugging and
local development without external logging services.
"""

import logging
from pathlib import Path
from typing import Any

from termcolor import colored


class ConsoleLogger:
    """
    Logger that outputs metrics to the console with pretty formatting.
    
    Uses termcolor for visual distinction between different metric prefixes.
    """
    
    # Color mapping for different prefixes
    PREFIX_COLORS = {
        "train": "blue",
        "eval": "green",
        "collector": "yellow",
        "critic": "magenta",
    }
    DEFAULT_COLOR = "cyan"
    
    def __init__(self, log_freq: int = 1, verbose: bool = True):
        """
        Args:
            log_freq: Only log every N calls (useful for high-frequency updates).
            verbose: If False, only log at INFO level instead of printing.
        """
        self.log_freq = log_freq
        self.verbose = verbose
        self._call_counts = {}  # Track call counts per prefix
    
    def _get_color(self, prefix: str) -> str:
        """Get the color for a given prefix."""
        return self.PREFIX_COLORS.get(prefix.lower(), self.DEFAULT_COLOR)
    
    def _format_value(self, value: float | int | str) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    
    def log_metrics(
        self, 
        metrics: dict[str, float | int | str], 
        step: int, 
        prefix: str = ""
    ) -> None:
        """Log metrics to the console."""
        # Track call counts per prefix so different log sources don't interfere
        if prefix not in self._call_counts:
            self._call_counts[prefix] = 0
        self._call_counts[prefix] += 1
        
        if self._call_counts[prefix] % self.log_freq != 0:
            return
        
        color = self._get_color(prefix)
        prefix_str = f"[{prefix}]" if prefix else ""
        
        # Build the log line
        metrics_str = " | ".join(
            f"{k}: {self._format_value(v)}" 
            for k, v in metrics.items()
        )
        
        log_line = f"Step {step} {prefix_str} {metrics_str}"
        
        if self.verbose:
            print(colored(log_line, color))
        else:
            logging.info(log_line)
    
    def log_video(
        self, 
        video_path: str | Path, 
        step: int, 
        prefix: str = "eval"
    ) -> None:
        """Log video path to console."""
        color = self._get_color(prefix)
        log_line = f"Step {step} [{prefix}] Video saved: {video_path}"
        
        if self.verbose:
            print(colored(log_line, color))
        else:
            logging.info(log_line)
    
    def log_artifact(
        self, 
        path: Path, 
        artifact_type: str = "model", 
        name: str | None = None
    ) -> None:
        """Log artifact save to console."""
        name_str = f" ({name})" if name else ""
        log_line = f"Artifact [{artifact_type}]{name_str} saved: {path}"
        
        if self.verbose:
            print(colored(log_line, "white", attrs=["bold"]))
        else:
            logging.info(log_line)
    
    def finish(self) -> None:
        """No cleanup needed for console logger."""
        pass