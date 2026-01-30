"""
Base Logger Protocol.

Defines the interface that all logger backends must implement.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class BaseLogger(Protocol):
    """
    Protocol defining the interface for all loggers.
    
    Using Protocol allows structural typing - any class that implements
    these methods is considered a valid logger, no inheritance required.
    """
    
    def log_metrics(
        self, 
        metrics: dict[str, float | int | str], 
        step: int, 
        prefix: str = ""
    ) -> None:
        """
        Log a dictionary of metrics at a given step.
        
        Args:
            metrics: Dictionary of metric name -> value pairs.
            step: The current training/evaluation step.
            prefix: Optional prefix for metric namespacing (e.g., "train", "eval").
        """
        ...
    
    def log_video(
        self, 
        video_path: str | Path, 
        step: int, 
        prefix: str = "eval"
    ) -> None:
        """
        Log a video file.
        
        Args:
            video_path: Path to the video file.
            step: The current step.
            prefix: Namespace prefix for the video.
        """
        ...
    
    def log_artifact(
        self, 
        path: Path, 
        artifact_type: str = "model", 
        name: str | None = None
    ) -> None:
        """
        Log an artifact (checkpoint, model, etc).
        
        Args:
            path: Path to the artifact file or directory.
            artifact_type: Type of artifact (e.g., "model", "dataset").
            name: Optional name for the artifact.
        """
        ...
    
    def finish(self) -> None:
        """
        Cleanup and finalize logging.
        
        Call this at the end of training to ensure all logs are synced.
        """
        ...


class NoOpLogger:
    """
    A logger that does nothing.
    
    Used when logging is disabled to avoid None checks throughout the code.
    """
    
    def log_metrics(
        self, 
        metrics: dict[str, float | int | str], 
        step: int, 
        prefix: str = ""
    ) -> None:
        pass
    
    def log_video(
        self, 
        video_path: str | Path, 
        step: int, 
        prefix: str = "eval"
    ) -> None:
        pass
    
    def log_artifact(
        self, 
        path: Path, 
        artifact_type: str = "model", 
        name: str | None = None
    ) -> None:
        pass
    
    def finish(self) -> None:
        pass
