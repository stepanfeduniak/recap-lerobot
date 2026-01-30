"""
WandB Logger Backend.

Provides integration with Weights & Biases for experiment tracking,
metric visualization, and artifact management.
"""

import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Any

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    """Get the WandB run ID from the filesystem for run resumption."""
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    return match.groups(0)[0]


def get_safe_artifact_name(name: str) -> str:
    """WandB artifacts don't accept ':' or '/' in their name."""
    return name.replace(":", "_").replace("/", "_")


def cfg_to_tags(cfg: Any) -> list[str]:
    """Extract tags from config for WandB run tagging."""
    tags = []
    
    # Policy type
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "type"):
        tags.append(f"policy:{cfg.policy.type}")
    
    # Seed
    if hasattr(cfg, "seed"):
        tags.append(f"seed:{cfg.seed}")
    
    # Dataset
    if hasattr(cfg, "dataset") and cfg.dataset is not None:
        ds = str(cfg.dataset.repo_id)
        ds_short = Path(ds).name if ds.startswith("/") else ds
        tags.append(f"dataset:{ds_short}")
    
    # Environment
    if hasattr(cfg, "env") and cfg.env is not None:
        tags.append(f"env:{cfg.env.type}")
    
    return tags


class WandbLogger:
    """
    Logger backend that integrates with Weights & Biases.
    
    Handles:
    - Run initialization and resumption
    - Metric logging with custom step keys for async RL
    - Video logging
    - Artifact checkpointing
    """
    
    def __init__(
        self,
        cfg: Any,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
    ):
        """
        Initialize WandB logger.
        
        Args:
            cfg: The training configuration object.
            project: WandB project name (overrides cfg.wandb.project).
            entity: WandB entity/team name (overrides cfg.wandb.entity).
            run_name: Name for this run (overrides cfg.job_name).
        """
        self.cfg = cfg
        wandb_cfg = cfg.wandb
        
        self.log_dir = Path(cfg.output_dir)
        self.project = project or wandb_cfg.project
        self.entity = entity or getattr(wandb_cfg, "entity", None)
        self.run_name = run_name or getattr(cfg, "job_name", None)
        self.env_fps = cfg.env.fps if hasattr(cfg, "env") and cfg.env else None
        
        # Set up WandB
        os.environ["WANDB_SILENT"] = "True"
        import wandb
        self._wandb = wandb
        
        # Handle run resumption
        wandb_run_id = None
        if hasattr(wandb_cfg, "run_id") and wandb_cfg.run_id:
            wandb_run_id = wandb_cfg.run_id
        elif hasattr(cfg, "resume") and cfg.resume:
            try:
                wandb_run_id = get_wandb_run_id_from_filesystem(self.log_dir)
            except RuntimeError:
                logging.warning("Could not find previous WandB run ID for resumption.")
        
        # Get config dict
        config_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        
        # Sanitize config_dict to convert any tensors to lists/primitives
        config_dict = self._sanitize_config(config_dict)
        
        # Determine mode
        mode = getattr(wandb_cfg, "mode", "online")
        if mode not in ["online", "offline", "disabled"]:
            mode = "online"
        
        # Initialize WandB run
        wandb.init(
            id=wandb_run_id,
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            notes=getattr(wandb_cfg, "notes", None),
            tags=cfg_to_tags(cfg),
            dir=self.log_dir,
            config=config_dict,
            save_code=False,
            job_type="train_eval",
            resume="must" if (hasattr(cfg, "resume") and cfg.resume) else None,
            mode=mode,
        )
        
        # Store run ID back to config for later resumption
        if hasattr(wandb_cfg, "run_id"):
            wandb_cfg.run_id = wandb.run.id
        
        # Track custom step keys for async RL logging
        self._custom_step_keys: set[str] = set()
        
        # Compute group name for artifacts
        self._group = "-".join(cfg_to_tags(cfg))
        
        logging.info(colored("Logs will be synced with WandB.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")

    def _sanitize_config(self, obj: Any) -> Any:
        """
        Recursively sanitize config dict to convert tensors/arrays to JSON-serializable types.
        
        Args:
            obj: The object to sanitize (dict, list, or primitive).
        
        Returns:
            A sanitized version suitable for WandB config.
        """
        import torch
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._sanitize_config(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_config(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def log_metrics(
        self, 
        metrics: dict[str, float | int | str], 
        step: int, 
        prefix: str = "",
        custom_step_key: str | None = None,
    ) -> None:
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metric name -> value.
            step: Current training step.
            prefix: Namespace prefix (e.g., "train", "eval").
            custom_step_key: Optional custom x-axis key for async logging.
        """
        # Handle custom step key for async RL
        if custom_step_key is not None:
            full_step_key = f"{prefix}/{custom_step_key}" if prefix else custom_step_key
            if full_step_key not in self._custom_step_keys:
                self._custom_step_keys.add(full_step_key)
                self._wandb.define_metric(full_step_key, hidden=True)
        
        for key, value in metrics.items():
            # Skip non-loggable types
            if not isinstance(value, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{key}" was ignored as type "{type(value)}" is not supported.'
                )
                continue
            
            # Skip the custom step key itself
            if custom_step_key is not None and key == custom_step_key:
                continue
            
            # Build prefixed key
            prefixed_key = f"{prefix}/{key}" if prefix else key
            
            if custom_step_key is not None:
                # Use custom step key as x-axis
                step_value = metrics[custom_step_key]
                full_step_key = f"{prefix}/{custom_step_key}" if prefix else custom_step_key
                self._wandb.log({prefixed_key: value, full_step_key: step_value})
            else:
                # Use standard step
                self._wandb.log({prefixed_key: value}, step=step)
    
    def log_video(
        self, 
        video_path: str | Path, 
        step: int, 
        prefix: str = "eval"
    ) -> None:
        """Log a video to WandB."""
        video = self._wandb.Video(str(video_path), fps=self.env_fps, format="mp4")
        key = f"{prefix}/video" if prefix else "video"
        self._wandb.log({key: video}, step=step)
    
    def log_artifact(
        self, 
        path: Path, 
        artifact_type: str = "model", 
        name: str | None = None
    ) -> None:
        
        """Log an artifact (checkpoint) to WandB."""
        
        # Check if artifacts are disabled
        if not hasattr(self.cfg.wandb, "enable_artifact") or self.cfg.wandb.enable_artifact is False:
            return
        
        # Build artifact name
        if name is None:
            step_id = path.name
            name = f"{self._group}-{step_id}"
        name = get_safe_artifact_name(name)
        
        artifact = self._wandb.Artifact(name, type=artifact_type)
        
        # Add file or directory
        model_path = path / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
        if model_path.exists():
            artifact.add_file(str(model_path))
        elif path.is_file():
            artifact.add_file(str(path))
        elif path.is_dir():
            artifact.add_dir(str(path))
        
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the WandB run."""
        self._wandb.finish()
