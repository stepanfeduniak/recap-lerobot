import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Dict
from pathlib import Path

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot_policy_recap.reinforcement_loop.common.logger import make_logger
from lerobot_policy_recap.policies.recap.processor_recap_pi import make_recap_pre_post_processors
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
    load_training_state,
)
from accelerate import Accelerator

class RLObjects:
    """
    The Base LEGO Box.
    Holds the state of all RL components (Policy, Env, Dataset, Logger).
    
    Buffers are lazily initialized - they're only created when accessed.
    This keeps the base object lightweight while providing clean access patterns.
    """
    def __init__(self, config: Any):
        self.cfg = config
        # self.cfg.policy.validate_features() # Temporary fix
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 0. Accelerator (Lazy Initialization)
        self._accelerator = None
        self._device = None
        # 1. Dataset & Dataloader
        self.dataset = None
        self.dataloader = None
        self._init_dataset()
        # 2. Environment (Vectorized)
        self.envs = None
        self.env_preprocessor = None
        self.env_postprocessor = None
        self._init_envs()
        # 3. Policy (The Brain)
        logging.info(f"Initializing Policy on {self.device}...")
        self.policy = None
        self.policy_preprocessor = None
        self.policy_postprocessor = None
        self._init_policy()
        # 4. Logger
        self.logger = make_logger(self.cfg)
        self.step=0
        # 6. Buffers (Lazy Initialization)
        self._offline_buffer = None
    def _init_dataset(self):
        # 1. Dataset & Dataloader
        if hasattr(self.cfg, "dataset") and self.cfg.dataset is not None:
            logging.info("Initializing Dataset...")
            self.dataset = make_dataset(self.cfg)
            self.dataloader = self._setup_dataloader()
    def _init_envs(self):
        self.envs = None
        if hasattr(self.cfg, "env") and self.cfg.env is not None:
            logging.info(f"Initializing {self.cfg.eval.batch_size} Environments...")
            self.envs = make_env(
                self.cfg.env, 
                n_envs=self.cfg.eval.batch_size, 
                use_async_envs=getattr(self.cfg.eval, "use_async_envs", False)
            )
            self._init_env_processors()
    def _init_policy(self):
        logging.info(f"Initializing Policy on {self.device}...")
        self.policy = make_policy(
            cfg=self.cfg.policy,
            ds_meta=self.dataset.meta if self.dataset else None,
            env_cfg=self.cfg.env if self.envs and not self.dataset else None,
            rename_map=self.cfg.rename_map,
        )
        self._init_policy_processors()
        self.policy.to(self.device)
    def _init_policy_processors(self):
        """Initialize policy pre/post processors."""
        processor_kwargs = {}
        if self.dataset and (not self.cfg.resume or not self.cfg.policy.pretrained_path):
            processor_kwargs["dataset_stats"] = self.dataset.meta.stats

        # Check if this is a RECAP policy - use custom processor
        if getattr(self.cfg.policy, "type", None) == "recap_pi":
            self.preprocessor, self.postprocessor = make_recap_pre_post_processors(
                config=self.cfg.policy,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        else:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.cfg.policy,
                pretrained_path=self.cfg.policy.pretrained_path,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                **processor_kwargs
            )
    def _init_env_processors(self):
        # Env Processors
        if self.envs:
            self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
                env_cfg=self.cfg.env,
                policy_cfg=self.cfg.policy
            )
    
    @property
    def offline_buffer(self):
        """Lazy initialization of OfflineReplayBuffer. Only created when accessed."""
        if self._offline_buffer is None:
            if self.dataset is None:
                raise ValueError(
                    "Cannot create offline_buffer without a dataset. "
                    "Make sure cfg.dataset is configured."
                )
            
            # Get horizon from config with sensible defaults
            horizon = getattr(self.cfg, "horizon", 10)
            if hasattr(self.cfg, "policy") and (hasattr(self.cfg.policy, "chunk_size") or hasattr(self.cfg.policy, "horizon")):
                horizon = getattr(self.cfg.policy, "chunk_size", getattr(self.cfg.policy, "horizon", horizon))
            
            from lerobot_policy_recap.reinforcement_loop.common.buffers.offline_buffer import OfflineReplayBuffer
            logging.info(f"Initializing OfflineReplayBuffer with horizon={horizon}...")
            self._offline_buffer = OfflineReplayBuffer(
                cfg=self.cfg,
                dataset=self.dataset,
                horizon=horizon
            )
        return self._offline_buffer
    
    @property
    def accelerator(self) -> Accelerator:
        """Lazy-initialized Accelerator."""
        if self._accelerator is None:
            logging.info("Initializing Accelerator (Lazy Loading)...")
            
            # RL policies often have parameters that aren't updated in every step 
            # (e.g., Actor vs Critic steps), so find_unused_parameters is crucial.
            from accelerate import DistributedDataParallelKwargs
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            
            self._accelerator = Accelerator(
                step_scheduler_with_optimizer=False,
                kwargs_handlers=[ddp_kwargs]
            )
            # Update device once accelerator is live
            self._device = self._accelerator.device
            logging.info(f"Accelerator ready on device: {self._device}")
            
        return self._accelerator
    
    @property
    def device(self):
        """Returns accelerator device, initializing accelerator if necessary."""
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    def _setup_dataloader(self) -> DataLoader:
        """Standard LeRobot Dataloader configuration."""
        sampler = None
        shuffle = True
        
        if hasattr(self.cfg.policy, "drop_n_last_frames"):
            sampler = EpisodeAwareSampler(
                self.dataset.meta.episodes["dataset_from_index"],
                self.dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=self.dataset.episodes,
                drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
            shuffle = False

        return DataLoader(
            self.dataset,
            num_workers=getattr(self.cfg, "num_workers", 4),
            batch_size=self.cfg.batch_size,
            shuffle=shuffle and not getattr(self.cfg.dataset, "streaming", False),
            sampler=sampler,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    def save_policy(self, step: int):
        """LEGO action: Save the current state of the policy."""
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, self.cfg.steps, step)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            cfg=self.cfg,
            policy=self.policy, 
            optimizer=None, # Managed by policy
            scheduler=None,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
        )
        update_last_checkpoint(checkpoint_dir)
        self.logger.log_artifact(checkpoint_dir, artifact_type="model")