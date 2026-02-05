import logging
import torch
from termcolor import colored
from typing import Dict

from lerobot_policy_recap.reinforcement_loop.learning_modules.trainers.base_trainer import BaseTrainer
from lerobot.datasets.utils import cycle
import time 

class OfflineTrainer(BaseTrainer):
    """
    Atomic block for Offline RL Training. 
    Provides a single-step training interface using the lazy offline_buffer from RLObjects.
    """
    def __init__(self, rl: any):
        super().__init__(rl)
        
        # Replay Buffer Setup
        # Access lazy-initialized buffer from RLObjects (creates on first access)
        self.offline_buffer = rl.offline_buffer
        
        # Cache observation keys for later use
        self.observation_keys = self.offline_buffer.observation_keys
        
        logging.info(colored("OfflineTrainer atomic block ready.", "green"))
        
        # Setup accelerator FIRST to get correct device
        self._setup_accelerator_and_dataloader()
    def _setup_accelerator_and_dataloader(self):
        """Setup accelerator first, then create dataloader with correct device for pin_memory."""
        self._setup_accelerator()
        print(f"DEVICE IN OFFLINE TRAINER: {self.device}")
        
        # Create dataloader AFTER we know the correct device
        self.dataloader = self._create_dataloader(
            self.offline_buffer,
            batch_size=self.rl.cfg.batch_size,
        )
        
        # Prepare policy, dataloader, and all optimizers
        self.dataloader = self._prepare_policy_and_optimizers(self.dataloader)
        self.batch_iterator = cycle(self.dataloader)
        
    def training_step(self, step: int) -> Dict[str, float]:
        """
        Executes a single update of the policy.
        Args:
            step: The current global iteration count.
        Returns:
            A dictionary of losses for logging.
        """
        self.rl.policy.train()

        # Sample Batch
        time_start = time.time()
        transition_batch = next(self.batch_iterator)   
        time_end = time.time()
        # Prepare Batch (reward processing + preprocessing)
        batch = self._prepare_batch(transition_batch)
        
        # Policy Gradient Update
        losses = self.rl.policy.training_step(batch, accelerator=self.accelerator)
        
        self.last_losses = losses

        # Logging
        self._log_training_metrics(losses, step)

        return losses
