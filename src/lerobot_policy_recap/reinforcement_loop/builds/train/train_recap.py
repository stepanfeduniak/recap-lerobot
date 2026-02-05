#!/usr/bin/env python
import logging
import time
from pathlib import Path
import numpy as np
import torch

from lerobot.configs import parser
from lerobot_policy_recap.configs.rl_train import RLTrainPipelineConfig
from lerobot_policy_recap.reinforcement_loop.learning_modules.rl_objects import RLObjects
from lerobot_policy_recap.reinforcement_loop.learning_modules.trainers.offline_trainer import OfflineTrainer
from lerobot_policy_recap.reinforcement_loop.learning_modules.evaluators.evaluator import BatchedEvaluator

@parser.wrap()
def main(cfg: RLTrainPipelineConfig):
    cfg.validate()
    
    # 1. Initialize the RL State Container
    # This loads the policy, dataset, preprocessors, and logger.
    rl = RLObjects(cfg)

    # 2. Assemble the Atomic Blocks
    trainer = OfflineTrainer(rl)
    evaluator = BatchedEvaluator(rl)
    data_collector = InteractionDatasetRecorder(rl)

    # 3. Training Loop
    logging.info(f"Starting Training: {cfg.steps} steps total.")
    
    # Track time for throughput metrics
    start_time = time.perf_counter()

    
    
    
    for step in range(rl.step+1, cfg.steps + 1):
        # --- A. Optimization Step ---
        time_step_start = time.perf_counter()
        losses = trainer.training_step(step)
        time_step_end = time.perf_counter()
        print(f"Step {step} training time: {time_step_end - time_step_start:.4f} seconds")
        # --- B. Evaluation & Checkpointing ---
        # Evaluate based on the frequency defined in cfg.eval_freq
        if step % cfg.eval_freq == 0:
            logging.info(f"Step {step}: Running Benchmark...")
            
            # Run the critic-aware evaluator (rollouts + video overlays)
            eval_results = evaluator.benchmark()
            
            # Log specific eval metrics to WandB/Tensorboard
            eval_metrics = {
                "avg_reward": np.mean(eval_results["sum_rewards"]),
                "success_rate": np.mean(eval_results["successes"]),
            }
            rl.logger.log_metrics(eval_metrics, step=step, prefix="eval")

        # --- C. Checkpointing ---
        if step % cfg.save_freq == 0 or step == cfg.steps:
            # RLObjects knows how to save the policy and its state
            rl.save_policy(step)

    # 4. Final Cleanup
    total_time = time.perf_counter() - start_time
    logging.info(f"Training finished in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()