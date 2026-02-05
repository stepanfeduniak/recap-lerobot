#!/usr/bin/env python
import logging
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

from lerobot.configs import parser
from lerobot_policy_recap.configs.rl_train import RLTrainPipelineConfig

from lerobot_policy_recap.reinforcement_loop.learning_modules.rl_objects import RLObjects
from lerobot_policy_recap.reinforcement_loop.learning_modules.data_collectors.interaction_collector import InteractionDatasetRecorder








@parser.wrap()
def main(cfg: RLTrainPipelineConfig):
    cfg.validate()
    # 1. Initialize the components
    rl = RLObjects(cfg)

    # 2. Assemble the blocks
    data_collector = InteractionDatasetRecorder(rl)

    # 3. Execute
    results = data_collector.record()

    # 4. Use the results
    if rl.logger:
        rl.logger.log_metrics({"mean_reward": np.mean(results["sum_rewards"])}, step=0, prefix="eval")


if __name__ == "__main__":
    main()
