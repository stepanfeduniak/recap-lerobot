import time
import logging
import threading
import concurrent.futures as cf
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Dict, List
import copy

import torch
import numpy as np
import gymnasium as gym
from termcolor import colored

from lerobot.envs.utils import (
    check_env_attributes_and_types, 
    preprocess_observation, 
    add_envs_task
)
import json
from lerobot.utils.io_utils import write_video
from lerobot_policy_recap.reinforcement_loop.learning_modules.rl_objects import RLObjects

import torch
import numpy as np
import logging
import threading
from collections import defaultdict
from typing import Any, Dict
from termcolor import colored

from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.utils.io_utils import write_video
from lerobot_policy_recap.reinforcement_loop.learning_modules.evaluators.evaluator import BatchedEvaluator
from lerobot_policy_recap.reinforcement_loop.common.visualizers.visualizer import ConsistencyMatplotlibPanel

class CriticEvaluator(BatchedEvaluator):
    """
    Subclass of BatchedEvaluator specialized for visualizing and 
    benchmarking the Value function (Critic) alongside policy performance.
    """
    def __init__(self, rl: any):
        super().__init__(rl)
        # Toggle between 'mean' and 'min' of the ensemble from config
        self.critic_reduction = getattr(rl.cfg.eval, "critic_reduction", "min")
        logging.info(f"Initialized CriticEvaluator: reduction={self.critic_reduction}")

    def _rollout(self, env, seeds: list[int] | None = None, render_callback=None) -> dict:
        self.rl.policy.reset()
        observation, info = env.reset(seed=seeds)
        
        rollout_data = {
            "rewards": [], 
            "success": [], 
            "done": [], 
            "critic_values": [] # (T, E)
        }
        
        step = 0
        done = np.array([False] * env.num_envs)
        # Use config max steps or env default
        max_steps = env.call("_max_episode_steps")[0]
        
        # Action queue setup (like STAC)
        n_action_steps = self.rl.policy.config.n_action_steps
        action_queues: List[deque] = [deque(maxlen=n_action_steps) for _ in range(env.num_envs)]
        
        # Track last critic value per env (to reuse when not sampling)
        last_critic_vals = [None] * env.num_envs
        
        while not np.all(done) and step < max_steps:
            print(colored(f"Eval Step: {step}/{max_steps}", "cyan"), end="\r")
            # 1. Transform Observation
            observation = preprocess_observation(observation)
            observation = add_envs_task(env, observation)
            observation = self.rl.env_preprocessor(observation)
            observation = self.rl.preprocessor(observation)
            
            # Check if any env needs new action samples
            needs_sampling = any(len(action_queues[i]) == 0 and not done[i] for i in range(env.num_envs))
            
            critic_val = None
            
            with torch.inference_mode():
                device_obs = {
                        k: v.to(self.rl.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in observation.items()
                    }
                    
                # 2. Extract Critic Value (only when sampling new actions)
                val_info = self.rl.policy._value_info(device_obs)
                critic_val = val_info["v_min"] if self.critic_reduction == "min" else val_info["v_mean"]
                if needs_sampling:
                    # 3. Get action chunk from policy
                    action_chunk = self.rl.policy.predict_action_chunk(device_obs)
                    
                    # Fill action queue for each env that needs it
                    for env_idx in range(env.num_envs):
                        if done[env_idx]:
                            continue
                        
                        if len(action_queues[env_idx]) == 0:
                            # Fill action queue for the next n_action_steps
                            for i in range(min(n_action_steps, action_chunk.shape[1])):
                                action_queues[env_idx].append(action_chunk[env_idx, i, :])
                for env_idx in range(env.num_envs):
                    # Store critic value for this env
                    last_critic_vals[env_idx] = critic_val[env_idx:env_idx+1]
                
            # Build action tensor from queues
            reference_action = None
            for env_idx in range(env.num_envs):
                if len(action_queues[env_idx]) > 0:
                    reference_action = action_queues[env_idx][0]
                    break
            
            action_list = []
            for env_idx in range(env.num_envs):
                if done[env_idx] or len(action_queues[env_idx]) == 0:
                    if reference_action is not None:
                        action_list.append(torch.zeros_like(reference_action))
                    else:
                        raise RuntimeError("No reference action available to determine action shape/device")
                else:
                    action_list.append(action_queues[env_idx].popleft())
            
            action = torch.stack(action_list, dim=0)

            # 4. Transform Action & Step Env
            action = self.rl.postprocessor(action)
            action_transition = self.rl.env_postprocessor({"action": action})
            action_np = action_transition["action"].to("cpu").numpy()

            observation, reward, terminated, truncated, info = env.step(action_np)
            
            # Use the last computed critic value for each env
            current_critic_vals = torch.cat([
                last_critic_vals[i] if last_critic_vals[i] is not None 
                else torch.zeros(1, device=self.rl.device) 
                for i in range(env.num_envs)
            ], dim=0)
            
            # 5. Render with critic overlay
            if render_callback:
                render_callback(env, critic_val=current_critic_vals)

            # 6. Metric Capture
            rollout_data["rewards"].append(reward)
            rollout_data["critic_values"].append(current_critic_vals.cpu().numpy())
            
            successes = info["final_info"]["is_success"].tolist() if "final_info" in info else [False] * env.num_envs
            rollout_data["success"].append(np.array(successes))

            done = terminated | truncated | done
            if step >= max_steps - 1:
                done = np.ones_like(done, dtype=bool)
                
            rollout_data["done"].append(done.copy())
            step += 1

        return {k: np.array(v) for k, v in rollout_data.items()}

    def _eval_single_env(self, env, task_name: str, start_seed: int) -> Dict[str, Any]:
        n_episodes = self.rl.cfg.eval.n_episodes
        max_rendered = 10 
        n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
        
        metrics = defaultdict(list)
        n_rendered = 0
        v_dir = self.output_dir / "videos" / task_name
        v_dir.mkdir(parents=True, exist_ok=True)
        threads = []

        # Initialize overlays for the vector env
        overlays = [ConsistencyMatplotlibPanel(cumulative_mode="mean") for _ in range(env.num_envs)]

        for batch_ix in range(n_batches):
            ep_frames = []

            def render_with_overlay(e, critic_val=None):
                nonlocal n_rendered
                if n_rendered >= max_rendered: return
                n_to_render = min(max_rendered - n_rendered, e.num_envs)
                
                # Fetch raw frames
                raw_frames = e.call("render")[:n_to_render]
                
                processed_frames = []
                for i in range(n_to_render):
                    val = critic_val[i].item() if critic_val is not None else None
                    if critic_val is None: overlays[i].reset()
                    # Feed the critic value into the plot panel (the 'consistency' panel reused for 'Value')
                    f = overlays[i].update(raw_frames[i], val)
                    processed_frames.append(f)
                
                ep_frames.append(np.stack(processed_frames))

            seeds = list(range(start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)))
            data = self._rollout(env, seeds=seeds, render_callback=render_with_overlay)

            # Process Rewards/Success/Values
            done_indices = np.argmax(data["done"], axis=0)
            for i in range(env.num_envs):
                length = done_indices[i] + 1
                metrics["sum_rewards"].append(data["rewards"][:length, i].sum())
                metrics["max_rewards"].append(data["rewards"][:length, i].max())
                metrics["avg_values"].append(data["critic_values"][:length, i].mean())
                
                if len(data["success"]) > 0:
                    metrics["successes"].append(data["success"][:length, i].any())

            # Async Video Saving
            if max_rendered > 0 and ep_frames:
                batch_frames = np.stack(ep_frames, axis=1) 
                for i in range(min(batch_frames.shape[0], max_rendered - n_rendered)):
                    path = v_dir / f"ep_{n_rendered}.mp4"
                    t = threading.Thread(
                        target=write_video, 
                        args=(str(path), batch_frames[i, :done_indices[i]+1], env.unwrapped.metadata["render_fps"])
                    )
                    t.start()
                    threads.append(t)
                    metrics["video_paths"].append(str(path))
                    n_rendered += 1

        for t in threads: t.join()
        return dict(metrics)

    def _save_results(self, overall_results: Dict[str, list], duration: float) -> Dict[str, Any]:
        # Add 'avg_values' to the standard results report
        final_metrics = super()._save_results(overall_results, duration)
        
        avg_v = float(np.mean(overall_results.get("avg_values", [0])))
        final_metrics["avg_value_estimate"] = avg_v
        
        logging.info(f"Average Value Estimate: {avg_v:.4f}")
        return final_metrics