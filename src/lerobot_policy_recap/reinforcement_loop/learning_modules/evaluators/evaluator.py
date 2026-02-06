import time
import logging
import threading
import concurrent.futures as cf
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict
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

class BatchedEvaluator:
    """The LEGO block for benchmarking a policy across vectorized environments."""
    def __init__(self, rl: RLObjects):
        self.rl = rl
        self.output_dir = rl.output_dir / "eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
    def _rollout(self, env: gym.vector.VectorEnv, seeds: list[int] | None = None, render_callback=None) -> dict:
        """Atomic execution of one vectorized rollout."""
        self.rl.policy.reset()
        observation, info = env.reset(seed=seeds)
        if render_callback: render_callback(env)

        rollout_data = {"rewards": [], "success": [], "done": []}
        step = 0
        done = np.array([False] * env.num_envs)
        
        # Get max steps concretely from the environment instance
        max_steps = env.call("_max_episode_steps")[0]
            
        print(f"Using max_steps={max_steps} for evaluation rollouts.") 
        while not np.all(done) and step < max_steps:
            # 1. Transform Observation
            observation = preprocess_observation(observation)
            observation = add_envs_task(env, observation)
            observation = self.rl.env_preprocessor(observation)
            observation = self.rl.preprocessor(observation)
            
            # 2. Policy Inference
            with torch.inference_mode():
                action = self.rl.policy.select_action(observation)
            
            # 3. Transform Action & Step Env
            action = self.rl.postprocessor(action)
            action_transition = self.rl.env_postprocessor({"action": action})
            action_np = action_transition["action"].to("cpu").numpy()

            observation, reward, terminated, truncated, info = env.step(action_np)
            if render_callback: render_callback(env)

            # 4. Metric Capture
            rollout_data["rewards"].append(reward)
            successes = info["final_info"]["is_success"].tolist() if "final_info" in info else [False] * env.num_envs
            rollout_data["success"].append(np.array(successes))

            done = terminated | truncated | done
            if step >= max_steps-1:
                done = np.array([True] * env.num_envs)
                print("Max steps reached, terminating rollout.")
            rollout_data["done"].append(done.copy())
            step += 1
        print(done)
        return {k: np.array(v) for k, v in rollout_data.items()}

    def _eval_single_env(self, env: gym.vector.VectorEnv, task_name: str, start_seed: int) -> Dict[str, Any]:
        """Evaluates one specific task/group."""
        n_episodes = self.rl.cfg.eval.n_episodes
        max_rendered = 10 
        n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
        
        metrics = defaultdict(list)
        n_rendered = 0
        v_dir = self.output_dir / "videos" / task_name
        v_dir.mkdir(parents=True, exist_ok=True)
        threads = []

        for batch_ix in range(n_batches):
            ep_frames = []
            
            def render_frame(e):
                nonlocal n_rendered
                if n_rendered >= max_rendered: return
                frames = e.call("render")[:min(max_rendered - n_rendered, e.num_envs)]
                ep_frames.append(np.stack(frames))

            seeds = list(range(start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)))
            data = self._rollout(env, seeds=seeds, render_callback=render_frame if max_rendered > 0 else None)

            # Process Rewards/Success
            # Note: We take the last recorded success for each env in the vector
            done_indices = np.argmax(data["done"], axis=0)
            for i in range(env.num_envs):
                metrics["sum_rewards"].append(data["rewards"][:, i].sum())
                metrics["max_rewards"].append(data["rewards"][:, i].max())
                # If success was recorded in any step for this env
                if len(data["success"]) > 0:
                    metrics["successes"].append(data["success"][:, i].any())

            # Async Video Saving
            if max_rendered > 0 and ep_frames:
                batch_frames = np.stack(ep_frames, axis=1) 
                for i in range(min(batch_frames.shape[0], max_rendered - n_rendered)):
                    path = v_dir / f"ep_{n_rendered}.mp4"
                    t = threading.Thread(target=write_video, args=(str(path), batch_frames[i, :done_indices[i]+1], env.unwrapped.metadata["render_fps"]))
                    t.start()
                    threads.append(t)
                    metrics["video_paths"].append(str(path))
                    n_rendered += 1

        for t in threads: t.join()
        return dict(metrics)
    def _save_results(self, overall_results: Dict[str, list], duration: float) -> Dict[str, Any]:
        """Aggregates metrics, logs to console, and saves to a JSON file."""
        
        # 1. Calculate Statistics
        # We use .get() to avoid KeyErrors if the rollout failed early
        successes = overall_results.get("successes", [])
        sum_rewards = overall_results.get("sum_rewards", [])
        max_rewards = overall_results.get("max_rewards", [])

        final_metrics = {
            "avg_success_rate": float(np.mean(successes)) if successes else 0.0,
            "avg_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
            "max_reward_achieved": float(np.max(max_rewards)) if max_rewards else 0.0,
            "total_episodes": len(successes),
            "duration_seconds": round(duration, 2),
            "seeds_used": self.rl.cfg.seed
        }

        # 2. Log to Console
        logging.info("--- Benchmark Results ---")
        logging.info(f"Total Episodes: {final_metrics['total_episodes']}")
        logging.info(colored(f"Success Rate: {final_metrics['avg_success_rate']:.2%}", "green" if final_metrics['avg_success_rate'] > 0 else "red"))
        logging.info(f"Average Reward: {final_metrics['avg_sum_reward']:.2f}")
        logging.info(f"Time Elapsed: {final_metrics['duration_seconds']}s")
        logging.info("-------------------------")

        # 3. Save to Disk
        save_path = self.output_dir / "results.json"
        with open(save_path, "w") as f:
            json.dump(final_metrics, f, indent=4)
        
        logging.info(f"Final metrics saved to {colored(str(save_path), 'cyan')}")
        
        return final_metrics
    def benchmark(self):
        if self.rl.cfg.eval.n_episodes <= 0 or not self.rl.cfg.eval.enable or self.rl.env is None:
            logging.info(colored("Evaluation skipped as per configuration.", "yellow"))
            return {}
        """Main entry point: Runs evaluation on all tasks."""
        logging.info(colored("Starting LEGO Benchmark...", "cyan"))
        self.rl.policy.eval()
        start_time = time.time()

        tasks = [(tg, tid, vec) for tg, group in self.rl.envs.items() for tid, vec in group.items()] if isinstance(self.rl.envs, dict) else [("default", 0, self.rl.envs)]
        overall_results = defaultdict(list)
        
        with torch.no_grad():
            with cf.ThreadPoolExecutor(max_workers=getattr(self.rl.cfg.env, "max_parallel_tasks", 1)) as executor:
                futures = {executor.submit(self._eval_single_env, env, f"{tg}_{tid}", self.rl.cfg.seed): (tg, tid) for tg, tid, env in tasks}
                for fut in cf.as_completed(futures):
                    res = fut.result()
                    for k, v in res.items(): 
                        overall_results[k].extend(v)

        # Separate call for saving and logging
        duration = time.time() - start_time
        self._save_results(overall_results, duration)
        
        return overall_results