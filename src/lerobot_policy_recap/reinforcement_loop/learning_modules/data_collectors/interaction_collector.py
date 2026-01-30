import logging
import datetime
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List
from termcolor import colored

from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot_policy_recap.reinforcement_loop.common.utils.utils import drop_non_observation_keys, to_cpu
from lerobot_policy_recap.reinforcement_loop.learning_modules.data_collectors.utils import (
    _flatten_envs,
    _extract_success_from_info,
    _scalar_np,
)
from lerobot.utils.constants import ACTION, REWARD, DONE, OBS_PREFIX


@dataclass
class EpisodeFrames:
    """Temporal buffer for a single episode's data."""
    obs: List[Dict[str, Any]] = field(default_factory=list)
    act: List[torch.Tensor] = field(default_factory=list)
    rew: List[float] = field(default_factory=list)
    done: List[bool] = field(default_factory=list)
    success: List[bool] = field(default_factory=list)
    task: List[str] = field(default_factory=list)
    gen_new_chunk: List[bool] = field(default_factory=list)

    def reset(self):
        self.obs.clear()
        self.act.clear()
        self.rew.clear()
        self.done.clear()
        self.success.clear()
        self.task.clear()
        self.gen_new_chunk.clear()

class InteractionDatasetRecorder:
    def __init__(self, rl: any):
        self.rl = rl
        self.device = rl.device
        
        # Config params
        self.num_episodes = getattr(rl.cfg.eval, "n_episodes", 200)
        self.fps = getattr(rl.cfg.env, "fps", 30)
        self.video_enabled = getattr(rl.cfg.eval, "use_videos", True)
        
        # Setup path
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.repo_id = rl.output_dir / f"interaction_ds_{timestamp}"
        
        self.tasks = _flatten_envs(rl.envs)
        
        # Get batch size from vectorized env
        suite, task_id, vec_env = self.tasks[0]
        self.num_envs = getattr(vec_env, "num_envs", 1)
        
        self.dataset = self._init_dataset()
        print(colored(f"Initialized Interaction Dataset at: {self.repo_id} with {self.num_envs} parallel envs", "cyan"))
        # Create separate buffer for each parallel environment
        self.buffers = [EpisodeFrames() for _ in range(self.num_envs)]
        self.recorded_episodes = 0

        logging.info(colored(f"Interaction Dataset initialized at: {self.repo_id} with {self.num_envs} parallel envs", "cyan"))

    def _init_dataset(self) -> LeRobotDataset:
        """Initializes the LeRobotDataset using shapes from a dummy environment reset."""
        dataset_features = combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=self.rl.env_postprocessor,
                    initial_features=create_initial_features(
                        action=self.rl.policy.config.output_features
                    ),
                    use_videos=self.video_enabled,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=self.rl.env_preprocessor,
                    initial_features=create_initial_features(observation=self.rl.policy.config.input_features),
                    use_videos=self.video_enabled,
                ),
            )
        dataset_features[REWARD] = {"dtype": "float32", "names":"reward" ,"shape": (1,)}
        dataset_features[DONE] = {"dtype": "bool", "names":"done" ,"shape": (1,)}
        dataset_features["gen_new_chunk"] = {"dtype": "bool", "names":"gen_new_chunk" ,"shape": (1,)}
        dataset_features["success"] = {"dtype": "bool", "names":"success" ,"shape": (1,)}
        return LeRobotDataset.create(
            repo_id=str(self.repo_id),
            fps=self.fps,
            features=dataset_features,
            use_videos=self.video_enabled,
        )

    def flush_episode(self, buffer: EpisodeFrames):
        """Writes the buffered episode frames to the physical dataset."""
        num_frames = len(buffer.rew)
        if num_frames == 0:
            return None

        for t in range(num_frames):
            frame_data = {
                **buffer.obs[t],
                ACTION: buffer.act[t],
                REWARD: torch.tensor([buffer.rew[t]], dtype=torch.float32),
                DONE: torch.tensor([buffer.done[t]], dtype=torch.bool),
                "success": torch.tensor([buffer.success[t]], dtype=torch.bool),
                "gen_new_chunk": torch.tensor([buffer.gen_new_chunk[t]], dtype=torch.bool),
                "task": buffer.task[t],
            }
            self.dataset.add_frame(frame_data)
        
        self.dataset.save_episode()
        success = buffer.success[-1] if buffer.success else False
        buffer.reset()
        return num_frames, success

    def flush_all_episodes(self):
        """Flushes all episode buffers to the dataset."""
        total_frames = 0
        successes = []
        
        for env_idx, buffer in enumerate(self.buffers):
            result = self.flush_episode(buffer)
            if result:
                num_frames, success = result
                total_frames += num_frames
                successes.append(success)
                self.recorded_episodes += 1
                logging.info(colored(f"Saved Episode {self.recorded_episodes} (env {env_idx}). Success: {success}", "green"))
        
        if total_frames > 0:
            num_successes = sum(successes)
            num_episodes = len(successes)
            success_rate = num_successes / num_episodes if num_episodes > 0 else 0.0
            
            logging.info(colored(
                f"Saved {num_episodes} episodes ({total_frames} total frames) to dataset, "
                f"success rate: {num_successes}/{num_episodes} ({success_rate:.2%})", "cyan"
            ))
            
            # Log metrics to wandb/other backends
            if self.rl.logger:
                self.rl.logger.log_metrics({
                    "success_rate": success_rate,
                    "num_successes": num_successes,
                    "num_episodes": num_episodes,
                    "total_frames": total_frames,
                    "recorded_episodes": self.recorded_episodes,
                }, step=self.recorded_episodes, prefix="data_collection")

    def record(self):
        self.rl.policy.eval()
        suite, task_id, vec_env = self.tasks[0] 
        max_steps = vec_env.call("_max_episode_steps")[0]
        logging.info(f"Beginning recording of {self.num_episodes} episodes to dataset at {self.repo_id}, with max {max_steps} steps per episode.")

        with VideoEncodingManager(self.dataset):
            while self.recorded_episodes < self.num_episodes:
                logging.info(f"Collecting episodes {self.recorded_episodes + 1}-{min(self.recorded_episodes + self.num_envs, self.num_episodes)}/{self.num_episodes}")
                
                self.rl.policy.reset()
                observation, info = vec_env.reset()
                
                # Track which envs are still running
                env_done = [False] * self.num_envs
                step = 0
                
                while not all(env_done) and step < max_steps:
                    print(colored(f"Recording step {step+1}/{max_steps}", "cyan"), end="\r")
                    # 1. Processing
                    obs_raw = preprocess_observation(observation)
                    obs_raw = add_envs_task(vec_env, obs_raw)

                    obs_proc = self.rl.env_preprocessor(obs_raw)
                    obs_policy = self.rl.preprocessor(obs_proc)

                    # 2. Inference
                    with torch.inference_mode():
                        out = self.rl.policy.select_action(obs_policy)
                        action_raw = out[0] if isinstance(out, tuple) else out
                        print(colored(f"Sampled actions", "yellow"))
                    
                    # 3. Step
                    action_proc = self.rl.postprocessor(action_raw)
                    action_tr = self.rl.env_postprocessor({"action": action_proc})
                    action_np = action_tr["action"].to("cpu").numpy()

                    next_obs, reward, terminated, truncated, info_env = vec_env.step(action_np)
                    
                    # 4. Buffer current frame for each environment
                    obs_frames_cpu = drop_non_observation_keys(to_cpu(obs_proc))
                    action_proc_cpu = to_cpu(action_proc)
                    
                    for env_idx in range(self.num_envs):
                        if env_done[env_idx]:
                            continue  # Skip already finished envs
                        
                        # Extract per-env observation (index into batch dimension)
                        obs_frame = {
                            k: v[env_idx] if isinstance(v, torch.Tensor) else v[env_idx] if isinstance(v, (list, np.ndarray)) else v
                            for k, v in obs_frames_cpu.items()
                        }
                        
                        task_str = str(obs_raw["task"][env_idx])
                        is_done = bool(terminated[env_idx] | truncated[env_idx] | (step + 1 >= max_steps))
                        
                        self.buffers[env_idx].obs.append(obs_frame)
                        self.buffers[env_idx].act.append(action_proc_cpu[env_idx])
                        self.buffers[env_idx].rew.append(float(reward[env_idx]))
                        self.buffers[env_idx].done.append(is_done)
                        self.buffers[env_idx].success.append(bool(_extract_success_from_info(info_env, env_index=env_idx)))
                        self.buffers[env_idx].task.append(task_str)
                        self.buffers[env_idx].gen_new_chunk.append(False)
                        
                        # Mark env as done if terminated or truncated
                        if terminated[env_idx] | truncated[env_idx]:
                            env_done[env_idx] = True

                    observation = next_obs
                    step += 1

                # End of rollout: commit all buffers to dataset
                self.flush_all_episodes()

        return str(self.repo_id)

class InteractionBufferRecorder:
    def __init__(self, rl: any):
        self.rl = rl
        self.device = rl.device
        
        # Config params
        self.fps = getattr(rl.cfg.env, "fps", 30)
        
        self.tasks = _flatten_envs(rl.envs)
        # Get batch size from vectorized env
        suite, task_id, vec_env = self.tasks[0]
        self.num_envs = getattr(vec_env, "num_envs", 1)
        
        # Create separate buffer for each parallel environment
        self.buffers = [EpisodeFrames() for _ in range(self.num_envs)]
        self.recorded_episodes = 0

        logging.info(colored(f"Interaction Buffer Recorder initialized with {self.num_envs} parallel envs", "cyan"))

        self.encoder = self._identify_encoder()
        if self.encoder:
            self.encoder.eval()
            self.encoder.to(self.device)
            logging.info(colored(f"Helper encoder found: {type(self.encoder).__name__}. Caching enabled.", "cyan"))

    def _identify_encoder(self) -> torch.nn.Module:
        policy = self.rl.policy
        if hasattr(policy, "encoder_v_critic"):
            return policy.encoder_v_critic
        elif hasattr(policy, "encoder"):
            return policy.encoder
        return None

    def flush_episode(self, buffer: EpisodeFrames):
        """Writes the buffered episode frames to the online buffer."""
        num_frames = len(buffer.rew)
        if num_frames == 0:
            return None

        for t in range(num_frames):
            frame_data = {
                **buffer.obs[t],
                ACTION: buffer.act[t],
                REWARD: torch.tensor([buffer.rew[t]], dtype=torch.float32),
                DONE: torch.tensor([buffer.done[t]], dtype=torch.bool),
                "success": torch.tensor([buffer.success[t]], dtype=torch.bool),
                "gen_new_chunk": torch.tensor([buffer.gen_new_chunk[t]], dtype=torch.bool),
                "task": buffer.task[t],
            }
            # Add to online buffer
            self.rl.online_buffer.add_frame(frame_data)
        
        success = buffer.success[-1] if buffer.success else False
        buffer.reset()
        return num_frames, success

    def flush_all_episodes(self):
        """Flushes all episode buffers to the online buffer."""
        total_frames = 0
        successes = []
        
        for env_idx, buffer in enumerate(self.buffers):
            result = self.flush_episode(buffer)
            if result:
                num_frames, success = result
                total_frames += num_frames
                successes.append(success)
                self.recorded_episodes += 1
                logging.info(colored(f"Buffered Episode {self.recorded_episodes} (env {env_idx}). Success: {success}", "green"))
        
        if total_frames > 0:
            num_successes = sum(successes)
            num_episodes = len(successes)
            success_rate = num_successes / num_episodes if num_episodes > 0 else 0.0
            buffer_size = len(self.rl.online_buffer)
            
            logging.info(colored(
                f"Flushed {total_frames} total frames from {num_episodes} episodes to online buffer, "
                f"new buffer size: {buffer_size}, success rate: {num_successes}/{num_episodes} ({success_rate:.2%})", "cyan"
            ))
            print(colored(
                f"Flushed {total_frames} total frames from {num_episodes} episodes to online buffer, "
                f"new buffer size: {buffer_size}, success rate: {num_successes}/{num_episodes} ({success_rate:.2%})", "cyan"
            ))
            
            # Log metrics to wandb/other backends
            if self.rl.logger:
                self.rl.logger.log_metrics({
                    "success_rate": success_rate,
                    "num_successes": num_successes,
                    "num_episodes": num_episodes,
                    "total_frames": total_frames,
                    "online_buffer_size": buffer_size,
                    "recorded_episodes": self.recorded_episodes,
                }, step=self.recorded_episodes, prefix="data_collection")

    def record_episodes(self):
        """Record episodes from all parallel environments simultaneously."""
        self.rl.policy.eval()
        suite, task_id, vec_env = self.tasks[0] 
        max_steps = vec_env.call("_max_episode_steps")[0]

        self.rl.policy.reset()
        observation, info = vec_env.reset()
        
        # Track which envs are still running
        env_done = [False] * self.num_envs
        step = 0
        
        while not all(env_done) and step < max_steps:
            # 1. Processing
            obs_raw = preprocess_observation(observation)
            obs_raw = add_envs_task(vec_env, obs_raw)

            obs_proc = self.rl.env_preprocessor(obs_raw)
            obs_policy = self.rl.preprocessor(obs_proc)

            # 2. Inference
            with torch.inference_mode():
                out = self.rl.policy.select_action(obs_policy)
                action_raw = out[0] if isinstance(out, tuple) else out
            
            # 3. Step
            action_proc = self.rl.postprocessor(action_raw)
            action_tr = self.rl.env_postprocessor({"action": action_proc})
            action_np = action_tr["action"].to("cpu").numpy()

            next_obs, reward, terminated, truncated, info_env = vec_env.step(action_np)
            
            # --- Caching Features (New) ---
            cached_feats = {}
            if self.encoder is not None:
                # Filter strictly observation keys for the encoder (like in dataset_encoder.py)
                obs_enc_input = {k: v for k, v in obs_policy.items() if k.startswith(OBS_PREFIX)}
                with torch.inference_mode():
                    feats = self.encoder(obs_enc_input)
                # Store as CPU tensors for buffering
                cached_feats = {
                    f"observation.cache.{k}": v.cpu() for k, v in feats.items()
                }

            # 4. Buffer current frame for each environment
            obs_frames_cpu = drop_non_observation_keys(to_cpu(obs_proc))
            action_proc_cpu = to_cpu(action_proc)
            
            for env_idx in range(self.num_envs):
                if env_done[env_idx]:
                    continue  # Skip already finished envs
                
                # Extract per-env observation (index into batch dimension)
                obs_frame = {
                    k: v[env_idx] if isinstance(v, torch.Tensor) else v[env_idx] if isinstance(v, (list, np.ndarray)) else v
                    for k, v in obs_frames_cpu.items()
                }

                # Add cached features to the frame
                for k, v in cached_feats.items():
                    obs_frame[k] = v[env_idx] if isinstance(v, torch.Tensor) else v[env_idx]
                
                task_str = str(obs_raw["task"][env_idx])
                is_done = bool(terminated[env_idx] | truncated[env_idx] | (step + 1 >= max_steps))
                
                self.buffers[env_idx].obs.append(obs_frame)
                self.buffers[env_idx].act.append(action_proc_cpu[env_idx])
                self.buffers[env_idx].rew.append(float(reward[env_idx]))
                self.buffers[env_idx].done.append(is_done)
                self.buffers[env_idx].success.append(bool(_extract_success_from_info(info_env, env_index=env_idx)))
                self.buffers[env_idx].task.append(task_str)
                self.buffers[env_idx].gen_new_chunk.append(False)
                
                # Mark env as done if terminated or truncated
                if terminated[env_idx] | truncated[env_idx]:
                    env_done[env_idx] = True

            observation = next_obs
            step += 1

        # End of rollout: commit all buffers to online buffer
        self.flush_all_episodes()

        return "online_buffer"
    
    # Alias for backwards compatibility
    def record_episode(self):
        """Alias for record_episodes for backwards compatibility."""
        return self.record_episodes()