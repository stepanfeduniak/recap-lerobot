#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot_policy_recap.processor import (
    ReturnNormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


@ProcessorStepRegistry.register(name="Recap_PI_prepare_state_tokenizer_processor_step")
@dataclass
class Recap_PIPrepareStateTokenizerProcessorStep(Pi05PrepareStateTokenizerProcessorStep):
    """
    Prepare state and create three prompt variants:
      - task (standard)
      - task_pos (advantage: positive)
      - task_neg (advantage: negative)

    This mirrors PI05's prompt formatting, only adding the advantage phrase for pos/neg.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for Recap_PI/PI05")

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # Keep behavior aligned with PI05: deepcopy, then pad + discretize
        state = deepcopy(state)
        state = pad_vector(state, self.max_state_dim)

        # Discretize into 256 bins; PI05 expects state already normalized to [-1, 1]
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts: list[str] = []
        full_prompts_pos: list[str] = []
        full_prompts_neg: list[str] = []
        full_prompts_critic: list[str] = []

        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))

            # Standard (identical to PI05)
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

            # Advantage variants (only change is injected phrase)
            full_prompts_pos.append(
                f"Task: {cleaned_text} Advantage: positive, State: {state_str};\nAction: "
            )
            full_prompts_neg.append(
                f"Task: {cleaned_text} Advantage: negative, State: {state_str};\nAction: "
            )
            # Critic prompt (only the task instruction)
            full_prompts_critic.append(cleaned_text)

        # Store prompts under keys expected by subsequent tokenizer steps
        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA][f"{self.task_key}_pos"] = full_prompts_pos
        transition[TransitionKey.COMPLEMENTARY_DATA][f"{self.task_key}_neg"] = full_prompts_neg
        transition[TransitionKey.COMPLEMENTARY_DATA][f"{self.task_key}_critic"] = full_prompts_critic

        return transition


@ProcessorStepRegistry.register(name="Recap_PI_tokenizer_processor")
@dataclass
class Recap_PITokenizerProcessorStep(TokenizerProcessorStep):
    """
    PI05-faithful tokenizer wrapper:
      - Delegates to TokenizerProcessorStep.observation() (so behavior matches PI05)
      - Optionally mirrors the produced base keys into suffixed keys (e.g. _pos/_neg)

    Important:
      - To avoid accidentally deleting the *standard* token keys, the default is to keep base keys.
      - If you want a "keys are only suffixed" world, do NOT run the standard TokenizerProcessorStep,
        and set keep_base_keys=False for all instances (including a "_std" suffix).
    """

    output_key_suffix: str = ""
    keep_base_keys: bool = True

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        # No suffix -> use standard parent behavior (for the first/standard tokenization)
        if not self.output_key_suffix:
            return super().observation(observation)

        # Get the task from the transition
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")

        # Tokenize the task (creates CPU tensors)
        tokenized_prompt = self._tokenize_text(task)

        # Detect device from existing tensors
        target_device = self._detect_device(self.transition)

        # Move tokenized tensors to the detected device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Write directly to suffixed keys, NOT touching the base keys
        suff_input_ids_key = f"{OBS_LANGUAGE_TOKENS}{self.output_key_suffix}"
        suff_attention_mask_key = f"{OBS_LANGUAGE_ATTENTION_MASK}{self.output_key_suffix}"

        new_observation = observation.copy()
        new_observation[suff_input_ids_key] = tokenized_prompt["input_ids"]
        new_observation[suff_attention_mask_key] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def transform_features(
        self, features: dict[Any, dict[str, Any]]
    ) -> dict[Any, dict[str, Any]]:
        # First, let the base class add the base language features (PI05 behavior)
        features = super().transform_features(features)

        # No suffix -> identical to PI05
        if not self.output_key_suffix:
            return features

        obs_feats = features[PipelineFeatureType.OBSERVATION]

        suff_input_ids_key = f"{OBS_LANGUAGE_TOKENS}{self.output_key_suffix}"
        suff_attention_mask_key = f"{OBS_LANGUAGE_ATTENTION_MASK}{self.output_key_suffix}"

        if suff_input_ids_key not in obs_feats:
            obs_feats[suff_input_ids_key] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        if suff_attention_mask_key not in obs_feats:
            obs_feats[suff_attention_mask_key] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # If we don't keep base keys, remove them from the feature schema too
        if not self.keep_base_keys:
            obs_feats.pop(OBS_LANGUAGE_TOKENS, None)
            obs_feats.pop(OBS_LANGUAGE_ATTENTION_MASK, None)
        return features


def get_inner_config(config: Any) -> Optional[Any]:
    """
    Helper to retrieve initialization arguments for Recap_PI's internal policy.
    Tries common attribute names used by wrapper policies.
    """
    path = getattr(config, "diffusion_policy_path", None)
    if path is None:
        path = getattr(config, "diffusion_repo_id", None)
    if path is None:
        path = getattr(config, "pretrained_path", None)

    if path:
        return PreTrainedConfig.from_pretrained(path)
    return None

def make_recap_pre_post_processors(
    config: Any,
    preprocessor_overrides: dict[str, dict[str, torch.Tensor]] | None = None,
    return_stats: dict[str, float] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the RECAP policy.

    Loads the base preprocessor from the pretrained diffusion policy (PI05) and extends
    it with RECAP-specific processing steps:
    1. Replaces Pi05PrepareStateTokenizerProcessorStep with Recap version (creates pos/neg/critic prompts)
    2. Adds tokenizer steps for pos/neg/gemma3 suffixes
    3. Adds ReturnNormalizerProcessorStep for return/reward normalization

    The post-processing pipeline is kept as-is from the pretrained model.

    Args:
        config: The configuration object for the RECAP policy,
            containing diffusion_repo_id for the underlying policy.
        preprocessor_overrides: Optional overrides for processor steps.
        return_stats: Optional statistics for return/reward normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    pretrained_path = config.diffusion_repo_id
    print(f"[RECAP Processor] Loading base processor from: {pretrained_path}")
    
    # Load the policy config to check the policy type
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
    
    # Apply smol_vla patch: set pad_language_to to "max_length"
    if policy_config.type == "smolvla":
        print("[RECAP Processor] Detected smolvla policy - applying pad_language_to='max_length' patch")
        if preprocessor_overrides is None:
            preprocessor_overrides = {}
        if "tokenizer_processor" not in preprocessor_overrides:
            preprocessor_overrides["tokenizer_processor"] = {}
        preprocessor_overrides["tokenizer_processor"]["padding"] = "max_length"
    
    # 1. Load base preprocessor from pretrained
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=None, 
        pretrained_path=pretrained_path, 
        preprocessor_overrides=preprocessor_overrides
    )
    print(f"[RECAP Processor] Loaded preprocessor with {len(preprocessor.steps)} steps")
    
    # 2. Build new steps list, replacing Pi05PrepareStateTokenizerProcessorStep with RECAP version
    new_steps: list[ProcessorStep] = []
    base_tokenizer_idx = None
    base_tokenizer_max_length = None
    
    for i, step in enumerate(preprocessor.steps):
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep):
            # Replace with RECAP version that creates pos/neg/critic prompts
            print(f"[RECAP Processor] Replacing Pi05PrepareStateTokenizerProcessorStep with RECAP version (max_state_dim={step.max_state_dim})")
            new_steps.append(Recap_PIPrepareStateTokenizerProcessorStep(
                max_state_dim=step.max_state_dim,
                task_key=step.task_key
            ))
        elif isinstance(step, TokenizerProcessorStep):
            # Keep track of base tokenizer position and max_length for RECAP tokenizers
            new_steps.append(step)
            base_tokenizer_idx = len(new_steps) - 1
            base_tokenizer_max_length = step.max_length
            print(f"[RECAP Processor] Found base TokenizerProcessorStep at index {base_tokenizer_idx} (max_length={base_tokenizer_max_length})")
        else:
            new_steps.append(step)
    
    if base_tokenizer_idx is None:
        raise ValueError("Could not find TokenizerProcessorStep in pretrained preprocessor")
    
    # Use loaded tokenizer max_length, fallback to config value
    tokenizer_max_length_pi = base_tokenizer_max_length or getattr(config, "tokenizer_max_length", 200)
    tokenizer_max_length_recap = getattr(config, "tokenizer_max_length", 200)
    
    # 3. Create RECAP-specific tokenizer steps
    recap_tokenizer_steps = [
        # Positive advantage tokenization -> suffixed keys
        Recap_PITokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=tokenizer_max_length_pi,
            padding_side="right",
            padding="max_length",
            task_key="task_pos",
            output_key_suffix="_pos",
            keep_base_keys=True,
        ),
        # Negative advantage tokenization -> suffixed keys
        Recap_PITokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=tokenizer_max_length_pi,
            padding_side="right",
            padding="max_length",
            task_key="task_neg",
            output_key_suffix="_neg",
            keep_base_keys=True,
        ),
        # Gemma-3 tokenization for Gemma3Encoder (single, no pos/neg)
        Recap_PITokenizerProcessorStep(
            tokenizer_name="google/gemma-3-270m",
            max_length=tokenizer_max_length_recap,
            padding_side="right",
            padding="max_length",
            task_key="task_critic",  # Use task-only prompt for critic
            output_key_suffix="_gemma3",  # -> OBS_LANGUAGE_TOKENS_gemma3
            keep_base_keys=True,
        ),
    ]
    
    # 4. Insert RECAP tokenizer steps right after the base tokenizer
    insert_idx = base_tokenizer_idx + 1
    for j, recap_step in enumerate(recap_tokenizer_steps):
        new_steps.insert(insert_idx + j, recap_step)
    print(f"[RECAP Processor] Inserted {len(recap_tokenizer_steps)} RECAP tokenizer steps after index {base_tokenizer_idx}")
    
    # 5. Find DeviceProcessorStep and insert ReturnNormalizerProcessorStep before it
    device_step_idx = None
    for i, step in enumerate(new_steps):
        if isinstance(step, DeviceProcessorStep):
            device_step_idx = i
            break
    
    if device_step_idx is not None:
        new_steps.insert(device_step_idx, ReturnNormalizerProcessorStep(return_stats=return_stats))
        print(f"[RECAP Processor] Inserted ReturnNormalizerProcessorStep before DeviceProcessorStep at index {device_step_idx}")
    else:
        # If no device step found, append at the end
        new_steps.append(ReturnNormalizerProcessorStep(return_stats=return_stats))
        print("[RECAP Processor] No DeviceProcessorStep found, appended ReturnNormalizerProcessorStep at end")
    
    # 6. Rebuild preprocessor with new steps
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=new_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    
    print(f"[RECAP Processor] Final preprocessor has {len(preprocessor.steps)} steps:")
    for i, step in enumerate(preprocessor.steps):
        print(f"  [{i}] {type(step).__name__}")
    
    return preprocessor, postprocessor
