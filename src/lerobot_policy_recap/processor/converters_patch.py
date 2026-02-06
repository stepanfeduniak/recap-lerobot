#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monkey-patch for lerobot.processor.converters to add return_to_go support.

This patches the `_extract_complementary_data` function in the original lerobot
package to include the `return_to_go` field in complementary data.

Import this module early to apply the patch before any converters are used.
"""

from typing import Any


def _extract_complementary_data_patched(batch: dict[str, Any]) -> dict[str, Any]:
    """
    Extract complementary data from a batch dictionary.

    This includes padding flags, task description, indices, and return_to_go.

    Args:
        batch: The batch dictionary.

    Returns:
        A dictionary with the extracted complementary data.
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    return_to_go_key = {"return_to_go": batch["return_to_go"]} if "return_to_go" in batch else {}
    improvement_indicator_key = {"improvement_indicator": batch["improvement_indicator"]} if "improvement_indicator" in batch else {}

    return {**pad_keys, **task_key, **index_key, **task_index_key, **return_to_go_key, **improvement_indicator_key}


def apply_patch():
    """
    Apply the monkey-patch to lerobot.processor.converters.
    
    This replaces the original `_extract_complementary_data` function with our
    patched version that includes return_to_go support.
    """
    import lerobot.processor.converters as converters_module
    
    # Store original for reference if needed
    converters_module._extract_complementary_data_original = getattr(
        converters_module, '_extract_complementary_data', None
    )
    
    # Apply the patch
    converters_module._extract_complementary_data = _extract_complementary_data_patched


# Automatically apply patch when this module is imported
apply_patch()
