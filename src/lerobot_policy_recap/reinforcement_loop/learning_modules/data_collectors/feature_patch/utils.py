#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import contextlib
import importlib.resources
import json
import logging
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from pprint import pformat
from typing import Any, Generic, TypeVar

import datasets
import numpy as np
import packaging.version
import pandas
import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.backward_compatibility import (
    FUTURE_MESSAGE,
    BackwardCompatibilityError,
    ForwardCompatibilityError,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from lerobot.utils.utils import SuppressProgressBars, is_valid_numpy_dtype_string
from lerobot.datasets.utils import _validate_feature_names
DEFAULT_CHUNK_SIZE = 1000  # Max number of files per chunk
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # Max size per file
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200  # Max size per file

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

T = TypeVar("T")

def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Convert hardware-specific features to a LeRobot dataset feature dictionary.

    This function takes a dictionary describing hardware outputs (like joint states
    or camera image shapes) and formats it into the standard LeRobot feature
    specification.

    Args:
        hw_features (dict): Dictionary mapping feature names to their type (float for
            joints) or shape (tuple for images).
        prefix (str): The prefix to add to the feature keys (e.g., "observation"
            or "action").
        use_video (bool): If True, image features are marked as "video", otherwise "image".

    Returns:
        dict: A LeRobot features dictionary.
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}
    if len(cam_fts) == 0:
        print("[hw_to_dataset_features] Using temp fix for camera lookup")
        cam_fts = {key: feat for key, feat in hw_features.items() if "image" in key}
        for key, feat in cam_fts.items():
            features[f"{prefix}.images.{key}"] = {
                "dtype": "video" if use_video else "image",
                "shape": feat.shape,
                "names": ["height", "width", "channels"],
            }
    else:
        for key, shape in cam_fts.items():
            features[f"{prefix}.images.{key}"] = {
                "dtype": "video" if use_video else "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
    if joint_fts and prefix == ACTION:
        joint_shape=sum(feat.shape[0] for key, feat in joint_fts.items())
        features[prefix] = {
            "dtype": "float32",
            "shape": (joint_shape,),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        joint_shape=sum(feat.shape[0] for key, feat in joint_fts.items())
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (joint_shape,),
            "names": list(joint_fts),
        }
    
    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Construct a single data frame from raw values based on dataset features.

    A "frame" is a dictionary containing all the data for a single timestep,
    formatted as numpy arrays according to the feature specification.

    Args:
        ds_features (dict): The LeRobot dataset features dictionary.
        values (dict): A dictionary of raw values from the hardware/environment.
        prefix (str): The prefix to filter features by (e.g., "observation"
            or "action").

    Returns:
        dict: A dictionary representing a single frame of data.
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame




def combine_feature_dicts(*dicts: dict) -> dict:
    """Merge LeRobot grouped feature dicts.

    - For 1D numeric specs (dtype not image/video/string) with "names": we merge the names and recompute the shape.
    - For others (e.g. `observation.images.*`), the last one wins (if they are identical).

    Args:
        *dicts: A variable number of LeRobot feature dictionaries to merge.

    Returns:
        dict: A single merged feature dictionary.

    Raises:
        ValueError: If there's a dtype mismatch for a feature being merged.
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # Initialize or retrieve the accumulating dict for this feature key
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # Ensure consistent data types across merged entries
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")
                # Merge feature names: append only new ones to preserve order without duplicates
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # Recompute the shape to reflect the updated number of features
                target["shape"] = (value["shape"][0],)
            else:
                # For images/videos and non-1D entries: override with the latest definition
                out[key] = value
    return out


