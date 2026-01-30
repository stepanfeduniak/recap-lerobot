

import torch
import numpy as np
from collections import deque
from typing import Any, Dict, Mapping, Tuple


def _flatten_envs(envs: dict) -> list[tuple[str, int, object]]:
    tasks = []
    for suite, group in envs.items():
        for task_id, vec_env in group.items():
            tasks.append((suite, task_id, vec_env))
    return tasks

def _extract_success_from_info(info: dict, env_index: int = 0) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            is_success = final_info.get("is_success", None)
            if is_success is not None:
                try:
                    return bool(is_success[env_index]) if hasattr(is_success, "__getitem__") else bool(is_success)
                except Exception:
                    pass

    if "is_success" in info:
        try:
            is_success = info["is_success"]
            return bool(is_success[env_index]) if hasattr(is_success, "__getitem__") else bool(is_success)
        except Exception:
            pass

    return False

def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _scalar_np(x, dtype):
    # Always returns shape (1,) np.ndarray
    return np.asarray([x], dtype=dtype)
