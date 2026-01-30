"""RECAP policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from lerobot_policy_recap.policies.recap.configuration_recap_pi import RECAP_PI_Config
from lerobot_policy_recap.policies.recap.modeling_recap_pi import RECAP_PI_Policy
from lerobot_policy_recap.policies.recap.processor_recap_pi import make_recap_pre_post_processors

__all__ = [
    "RECAP_PI_Config",
    "RECAP_PI_Policy",
    "make_recap_pre_post_processors",
]
