# Apply lerobot.processor.converters patch to add return_to_go support
from . import converters_patch  # noqa: F401 - imported for side effects

from .return_normalizer import ReturnNormalizerProcessorStep, ReturnUnnormalizerProcessorStep

__all__ = [
    "ReturnNormalizerProcessorStep",
    "ReturnUnnormalizerProcessorStep",
]
