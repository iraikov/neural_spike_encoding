"""
SpikeEncoder: A flexible library for neural encoding that transforms continuous signals into spike trains.

This package provides various methods for encoding continuous signals into spike trains,
including rate-based encoding and receptive field approaches.
"""

# Version info
__version__ = "0.1.0"

# Import main classes from submodules
from .base import SpikeEncoder, EncoderTimeConfig
from .rate_encoders import RateEncoder, LinearRateEncoder, ReceptiveFieldEncoder
from .spike_generators import PoissonSpikeGenerator
from .pipeline import EncodingPipeline

from .temporal_encoders import (
    TemporalEncoder,
    BurstEncoder,
    LatencyEncoder,
    RankOrderEncoder,
    create_temporal_encoder,
)

# Import convenience functions
from .pipeline import (
    encoder_generator,
    linear_rate_encoder,
    receptive_field_encoder,
    poisson_spike_generator,
    linear_to_poisson_encoder,
    receptive_field_to_poisson_encoder,
)

# Define what gets imported with "from spike_encoder import *"
__all__ = [
    # Base classes
    "SpikeEncoder",
    "EncoderTimeConfig",
    # Encoder classes
    "RateEncoder",
    "LinearRateEncoder",
    "ReceptiveFieldEncoder",
    "PoissonSpikeGenerator",
    "EncodingPipeline",
    "TemporalEncoder",
    "BurstEncoder",
    "LatencyEncoder",
    "RankOrderEncoder",
    # Convenience functions
    "encoder_generator",
    "linear_rate_encoder",
    "receptive_field_encoder",
    "poisson_spike_generator",
    "linear_to_poisson_encoder",
    "receptive_field_to_poisson_encoder",
    "create_temporal_encoder",
]
