from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator, Union, List
from spike_encoder.base import EncoderTimeConfig, SpikeEncoder
from spike_encoder.rate_encoders import (
    RateEncoder,
    LinearRateEncoder,
    ReceptiveFieldEncoder,
)
from spike_encoder.spike_generators import PoissonSpikeGenerator


class EncodingPipeline:
    """Pipeline for multi-stage encoding of signals with specialized handling for rate-to-spike conversions."""

    def __init__(
        self,
        encoders: List[SpikeEncoder],
        time_config: Optional[EncoderTimeConfig] = None,
    ):
        """
        Initialize the encoding pipeline.

        Args:
            encoders: List of encoders to apply in sequence.
            time_config: Time configuration to use for all encoders.
                If provided, it overrides the time config of individual encoders.
        """
        if len(encoders) < 1:
            raise ValueError("Pipeline must contain at least one encoder")

        self.encoders = encoders

        # If time_config is provided, use it for all encoders
        if time_config is not None:
            for encoder in self.encoders:
                encoder.time_config = time_config
        else:
            time_config = self.encoders[0].time_config
        self.time_config = time_config

    def compute_rates(self, signal: ndarray):
        if isinstance(self.encoders[0], RateEncoder):
            rate_encoder = self.encoders[0]
            rates = rate_encoder.compute_rates(signal)
            return rates
        else:
            raise RuntimeError(
                "EncodingPipeline: rate encoder required for compute_rates"
            )

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[Union[ndarray, List[List[ndarray]]], Optional[float]]:
        """
        Encode a signal through the entire pipeline.

        Args:
            signal: Input signal.
            start_time_ms: Start time for encoding in milliseconds.
            return_times: Whether to return spike times (only applies when last encoder is PoissonSpikeGenerator).

        Returns:
            Tuple of (encoded signal, next start time in milliseconds).
        """
        # Special case for rate encoder followed by spike generator
        if (
            len(self.encoders) == 2
            and isinstance(self.encoders[0], RateEncoder)
            and isinstance(self.encoders[1], PoissonSpikeGenerator)
        ):
            rate_encoder = self.encoders[0]
            spike_generator = self.encoders[1]

            # Get firing rates from rate encoder
            rates = rate_encoder.compute_rates(signal)

            # Generate spikes using the rates
            spikes, next_time_ms = spike_generator.encode(
                rates, start_time_ms=start_time_ms, return_times=return_times
            )

            return spikes, next_time_ms

        # Default case: encode sequentially
        current_signal = signal
        current_time_ms = start_time_ms

        for i, encoder in enumerate(self.encoders):
            # For the last encoder that is a PoissonSpikeGenerator, pass return_times
            if i == len(self.encoders) - 1 and isinstance(
                encoder, PoissonSpikeGenerator
            ):
                current_signal, current_time_ms = encoder.encode(
                    current_signal,
                    start_time_ms=current_time_ms,
                    return_times=return_times,
                )
            else:
                current_signal, current_time_ms = encoder.encode(
                    current_signal, start_time_ms=current_time_ms
                )

        return current_signal, current_time_ms


def encoder_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    encoder: Union[SpikeEncoder, EncodingPipeline],
    start_time_ms: float = 0.0,
) -> Iterator[Tuple[ndarray, float]]:
    """
    Lazily encode a sequence of data using the provided encoder or pipeline.

    Args:
        signal: Input signal or iterable of signals.
        encoder: Encoder or pipeline to use for encoding.
        start_time_ms: Start time for encoding in milliseconds.

    Yields:
        Tuple of (encoded signal, current time in milliseconds).
    """
    current_time_ms = start_time_ms

    # If signal is not an iterable, convert it to one
    if isinstance(signal, ndarray):
        signal_iter = [signal]
    else:
        signal_iter = signal

    for chunk in signal_iter:
        # Ensure the chunk has the right shape
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)

        # Encode the chunk
        output, next_time_ms = encoder.encode(chunk, start_time_ms=current_time_ms)

        yield output, current_time_ms

        if next_time_ms is not None:
            current_time_ms = next_time_ms


# Convenience functions for common encoding patterns
def linear_rate_encoder(
    signal: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    input_range: Tuple[float, float] = (0, 1),
    max_firing_rate_hz: float = 100.0,
    min_firing_rate_hz: float = 0.0,
    neurons_per_dim: int = 1,
    t_start: float = 0.0,
) -> Iterator[Tuple[ndarray, float]]:
    """
    Encode a signal using linear rate encoding.

    Args:
        signal: Input signal or iterable of signals.
        neurons_per_dim: Number of neurons per input dimension.
        input_range: Range of input values.
        output_freq_range: Range of output frequencies.
        time_window: Length of encoding window.
        dt: Time step size.
        t_start: Start time for encoding.

    Yields:
        Tuple of (encoded signal, current time).
    """
    encoder = LinearRateEncoder(
        time_config=time_config,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        input_range=input_range,
        max_firing_rate_hz=max_firing_rate_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        neurons_per_dim=neurons_per_dim,
    )

    yield from encoder_generator(signal, encoder, t_start)


def receptive_field_encoder(
    signal: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    tuning_width: float = 0.1,
    tuning_function: str = "gaussian",
    input_range: Tuple[float, float] = (0, 1),
    max_firing_rate_hz: float = 100.0,
    min_firing_rate_hz: float = 0.0,
    neurons_per_dim: int = 1,
    t_start: float = 0.0,
) -> Iterator[Tuple[ndarray, float]]:
    """
    Encode a signal using linear rate encoding.

    Args:
        signal: Input signal or iterable of signals.
        neurons_per_dim: Number of neurons per input dimension.
        input_range: Range of input values.
        output_freq_range: Range of output frequencies.
        time_window: Length of encoding window.
        dt: Time step size.
        t_start: Start time for encoding.

    Yields:
        Tuple of (encoded signal, current time).
    """
    encoder = ReceptiveFieldEncoder(
        time_config=time_config,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        input_range=input_range,
        max_firing_rate_hz=max_firing_rate_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        neurons_per_dim=neurons_per_dim,
        tuning_width=tuning_width,
        tuning_function=tuning_function,
    )

    yield from encoder_generator(signal, encoder, t_start)


def linear_to_poisson_encoder(
    signal: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    input_range: Tuple[float, float] = (0, 1),
    max_firing_rate_hz: float = 100.0,
    min_firing_rate_hz: float = 0.0,
    neurons_per_dim: int = 1,
    start_time_ms: float = 0.0,
    return_times: bool = False,
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[Union[ndarray, List[List[ndarray]]], float]]:
    """
    Encode a signal using linear rate encoding followed by Poisson spike generation.

    Args:
        signal: Input signal or iterable of signals.
        time_config: Time configuration object.
        duration_ms: Duration of encoding window in milliseconds.
        dt_ms: Time step size in milliseconds.
        input_range: Range of input values.
        max_firing_rate_hz: Maximum firing rate in Hz.
        min_firing_rate_hz: Minimum firing rate in Hz.
        neurons_per_dim: Number of neurons per input dimension.
        start_time_ms: Start time for encoding in milliseconds.
        return_times: If True, return spike times instead of binary array.
        random_seed: Random seed for reproducibility.

    Yields:
        Tuple of (spike train, current time).
    """
    # Create shared time configuration
    if time_config is None:
        time_config = EncoderTimeConfig(duration_ms=duration_ms, dt_ms=dt_ms)

    # Create the pipeline
    pipeline = EncodingPipeline(
        [
            LinearRateEncoder(
                time_config=time_config,
                input_range=input_range,
                max_firing_rate_hz=max_firing_rate_hz,
                min_firing_rate_hz=min_firing_rate_hz,
                neurons_per_dim=neurons_per_dim,
            ),
            PoissonSpikeGenerator(time_config=time_config, random_seed=random_seed),
        ],
        time_config=time_config,
    )

    # Create a custom generator
    current_time_ms = start_time_ms

    # If signal is not an iterable, convert it to one
    if isinstance(signal, ndarray):
        signal_iter = [signal]
    else:
        signal_iter = signal

    for chunk in signal_iter:
        # Ensure the chunk has the right shape
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)

        # Encode the chunk
        output, next_time_ms = pipeline.encode(
            chunk,
            start_time_ms=current_time_ms,
            return_times=return_times,  # Pass return_times parameter
        )

        yield output, current_time_ms

        if next_time_ms is not None:
            current_time_ms = next_time_ms


def receptive_field_to_poisson_encoder(
    signal: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    input_range: Tuple[float, float] = (0, 1),
    max_firing_rate_hz: float = 100.0,
    min_firing_rate_hz: float = 0.0,
    neurons_per_dim: int = 1,
    tuning_width: float = 0.1,
    tuning_function: str = "gaussian",
    start_time_ms: float = 0.0,
    return_times: bool = False,
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[Union[ndarray, List[List[ndarray]]], float]]:
    """
    Encode a signal using receptive field encoding followed by Poisson spike generation.

    Args:
        signal: Input signal or iterable of signals.
        time_config: Time configuration object.
        duration_ms: Duration of encoding window in milliseconds.
        dt_ms: Time step size in milliseconds.
        input_range: Range of input values.
        max_firing_rate_hz: Maximum firing rate in Hz.
        min_firing_rate_hz: Minimum firing rate in Hz.
        neurons_per_dim: Number of neurons per input dimension.
        tuning_width: Width of receptive fields.
        tuning_function: Type of tuning function.
        start_time_ms: Start time for encoding in milliseconds.
        return_times: If True, return spike times instead of binary array.
        random_seed: Random seed for reproducibility.

    Yields:
        Tuple of (spike train, current time).
    """
    # Create shared time configuration
    if time_config is None:
        time_config = EncoderTimeConfig(duration_ms=duration_ms, dt_ms=dt_ms)

    # Create the pipeline
    pipeline = EncodingPipeline(
        [
            ReceptiveFieldEncoder(
                time_config=time_config,
                input_range=input_range,
                max_firing_rate_hz=max_firing_rate_hz,
                min_firing_rate_hz=min_firing_rate_hz,
                neurons_per_dim=neurons_per_dim,
                tuning_width=tuning_width,
                tuning_function=tuning_function,
            ),
            PoissonSpikeGenerator(time_config=time_config, random_seed=random_seed),
        ],
        time_config=time_config,
    )

    # Create a custom generator
    current_time_ms = start_time_ms

    # If signal is not an iterable, convert it to one
    if isinstance(signal, ndarray):
        signal_iter = [signal]
    else:
        signal_iter = signal

    for chunk in signal_iter:
        # Ensure the chunk has the right shape
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)

        # Encode the chunk
        output, next_time_ms = pipeline.encode(
            chunk,
            start_time_ms=current_time_ms,
            return_times=return_times,  # Pass return_times parameter
        )

        yield output, current_time_ms

        if next_time_ms is not None:
            current_time_ms = next_time_ms
