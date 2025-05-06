import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator, Union
import warnings

# Import base classes and time configuration
from spike_encoder.base import EncoderTimeConfig, SpikeEncoder
from spike_encoder.pipeline import encoder_generator


class TemporalEncoder(SpikeEncoder):
    """Base class for temporal encoding methods with unified time handling."""

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
    ):
        """
        Initialize the temporal encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)


class PhaseEncoder(TemporalEncoder):
    """
    Encodes signals as phase shifts relative to a reference oscillation.

    This encoder models how some sensory systems encode information by the
    phase of spiking relative to a background oscillation (like theta rhythm).
    """

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        base_freq_hz: float = 5.0,  # Hz
        neurons_per_dim: int = 1,
        max_phase_shift_rad: float = np.pi,  # radians
    ):
        """
        Initialize the phase encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            base_freq_hz: Base oscillation frequency in Hz.
            neurons_per_dim: Number of neurons per input dimension.
            max_phase_shift_rad: Maximum phase shift in radians.
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)
        self.input_range = input_range
        self.base_freq_hz = base_freq_hz
        self.neurons_per_dim = neurons_per_dim
        self.max_phase_shift_rad = max_phase_shift_rad

        # Calculate the period of the base oscillation in milliseconds
        self.base_period_ms = 1000.0 / base_freq_hz

        # Calculate the period in time steps
        self.base_period_steps = self.time_config.ms_to_steps(self.base_period_ms)

        # Ensure the time window is large enough for at least one cycle
        if self.time_config.num_steps < self.base_period_steps:
            warnings.warn(
                f"Time window ({self.time_config.num_steps} steps) is smaller than one oscillation period "
                f"({self.base_period_steps} steps). Increase duration_ms for proper encoding."
            )

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode signal using phase encoding.

        Args:
            signal: Input signal of shape [n_samples, n_features].
            start_time_ms: Start time for encoding in milliseconds.

        Returns:
            Tuple of (spike array, next start time in milliseconds).
            Spike array has shape [n_samples, num_steps, n_features * neurons_per_dim].
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        n_neurons = n_features * self.neurons_per_dim

        # Initialize spike array
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        # Create time vector in seconds (for phase calculation)
        time_vec_sec = self.time_config.get_time_vector_sec()

        # Encode each feature
        for i in range(n_samples):
            for j in range(n_features):
                # Normalize input to [0, 1]
                normalized_input = (signal[i, j] - self.input_range[0]) / (
                    self.input_range[1] - self.input_range[0]
                )
                normalized_input = np.clip(normalized_input, 0, 1)

                # Calculate phase shift based on input value
                phase_shift_rad = normalized_input * self.max_phase_shift_rad

                # For each neuron assigned to this feature
                for k in range(self.neurons_per_dim):
                    # Calculate neuron index
                    neuron_idx = j * self.neurons_per_dim + k

                    # Generate spikes based on phase-shifted sine wave
                    # Threshold crossing of a sine wave creates spikes at specific phases
                    phase_rad = (
                        2 * np.pi * self.base_freq_hz * time_vec_sec - phase_shift_rad
                    )
                    spike_times = np.where(np.sin(phase_rad) >= 0.99)[0]

                    # Set spikes
                    if len(spike_times) > 0:
                        spike_array[i, spike_times, neuron_idx] = 1

        # Calculate next start time
        next_time_ms = None
        if start_time_ms is not None:
            next_time_ms = start_time_ms + self.time_config.duration_ms

        if not return_times:
            return spike_array, next_time_ms
        else:
            # Convert binary spike array to spike times in milliseconds
            spike_times = [[] * n_neurons]
            for i in range(n_samples):
                for j in range(n_neurons):
                    neuron_times = self.time_config.steps_to_ms(
                        np.where(spike_array[i, :, j])[0]
                    )
                    if start_time_ms is not None:
                        neuron_times += start_time_ms
                    spike_times[j].append(neuron_times)
            return spike_times, next_time_ms


class LatencyEncoder(TemporalEncoder):
    """
    Encodes signals as the latency (delay) from stimulus onset to first spike.

    This encoder models how some sensory systems encode information by the
    timing of the first spike after a stimulus onset.
    """

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        neurons_per_dim: int = 1,
        min_latency_ms: float = 5.0,
        max_latency_ms: float = 50.0,
        baseline_firing: bool = False,
        jitter_ms: float = 0.0,
    ):
        """
        Initialize the latency encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            neurons_per_dim: Number of neurons per input dimension.
            min_latency_ms: Minimum latency in milliseconds.
            max_latency_ms: Maximum latency in milliseconds.
            baseline_firing: Whether to include baseline firing.
            jitter_ms: Noise in spike timing (standard deviation in milliseconds).
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)
        self.input_range = input_range
        self.neurons_per_dim = neurons_per_dim
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.baseline_firing = baseline_firing
        self.jitter_ms = jitter_ms

        # Convert latency limits to time steps for internal calculations
        self.min_latency_steps = self.time_config.ms_to_steps(min_latency_ms)
        self.max_latency_steps = self.time_config.ms_to_steps(max_latency_ms)
        self.jitter_steps = self.time_config.ms_to_steps(jitter_ms)

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode signal using latency encoding.

        Args:
            signal: Input signal of shape [n_samples, n_features].
            start_time_ms: Start time for encoding in milliseconds.

        Returns:
            Tuple of (spike array, next start time in milliseconds).
            Spike array has shape [n_samples, num_steps, n_features * neurons_per_dim].
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        n_neurons = n_features * self.neurons_per_dim

        # Initialize spike array
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        # Generate random jitter if needed
        if self.jitter_ms > 0:
            jitter_values = np.random.normal(
                0, self.jitter_steps, (n_samples, n_neurons)
            )
        else:
            jitter_values = np.zeros((n_samples, n_neurons))

        # Encode each feature
        for i in range(n_samples):
            for j in range(n_features):
                # Normalize input to [0, 1]
                normalized_input = (signal[i, j] - self.input_range[0]) / (
                    self.input_range[1] - self.input_range[0]
                )
                normalized_input = np.clip(normalized_input, 0, 1)

                # Higher values have shorter latency (stronger stimulus -> faster response)
                latency_steps = self.max_latency_steps - normalized_input * (
                    self.max_latency_steps - self.min_latency_steps
                )

                # For each neuron assigned to this feature
                for k in range(self.neurons_per_dim):
                    # Calculate neuron index
                    neuron_idx = j * self.neurons_per_dim + k

                    # Add jitter
                    spike_time = int(latency_steps + jitter_values[i, neuron_idx])
                    spike_time = max(0, min(self.time_config.num_steps - 1, spike_time))

                    # Set spike
                    spike_array[i, spike_time, neuron_idx] = 1

                    # Add baseline firing if enabled
                    if self.baseline_firing:
                        # Generate random baseline spikes at ~5Hz
                        baseline_rate_hz = 5.0  # Hz
                        p_spike = baseline_rate_hz * (self.time_config.dt_ms / 1000.0)

                        for t in range(self.time_config.num_steps):
                            if t != spike_time and np.random.random() < p_spike:
                                spike_array[i, t, neuron_idx] = 1

        # Calculate next start time
        next_time_ms = None
        if start_time_ms is not None:
            next_time_ms = start_time_ms + self.time_config.duration_ms

        if not return_times:
            return spike_array, next_time_ms
        else:
            # Convert binary spike array to spike times in milliseconds
            spike_times = []
            for i in range(n_samples):
                for j in range(n_neurons):
                    neuron_times = self.time_config.steps_to_ms(
                        np.where(spike_array[i, :, j])[0]
                    )
                    if start_time_ms is not None:
                        neuron_times += start_time_ms
                    spike_times[j].append(neuron_times)
            return spike_times, next_time_ms


class RankOrderEncoder(TemporalEncoder):
    """
    Encodes signals as the relative order of first spikes across neurons.

    This encoder is inspired by the rank-order coding hypothesis in the visual system,
    where the order of neural firing carries information.
    """

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        neurons_per_dim: int = 5,
        base_latency_ms: float = 5.0,
        latency_range_ms: float = 30.0,
        max_spikes: int = 5,
    ):
        """
        Initialize the rank order encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            neurons_per_dim: Number of neurons per input dimension.
            base_latency_ms: Base latency for all spikes in milliseconds.
            latency_range_ms: Range of latencies in milliseconds.
            max_spikes: Maximum number of spikes per neuron
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)
        self.input_range = input_range
        self.neurons_per_dim = neurons_per_dim
        self.base_latency_ms = base_latency_ms
        self.latency_range_ms = latency_range_ms
        self.max_spikes = max_spikes

        # Convert latency to time steps for internal calculations
        self.base_latency_steps = self.time_config.ms_to_steps(base_latency_ms)
        self.latency_range_steps = self.time_config.ms_to_steps(latency_range_ms)

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode signal using rank order encoding.

        Args:
            signal: Input signal of shape [n_samples, n_features].
            start_time_ms: Start time for encoding in milliseconds.

        Returns:
            Tuple of (spike array, next start time in milliseconds).
            Spike array has shape [n_samples, num_steps, n_features * neurons_per_dim].
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        n_neurons = n_features * self.neurons_per_dim

        # Initialize spike array
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        # Encode each sample
        for i in range(n_samples):
            # Reshape signal and calculate neuron preferences
            preferences = np.zeros((n_features, self.neurons_per_dim))

            # Assign preferences evenly across the input range for each feature
            for j in range(n_features):
                preferences[j] = np.linspace(
                    self.input_range[0], self.input_range[1], self.neurons_per_dim
                )

            # Calculate activation based on distance between signal and preferences
            activations = np.zeros((n_features, self.neurons_per_dim))
            for j in range(n_features):
                # Compute activation as negative distance (closer = higher activation)
                distances = -np.abs(signal[i, j] - preferences[j])
                # Normalize to [0, 1] for this feature
                if np.max(distances) != np.min(distances):
                    activations[j] = (distances - np.min(distances)) / (
                        np.max(distances) - np.min(distances)
                    )
                else:
                    activations[j] = np.ones_like(distances)

            # Flatten activations
            activations = activations.flatten()

            # Order neurons by activation (highest first)
            neuron_order = np.argsort(-activations)

            # Assign spike times based on rank order
            for rank, neuron_idx in enumerate(neuron_order):
                # Calculate delay based on rank
                rank_ratio = rank / len(neuron_order)
                spike_time_steps = self.base_latency_steps + int(
                    rank_ratio * self.latency_range_steps
                )

                # Ensure spike time is within window
                if spike_time_steps < self.time_config.num_steps:
                    spike_array[i, spike_time_steps, neuron_idx] = 1

                    # Add additional spikes if max_spikes > 1
                    if self.max_spikes > 1:
                        # Higher activations get more spikes
                        activation = activations[neuron_idx]
                        n_additional_spikes = int(
                            (self.max_spikes - 1) * activation
                        )  # Up to max_spikes-1 additional spikes

                        for s in range(1, n_additional_spikes + 1):
                            # Spikes occur at intervals of 5ms
                            interval_steps = self.time_config.ms_to_steps(5.0)
                            additional_time = spike_time_steps + s * interval_steps
                            if additional_time < self.time_config.num_steps:
                                spike_array[i, additional_time, neuron_idx] = 1

        # Calculate next start time
        next_time_ms = None
        if start_time_ms is not None:
            next_time_ms = start_time_ms + self.time_config.duration_ms

        if not return_times:
            return spike_array, next_time_ms
        else:
            # Convert binary spike array to spike times in milliseconds
            spike_times = [[] * n_neurons]
            for i in range(n_samples):
                for j in range(n_neurons):
                    neuron_times = self.time_config.steps_to_ms(
                        np.where(spike_array[i, :, j])[0]
                    )
                    if start_time_ms is not None:
                        neuron_times += start_time_ms
                    spike_times[j].append(neuron_times)
        return spike_array, next_time_ms


class BurstEncoder(SpikeEncoder):
    """
    Encodes signals as the number of spikes in a burst.

    This encoder models how some neurons encode information in the number
    of spikes within a burst, which can transmit more information than
    single spikes.
    """

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        neurons_per_dim: int = 1,
        min_spikes: int = 1,
        max_spikes: int = 5,
        burst_window_ms: float = 20.0,  # Duration of burst window in milliseconds
        inter_spike_interval_ms: float = 3.0,  # Interval between spikes in milliseconds
    ):
        """
        Initialize the burst encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            neurons_per_dim: Number of neurons per input dimension.
            min_spikes: Minimum number of spikes in a burst.
            max_spikes: Maximum number of spikes in a burst.
            burst_window_ms: Duration allocated to each burst in milliseconds.
            inter_spike_interval_ms: Interval between spikes in a burst in milliseconds.
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)

        self.input_range = input_range
        self.neurons_per_dim = neurons_per_dim
        self.min_spikes = min_spikes
        self.max_spikes = max_spikes
        self.burst_window_ms = burst_window_ms
        self.inter_spike_interval_ms = inter_spike_interval_ms

        # Convert time values to steps for internal use
        self.burst_window_steps = self.time_config.ms_to_steps(burst_window_ms)
        self.inter_spike_interval_steps = self.time_config.ms_to_steps(
            inter_spike_interval_ms
        )

        # Check if burst window is large enough
        max_burst_length_ms = (max_spikes - 1) * inter_spike_interval_ms + dt_ms
        if burst_window_ms < max_burst_length_ms:
            warnings.warn(
                f"Burst window ({burst_window_ms} ms) is smaller than maximum burst length "
                f"({max_burst_length_ms} ms). Increase burst_window_ms for proper encoding."
            )

        # Validate parameters
        if input_range[1] <= input_range[0]:
            raise ValueError(
                f"Input range maximum must be greater than minimum, got {input_range}"
            )
        if min_spikes > max_spikes:
            raise ValueError(
                f"Minimum spikes ({min_spikes}) cannot be greater than maximum spikes ({max_spikes})"
            )
        if neurons_per_dim <= 0:
            raise ValueError(
                f"Number of neurons per dimension must be positive, got {neurons_per_dim}"
            )

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode signal using burst encoding.

        Args:
            signal: Input signal of shape [n_samples, n_features].
            start_time_ms: Start time for encoding in milliseconds.

        Returns:
            Tuple of (spike array, next start time in milliseconds).
            Spike array has shape [n_samples, num_steps, n_features * neurons_per_dim].
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        n_neurons = n_features * self.neurons_per_dim

        # Initialize spike array
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        # Encode each feature
        for i in range(n_samples):
            for j in range(n_features):
                # Normalize input to [0, 1]
                normalized_input = (signal[i, j] - self.input_range[0]) / (
                    self.input_range[1] - self.input_range[0]
                )
                normalized_input = np.clip(normalized_input, 0, 1)

                # Map normalized input to number of spikes in burst
                n_spikes = self.min_spikes + int(
                    normalized_input * (self.max_spikes - self.min_spikes)
                )

                # For each neuron assigned to this feature
                for k in range(self.neurons_per_dim):
                    # Calculate neuron index
                    neuron_idx = j * self.neurons_per_dim + k

                    # Generate burst at the start of the window
                    for s in range(n_spikes):
                        spike_time_steps = s * self.inter_spike_interval_steps
                        if spike_time_steps < self.time_config.num_steps:
                            spike_array[i, spike_time_steps, neuron_idx] = 1

        # Calculate next start time
        next_time_ms = None
        if start_time_ms is not None:
            next_time_ms = start_time_ms + self.time_config.duration_ms

        if not return_times:
            return spike_array, next_time_ms
        else:
            # Convert binary spike array to spike times in milliseconds
            spike_times = [[] * n_neurons]
            for i in range(n_samples):
                for j in range(n_neurons):
                    neuron_times = self.time_config.steps_to_ms(
                        np.where(spike_array[i, :, j])[0]
                    )
                    if start_time_ms is not None:
                        neuron_times += start_time_ms
                    spike_times[j].append(neuron_times)
        return spike_array, next_time_ms


def burst_encoder(
    signal: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    input_range: Tuple[float, float] = (0, 1),
    neurons_per_dim: int = 1,
    min_spikes: int = 1,
    max_spikes: int = 5,
    burst_window_ms: float = 20.0,
    inter_spike_interval_ms: float = 3.0,
    start_time_ms: float = 0.0,
) -> Iterator[Tuple[ndarray, float]]:
    """
    Encode a signal using burst encoding.

    Args:
        signal: Input signal or iterable of signals.
        time_config: Time configuration object.
        duration_ms: Duration of encoding window in milliseconds.
        dt_ms: Time step size in milliseconds.
        input_range: Range of input values (min, max).
        neurons_per_dim: Number of neurons per input dimension.
        min_spikes: Minimum number of spikes in a burst.
        max_spikes: Maximum number of spikes in a burst.
        burst_window_ms: Duration allocated to each burst in milliseconds.
        inter_spike_interval_ms: Interval between spikes in a burst in milliseconds.
        start_time_ms: Start time for encoding in milliseconds.

    Yields:
        Tuple of (encoded signal, current time in milliseconds).
    """

    encoder = BurstEncoder(
        time_config=time_config,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        input_range=input_range,
        neurons_per_dim=neurons_per_dim,
        min_spikes=min_spikes,
        max_spikes=max_spikes,
        burst_window_ms=burst_window_ms,
        inter_spike_interval_ms=inter_spike_interval_ms,
    )

    yield from encoder_generator(signal, encoder, start_time_ms)


# Convenience factory functions
def create_temporal_encoder(
    encoder_type: str,
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    **kwargs,
) -> TemporalEncoder:
    """
    Create a temporal encoder based on type.

    Args:
        encoder_type: Type of encoder ("phase", "latency", or "rank_order").
        time_config: Time configuration object.
        duration_ms: Duration of encoding window in milliseconds.
        dt_ms: Time step size in milliseconds.
        **kwargs: Additional parameters for specific encoder types.

    Returns:
        A temporal encoder instance.
    """
    if encoder_type == "phase":
        return PhaseEncoder(
            time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms, **kwargs
        )
    elif encoder_type == "latency":
        return LatencyEncoder(
            time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms, **kwargs
        )
    elif encoder_type == "rank_order":
        return RankOrderEncoder(
            time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms, **kwargs
        )
    elif encoder_type == "burst":
        return BurstEncoder(
            time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
