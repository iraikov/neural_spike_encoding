import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Union, List, Callable
from spike_encoder.base import EncoderTimeConfig, SpikeEncoder


class RateEncoder(SpikeEncoder):
    """Base class for rate-based encoding methods."""

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        max_firing_rate_hz: float = 100.0,
        min_firing_rate_hz: float = 0.0,
        neurons_per_dim: int = 1,
    ):
        """
        Initialize the rate encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            max_firing_rate_hz: Maximum firing rate in Hz.
            min_firing_rate_hz: Minimum firing rate in Hz.
            neurons_per_dim: Number of neurons per input dimension.
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)

        self.input_range = input_range
        self.max_firing_rate_hz = max_firing_rate_hz
        self.min_firing_rate_hz = min_firing_rate_hz
        self.neurons_per_dim = neurons_per_dim

        # Validate parameters
        if input_range[1] <= input_range[0]:
            raise ValueError(
                f"Input range maximum must be greater than minimum, got {input_range}"
            )
        if max_firing_rate_hz < min_firing_rate_hz:
            raise ValueError(
                f"Maximum firing rate must be greater than minimum, got {max_firing_rate_hz} and {min_firing_rate_hz}"
            )
        if neurons_per_dim <= 0:
            raise ValueError(
                f"Number of neurons per dimension must be positive, got {neurons_per_dim}"
            )


class LinearRateEncoder(RateEncoder):
    """Encodes signals into firing rates using linear mapping."""

    def compute_rates(self, signal: ndarray) -> ndarray:
        """
        Compute firing rates from input signal.

        Args:
            signal: Input signal of shape [n_samples, n_features].

        Returns:
            Firing rates of shape [n_samples, n_features * neurons_per_dim] in Hz.
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        rates = np.zeros((n_samples, n_features * self.neurons_per_dim))

        for i in range(n_features):
            # Linear interpolation from input range to firing rate range
            feature_rates = np.interp(
                signal[:, i],
                self.input_range,
                (self.min_firing_rate_hz, self.max_firing_rate_hz),
            )

            # If multiple neurons per dimension, duplicate the rates
            for j in range(self.neurons_per_dim):
                rates[:, i * self.neurons_per_dim + j] = feature_rates

        return rates

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode signal into a rate-based spike train.

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

        # Compute firing rates in Hz
        rates_hz = self.compute_rates(signal)

        # Convert to binary spike trains
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        # For each sample and neuron
        for i in range(n_samples):
            for j in range(n_neurons):
                rate_hz = rates_hz[i, j]
                if rate_hz > 0:
                    # Calculate inter-spike interval in time steps
                    isi_ms = (
                        1000.0 / rate_hz
                    )  # Convert Hz to inter-spike interval in ms
                    isi_steps = self.time_config.ms_to_steps(isi_ms)

                    if isi_steps > -1:
                        # Ensure minimum interval of 1 time step
                        isi_steps = max(1, isi_steps)

                        # Generate regular spike train
                        indices = np.arange(0, self.time_config.num_steps, isi_steps)
                        spike_array[i, indices, j] = 1

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


class ReceptiveFieldEncoder(RateEncoder):
    """Encodes signals using receptive fields (tuning curves)."""

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_range: Tuple[float, float] = (0, 1),
        max_firing_rate_hz: float = 100.0,
        min_firing_rate_hz: float = 0.0,
        neurons_per_dim: int = 5,
        tuning_width: float = 0.2,  # Width relative to input range
        tuning_function: Union[
            str, Callable
        ] = "gaussian",  # "gaussian", "cosine", or callable
        num_centers_per_neuron: int = 1,  # Number of receptive field centers per neuron
        response_combination: Union[
            str, Callable
        ] = "max",  # How to combine responses from multiple centers: "max", "sum", or callable
    ):
        """
        Initialize the receptive field encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            input_range: Range of input values (min, max).
            max_firing_rate_hz: Maximum firing rate in Hz.
            min_firing_rate_hz: Minimum firing rate in Hz.
            neurons_per_dim: Number of neurons per input dimension.
            tuning_width: Width of the tuning curve relative to input range.
            tuning_function: Type of tuning function ("gaussian" or "cosine").
            num_centers_per_neuron: Number of receptive field centers per neuron.
            response_combination: How to combine responses from multiple centers ("max" or "sum").
        """
        super().__init__(
            time_config=time_config,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            input_range=input_range,
            max_firing_rate_hz=max_firing_rate_hz,
            min_firing_rate_hz=min_firing_rate_hz,
            neurons_per_dim=neurons_per_dim,
        )

        self.tuning_width = tuning_width
        self.tuning_function = tuning_function
        self.num_centers_per_neuron = num_centers_per_neuron
        self.response_combination = response_combination

        # Calculate actual tuning width in input units
        input_range_size = input_range[1] - input_range[0]
        self.tuning_width_units = tuning_width * input_range_size

        # Create centers for receptive fields
        self.centers = self._create_centers()

    def _create_centers(self) -> List[List[ndarray]]:
        """Create receptive field centers for each feature dimension and neuron."""
        centers = []
        for _ in range(1):  # We don't know n_features yet, so create a template
            dim_neurons = []
            for n in range(self.neurons_per_dim):
                if self.num_centers_per_neuron == 1:
                    # Default behavior: one center per neuron
                    neuron_center = (
                        self.input_range[0]
                        + (self.input_range[1] - self.input_range[0])
                        * (n + 0.5)
                        / self.neurons_per_dim
                    )
                    dim_neurons.append(np.array([neuron_center]))
                else:
                    # Multiple centers per neuron, evenly spaced across the input range
                    neuron_centers = np.linspace(
                        self.input_range[0],
                        self.input_range[1],
                        self.num_centers_per_neuron,
                    )
                    dim_neurons.append(neuron_centers)
            centers.append(dim_neurons)
        return centers

    def _gaussian_tuning(self, x: float, center: float) -> float:
        """Compute Gaussian tuning curve value."""
        return np.exp(-0.5 * ((x - center) / self.tuning_width_units) ** 2)

    def _cosine_tuning(self, x: float, center: float) -> float:
        """Compute cosine tuning curve value."""
        # Scale input to [0, pi]
        input_range_size = self.input_range[1] - self.input_range[0]
        scaled_dist = np.minimum(np.abs(x - center) / input_range_size, 1.0) * np.pi
        return (np.cos(scaled_dist) + 1) / 2

    def compute_rates(self, signal: ndarray) -> ndarray:
        """
        Compute firing rates from input signal using receptive fields.

        Args:
            signal: Input signal of shape [n_samples, n_features].

        Returns:
            Firing rates of shape [n_samples, n_features * neurons_per_dim] in Hz.
        """
        self._validate_input(signal)

        n_samples, n_features = signal.shape
        n_neurons = n_features * self.neurons_per_dim
        rates = np.zeros((n_samples, n_neurons))

        # Ensure we have centers for all features
        while len(self.centers) < n_features:
            dim_neurons = []
            for n in range(self.neurons_per_dim):
                if self.num_centers_per_neuron == 1:
                    # Default behavior: one center per neuron
                    neuron_center = (
                        self.input_range[0]
                        + (self.input_range[1] - self.input_range[0])
                        * (n + 0.5)
                        / self.neurons_per_dim
                    )
                    dim_neurons.append(np.array([neuron_center]))
                else:
                    # Multiple centers per neuron, evenly spaced across the input range
                    neuron_centers = np.linspace(
                        self.input_range[0],
                        self.input_range[1],
                        self.num_centers_per_neuron,
                    )
                    dim_neurons.append(neuron_centers)
            self.centers.append(dim_neurons)

        # Select tuning function
        if self.tuning_function == "gaussian":
            tuning_func = self._gaussian_tuning
        elif self.tuning_function == "cosine":
            tuning_func = self._cosine_tuning
        elif isinstance(self.tuning_function, Callable):
            tuning_func = self.tuning_function
        else:
            raise ValueError(f"Unknown tuning function: {self.tuning_function}")

        # Compute response for each neuron
        for i in range(n_samples):
            for f in range(n_features):
                feature_val = signal[i, f]
                for n in range(self.neurons_per_dim):
                    neuron_idx = f * self.neurons_per_dim + n
                    centers = self.centers[f][n]

                    # Compute response for each center
                    responses = np.array(
                        [tuning_func(feature_val, center) for center in centers]
                    )

                    # Combine responses from multiple centers
                    if self.response_combination == "max":
                        combined_response = np.max(responses)
                    elif self.response_combination == "sum":
                        # Normalize sum to be in [0, 1] range
                        combined_response = np.sum(responses) / len(responses)
                    elif isinstance(self.response_combination, Callable):
                        self.response_combination(responses)
                    else:
                        raise ValueError(
                            f"Unknown response combination method: {self.response_combination}"
                        )

                    # Scale to firing rate range
                    rates[i, neuron_idx] = (
                        self.min_firing_rate_hz
                        + (self.max_firing_rate_hz - self.min_firing_rate_hz)
                        * combined_response
                    )

        return rates
