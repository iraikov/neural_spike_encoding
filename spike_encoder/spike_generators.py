import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator, Union, List
from spike_encoder.base import EncoderTimeConfig, SpikeEncoder


class PoissonSpikeGenerator(SpikeEncoder):
    """Generates Poisson spike trains from firing rates."""

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        random_seed: Optional[Union[int, np.random.RandomState]] = None,
    ):
        """
        Initialize the Poisson spike generator.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(time_config=time_config, duration_ms=duration_ms, dt_ms=dt_ms)

        # Set up random number generator
        if random_seed is not None:
            if isinstance(random_seed, np.random.RandomState):
                self.rng = random_seed
            else:
                self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random

    def encode(
        self,
        rates_hz: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[Union[ndarray, List[List[ndarray]]], Optional[float]]:
        """
        Generate Poisson spike trains from firing rates.

        Args:
            rates_hz: Firing rates in Hz, shape [n_samples, n_neurons].
            start_time_ms: Start time for encoding in milliseconds.
            return_times: If True, return spike times instead of binary spike array.

        Returns:
            Tuple of (spike array, next start time in milliseconds).
            If return_times=False: spike array has shape [n_samples, num_steps, n_neurons]
            If return_times=True: list of lists of spike time arrays in milliseconds
        """
        self._validate_input(rates_hz)

        n_samples, n_neurons = rates_hz.shape

        # Initialize spike array
        spike_array = np.zeros(
            (n_samples, self.time_config.num_steps, n_neurons), dtype=bool
        )

        if start_time_ms is None:
            start_time_ms = 0.0

        # Generate Poisson spike trains
        for i in range(n_samples):
            for j in range(n_neurons):
                rate_hz = rates_hz[i, j]
                if rate_hz > 0:
                    # Probability of spike per time step
                    p_spike = rate_hz * (self.time_config.dt_ms / 1000.0)

                    # Ensure probability is valid
                    p_spike = min(p_spike, 1.0)

                    # Generate random values
                    rand_vals = self.rng.random(self.time_config.num_steps)

                    # Set spikes where random value is less than spike probability
                    spike_array[i, rand_vals < p_spike, j] = 1

        # Calculate next start time
        next_time_ms = None
        if start_time_ms is not None:
            next_time_ms = start_time_ms + self.time_config.duration_ms

        if not return_times:
            return spike_array, next_time_ms
        else:
            # Convert binary spike array to spike times in milliseconds
            spike_times = list([[] for _ in range(n_neurons)])
            for i in range(n_samples):
                for j in range(n_neurons):
                    neuron_times = self.time_config.steps_to_ms(
                        np.where(spike_array[i, :, j])[0]
                    )
                    if start_time_ms is not None:
                        neuron_times += start_time_ms
                    if len(neuron_times) > 0:
                        spike_times[j].append(neuron_times)
                start_time_ms += self.time_config.dt_ms
            return spike_times, next_time_ms


def poisson_spike_generator(
    rates_hz: Union[ndarray, Iterable[ndarray]],
    time_config: Optional[EncoderTimeConfig] = None,
    duration_ms: float = 100.0,
    dt_ms: float = 1.0,
    start_time_ms: float = 0.0,
    return_times: bool = False,
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[Union[ndarray, List[List[ndarray]]], float]]:
    """
    Generate Poisson spike trains from firing rates.

    Args:
        rates_hz: Firing rates or iterable of firing rates.
        time_config: Time configuration object.
        duration_ms: Duration of encoding window in milliseconds.
        dt_ms: Time step size in milliseconds.
        start_time_ms: Start time for encoding in milliseconds.
        return_times: If True, return spike times instead of binary array.
        random_seed: Random seed for reproducibility.

    Yields:
        Tuple of (spike train, current time).
    """
    generator = PoissonSpikeGenerator(
        time_config=time_config,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        random_seed=random_seed,
    )

    # Create a custom generator to handle the return_times parameter
    current_time_ms = start_time_ms

    # If rates is not an iterable, convert it to one
    if isinstance(rates_hz, ndarray):
        rates_iter = [rates_hz]
    else:
        rates_iter = rates_hz

    for chunk in rates_iter:
        # Ensure the chunk has the right shape
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)

        # Encode the chunk - using generator, not undefined encoder
        output, next_time_ms = generator.encode(
            chunk, start_time_ms=current_time_ms, return_times=return_times
        )

        yield output, current_time_ms

        if next_time_ms is not None:
            current_time_ms = next_time_ms
