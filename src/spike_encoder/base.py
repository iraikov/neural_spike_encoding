import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Union


class EncoderTimeConfig:
    """
    Unified time configuration for spike encoders.

    This class standardizes time units and provides conversion methods between
    different time representations.
    """

    def __init__(
        self,
        duration_ms: float = 100.0,  # Duration in milliseconds
        dt_ms: float = 1.0,  # Time step size in milliseconds
    ):
        """
        Initialize time configuration.

        Args:
            duration_ms: Total encoding duration in milliseconds.
            dt_ms: Time step size in milliseconds.
        """
        self.duration_ms = duration_ms
        self.dt_ms = dt_ms

        # Calculate derived properties
        self.num_steps = int(duration_ms / dt_ms)
        self.dt_sec = dt_ms / 1000.0
        self.duration_sec = duration_ms / 1000.0

        # Validate
        if self.num_steps <= 0:
            raise ValueError(
                f"Invalid time configuration: "
                f"duration_ms={duration_ms} and dt_ms={dt_ms} "
                f"result in {self.num_steps} time steps"
            )

    def ms_to_steps(
        self, time_ms: Union[float, np.ndarray], clip: Optional[bool] = False
    ) -> Union[int, np.ndarray]:
        """Convert time in milliseconds to time steps."""
        steps = np.round(time_ms / self.dt_ms).astype(int)
        # Ensure values are within valid range
        if clip:
            if isinstance(steps, np.ndarray):
                steps = np.clip(steps, 0, self.num_steps - 1)
            else:
                steps = max(0, min(self.num_steps - 1, steps))
        else:
            steps = np.where(steps > self.num_steps, -1, steps)
        return steps

    def steps_to_ms(self, steps: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time steps to time in milliseconds."""
        return steps * self.dt_ms

    def ms_to_sec(self, time_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time in milliseconds to seconds."""
        return time_ms / 1000.0

    def sec_to_ms(self, time_sec: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time in seconds to milliseconds."""
        return time_sec * 1000.0

    def sec_to_steps(
        self, time_sec: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
        """Convert time in seconds to time steps."""
        return self.ms_to_steps(self.sec_to_ms(time_sec))

    def steps_to_sec(self, steps: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time steps to time in seconds."""
        return self.ms_to_sec(self.steps_to_ms(steps))

    def get_time_vector_ms(self) -> np.ndarray:
        """Return time vector in milliseconds."""
        return np.arange(self.num_steps) * self.dt_ms

    def get_time_vector_sec(self) -> np.ndarray:
        """Return time vector in seconds."""
        return np.arange(self.num_steps) * self.dt_sec

    def __str__(self) -> str:
        """String representation of time configuration."""
        return (
            f"EncoderTimeConfig(duration={self.duration_ms} ms ({self.duration_sec} s), "
            f"dt={self.dt_ms} ms ({self.dt_sec} s), num_steps={self.num_steps})"
        )


class SpikeEncoder:
    """Base class for all spike encoders with unified time handling."""

    def __init__(
        self,
        time_config: Optional[EncoderTimeConfig] = None,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
    ):
        """
        Initialize the spike encoder.

        Args:
            time_config: Time configuration object. If provided, duration_ms and dt_ms are ignored.
            duration_ms: Duration of encoding window in milliseconds.
            dt_ms: Time step size in milliseconds.
        """
        if time_config is not None:
            self.time_config = time_config
        else:
            self.time_config = EncoderTimeConfig(duration_ms=duration_ms, dt_ms=dt_ms)

    def encode(
        self,
        signal: ndarray,
        start_time_ms: Optional[float] = None,
        return_times: bool = False,
    ) -> Tuple[ndarray, Optional[float]]:
        """
        Encode a signal into a spike train.

        Args:
            signal: Input signal of shape [n_samples, n_features].
            start_time_ms: Start time for encoding in milliseconds.

        Returns:
            Tuple of (encoded signal, next start time in milliseconds).
        """
        raise NotImplementedError("SpikeEncoder is an abstract class.")

    def _validate_input(self, signal: ndarray) -> None:
        """Validate input signal shape."""
        if len(signal.shape) != 2:
            raise ValueError(
                f"Input signal must have shape [n_samples, n_features], got {signal.shape}"
            )
