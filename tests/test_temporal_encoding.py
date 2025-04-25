import numpy as np
from spike_encoder import EncoderTimeConfig, LatencyEncoder, PhaseEncoder, BurstEncoder


def test_temporal_encoding():
    """Example demonstrating temporal encoding API."""
    # Create a signal
    signal = np.array(
        [
            [0.2, 0.5, 0.8],  # Sample 1 with 3 features
            [0.9, 0.3, 0.1],  # Sample 2 with 3 features
        ]
    )

    # Create a time configuration
    time_config = EncoderTimeConfig(duration_ms=100.0, dt_ms=0.5)
    print(f"Time configuration: {time_config}")

    # Create a latency encoder
    latency_encoder = LatencyEncoder(
        time_config=time_config,
        input_range=(0, 1),
        neurons_per_dim=1,
        min_latency_ms=5.0,
        max_latency_ms=50.0,
    )

    # Encode the signal
    latency_spikes, next_time_ms = latency_encoder.encode(signal)

    print(f"Signal shape: {signal.shape}")
    print(f"Latency-encoded spikes shape: {latency_spikes.shape}")

    # Count spikes for each neuron
    print("\nLatency encoding spike counts:")
    for i in range(signal.shape[0]):
        print(f"Sample {i + 1}:")
        for j in range(signal.shape[1]):
            count = latency_spikes[i, :, j].sum()
            print(f"  Feature {j + 1}: {count} spikes")

    # Create a phase encoder
    phase_encoder = PhaseEncoder(
        time_config=time_config,
        input_range=(0, 1),
        neurons_per_dim=1,
        base_freq_hz=10.0,
        max_phase_shift_rad=np.pi,
    )

    # Encode the signal
    phase_spikes, _ = phase_encoder.encode(signal)

    print("\nPhase encoding spike counts:")
    for i in range(signal.shape[0]):
        print(f"Sample {i + 1}:")
        for j in range(signal.shape[1]):
            count = phase_spikes[i, :, j].sum()
            print(f"  Feature {j + 1}: {count} spikes")

    # Find spike times for the first sample, first feature
    spike_indices = np.where(latency_spikes[0, :, 0])[0]
    spike_times_ms = time_config.steps_to_ms(spike_indices)
    print(
        f"\nLatency encoding spike times for Sample 1, Feature 1: {spike_times_ms} ms"
    )

    # Find phase encoding spike times for comparison
    phase_indices = np.where(phase_spikes[0, :, 0])[0]
    phase_times_ms = time_config.steps_to_ms(phase_indices)
    print(f"Phase encoding spike times for Sample 1, Feature 1: {phase_times_ms} ms")


def test_burst_encoding():
    """Test the burst encoder with the TimeConfig API."""
    # Create a time configuration
    time_config = EncoderTimeConfig(duration_ms=100.0, dt_ms=1.0)
    print(f"Time configuration: {time_config}")

    # Create a burst encoder with explicit time parameters
    encoder = BurstEncoder(
        time_config=time_config,
        input_range=(0, 1),
        neurons_per_dim=1,
        min_spikes=1,
        max_spikes=5,
        burst_window_ms=20.0,
        inter_spike_interval_ms=3.0,
    )

    # Generate example signal with different intensities
    signal = np.array(
        [
            [0.2, 0.8],  # First sample with two features
            [0.5, 0.5],  # Second sample with two features
            [0.9, 0.1],  # Third sample with two features
        ]
    )

    # Encode signal
    spikes, next_time_ms = encoder.encode(signal)

    print(f"Signal shape: {signal.shape}")
    print(f"Encoded spikes shape: {spikes.shape}")
    print(f"Next start time: {next_time_ms} ms")

    # Count spikes for each feature
    for i in range(signal.shape[0]):
        print(f"\nSample {i + 1}:")
        for j in range(signal.shape[1]):
            neuron_idx = j  # With neurons_per_dim=1
            spike_count = np.sum(spikes[i, :, neuron_idx])
            expected_spikes = 1 + int(
                signal[i, j] * (5 - 1)
            )  # min_spikes + normalized * (max_spikes - min_spikes)

            print(
                f"  Feature {j + 1} (value {signal[i, j]:.2f}): {spike_count} spikes (expected: {expected_spikes})"
            )

            # Find spike times
            spike_indices = np.where(spikes[i, :, neuron_idx])[0]
            spike_times_ms = time_config.steps_to_ms(spike_indices)
            print(f"  Spike times: {spike_times_ms} ms")

    return spikes


if __name__ == "__main__":
    test_temporal_encoding()
    test_burst_encoding()
