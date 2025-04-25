import numpy as np
from spike_encoder import (
    EncoderTimeConfig,
    EncodingPipeline,
    LinearRateEncoder,
    ReceptiveFieldEncoder,
    PoissonSpikeGenerator,
)


def test_encoder():
    """Test the pipeline with rate encoder to spike generator."""

    # Create a time configuration
    time_config = EncoderTimeConfig(duration_ms=100.0, dt_ms=1.0)
    print(f"Time configuration: {time_config}")

    # Create example signal
    signal = np.array([[0.3, 0.7]])

    # Create a LinearRateEncoder to ReceptiveFieldEncoder to PoissonSpikeGenerator pipeline
    pipeline1 = EncodingPipeline(
        [
            LinearRateEncoder(
                time_config=time_config,
                input_range=(0, 1),
                max_firing_rate_hz=100.0,
                neurons_per_dim=1,
            ),
            PoissonSpikeGenerator(time_config=time_config, random_seed=42),
        ]
    )

    # Create a ReceptiveFieldEncoder to PoissonSpikeGenerator pipeline
    pipeline2 = EncodingPipeline(
        [
            ReceptiveFieldEncoder(
                time_config=time_config,
                input_range=(0, 1),
                max_firing_rate_hz=100.0,
                neurons_per_dim=3,
                tuning_width=0.2,
            ),
            PoissonSpikeGenerator(time_config=time_config, random_seed=42),
        ]
    )

    # Encode signal with both pipelines
    spikes1, next_time_ms1 = pipeline1.encode(signal)
    spikes2, next_time_ms2 = pipeline2.encode(signal)

    print(f"Signal shape: {signal.shape}")
    print(f"LinearRateEncoder + PoissonSpikeGenerator output shape: {spikes1.shape}")
    print(
        f"ReceptiveFieldEncoder + PoissonSpikeGenerator output shape: {spikes2.shape}"
    )

    # Count spikes to verify the pipelines work correctly
    print(f"LinearRateEncoder pipeline spike count: {np.sum(spikes1)}")
    print(f"ReceptiveFieldEncoder pipeline spike count: {np.sum(spikes2)}")

    # Test returning spike times
    spike_times1, _ = pipeline1.encode(signal, return_times=True)
    spike_times2, _ = pipeline2.encode(signal, return_times=True)

    print(f"Pipeline 1 can return spike times: {isinstance(spike_times1, list)}")
    if isinstance(spike_times1, list):
        for i, neuron_times in enumerate(spike_times1[0]):
            print(f"Neuron {i} spike times (ms): {neuron_times}")

    print(f"Pipeline 2 can return spike times: {isinstance(spike_times2, list)}")
    if isinstance(spike_times2, list):
        for i, neuron_times in enumerate(spike_times2[0]):
            print(f"Neuron {i} spike times (ms): {neuron_times}")

    return spikes1, spikes2


if __name__ == "__main__":
    test_encoder()
