import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from spike_encoder import EncoderTimeConfig, ReceptiveFieldEncoder


def test_multiple_receptive_fields():
    """Test the multiple receptive fields per neuron functionality."""

    # Create a time configuration
    time_config = EncoderTimeConfig(duration_ms=100.0, dt_ms=1.0)

    # Test parameters
    input_range = (0, 1)
    neurons_per_dim = 3
    num_centers_per_neuron = 3
    tuning_width = 0.1
    max_firing_rate_hz = 100.0
    min_firing_rate_hz = 0.0

    # Create encoders with different configurations
    single_center_encoder = ReceptiveFieldEncoder(
        time_config=time_config,
        input_range=input_range,
        max_firing_rate_hz=max_firing_rate_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        neurons_per_dim=neurons_per_dim,
        tuning_width=tuning_width,
        num_centers_per_neuron=1,  # Default: one center per neuron
    )

    multi_center_max_encoder = ReceptiveFieldEncoder(
        time_config=time_config,
        input_range=input_range,
        max_firing_rate_hz=max_firing_rate_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        neurons_per_dim=neurons_per_dim,
        tuning_width=tuning_width,
        num_centers_per_neuron=num_centers_per_neuron,
        response_combination="max",
    )

    multi_center_sum_encoder = ReceptiveFieldEncoder(
        time_config=time_config,
        input_range=input_range,
        max_firing_rate_hz=max_firing_rate_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        neurons_per_dim=neurons_per_dim,
        tuning_width=tuning_width,
        num_centers_per_neuron=num_centers_per_neuron,
        response_combination="sum",
    )

    # Test 1: Verify centers were created correctly
    print("Single center encoder - Centers for first feature dimension:")
    for i, centers in enumerate(single_center_encoder.centers[0]):
        print(f"  Neuron {i}: {centers}")

    print("\nMulti-center encoder - Centers for first feature dimension:")
    for i, centers in enumerate(multi_center_max_encoder.centers[0]):
        print(f"  Neuron {i}: {centers}")

    # Test 2: Generate tuning curves
    test_inputs = np.linspace(0, 1, 100).reshape(-1, 1)
    print(f"Test inputs shape: {test_inputs.shape}")

    # Compute rates for each encoder
    single_rates = single_center_encoder.compute_rates(test_inputs)
    multi_max_rates = multi_center_max_encoder.compute_rates(test_inputs)
    multi_sum_rates = multi_center_sum_encoder.compute_rates(test_inputs)

    # Print the shapes to diagnose
    print(f"Single rates shape: {single_rates.shape}")
    print(f"Multi max rates shape: {multi_max_rates.shape}")
    print(f"Multi sum rates shape: {multi_sum_rates.shape}")

    assert (len(single_rates.shape)) == 2
    assert (len(multi_max_rates.shape)) == 2
    assert (len(multi_sum_rates.shape)) == 2

    # Visualize tuning curves
    plt.figure(figsize=(15, 5))

    # Single center encoder
    plt.subplot(1, 3, 1)
    for i in range(neurons_per_dim):
        plt.plot(test_inputs, single_rates[:, i], label=f"Neuron {i}")
    plt.title("Single Center per Neuron")
    plt.xlabel("Input")
    plt.ylabel("Firing Rate (Hz)")
    plt.legend()

    # Multi-center max encoder
    plt.subplot(1, 3, 2)
    for i in range(neurons_per_dim):
        plt.plot(test_inputs, multi_max_rates[:, i], label=f"Neuron {i}")
    plt.title("Multiple Centers (Max Combination)")
    plt.xlabel("Input")
    plt.legend()

    # Multi-center sum encoder
    plt.subplot(1, 3, 3)
    for i in range(neurons_per_dim):
        plt.plot(test_inputs, multi_sum_rates[:, i], label=f"Neuron {i}")
    plt.title("Multiple Centers (Sum Combination)")
    plt.xlabel("Input")
    plt.legend()

    plt.tight_layout()
    plt.savefig("receptive_field_comparison.png")
    plt.show()

    # Test 3: Test specific input values to verify multiple peaks
    test_points = np.array(
        [
            [0.1],  # Near first center of multi-center neurons
            [0.5],  # Near middle center of multi-center neurons
            [0.9],  # Near last center of multi-center neurons
        ]
    )

    print("\nTesting specific input points:")
    for i, point in enumerate(test_points):
        s_rates = single_center_encoder.compute_rates(point.reshape(1, -1))
        m_max_rates = multi_center_max_encoder.compute_rates(point.reshape(1, -1))
        m_sum_rates = multi_center_sum_encoder.compute_rates(point.reshape(1, -1))

        # Reshape if necessary
        if len(s_rates.shape) == 1:
            s_rates = s_rates.reshape(-1, 1)
            m_max_rates = m_max_rates.reshape(-1, 1)
            m_sum_rates = m_sum_rates.reshape(-1, 1)

        print(f"Input {point[0]:.1f}:")
        print(f"  Single center rates: {s_rates}")
        print(f"  Multi-center (max) rates: {m_max_rates}")
        print(f"  Multi-center (sum) rates: {m_sum_rates}")

    # Test 4: Count peaks in tuning curves
    def count_peaks(rates_array, threshold=0.5):
        """
        Count peaks in a 1D array of rates using SciPy's find_peaks.

        Args:
        rates_array: 1D array of rates
        threshold: Minimum height for a peak (relative to max_firing_rate_hz)

        Returns:
        List of (input_value, rate) tuples for each peak
        """
        # Find peaks with minimum height requirement
        min_height = max_firing_rate_hz * threshold
        padded_rates_array = np.concatenate(
            ([np.min(rates_array) - 1], rates_array, [np.min(rates_array) - 1])
        )

        # Use prominence to ensure we only detect significant peaks
        # Prominence helps distinguish meaningful peaks from noise
        peak_indices, properties = find_peaks(
            padded_rates_array,
            height=min_height,
            prominence=max_firing_rate_hz
            * 0.1,  # 10% of max rate as minimum prominence
        )

        # Account for padding
        if peak_indices.size > 0:
            peak_indices -= 1

        print(f"peak_indices = {peak_indices}")
        # Convert to (input_value, rate) tuples
        peaks = [(test_inputs[idx, 0], rates_array[idx]) for idx in peak_indices]
        return peaks

    print("\nNumber of peaks detected:")
    for i in range(neurons_per_dim):
        # Extract 1D arrays for each neuron
        if neurons_per_dim > 1:
            single_neuron_rates = single_rates[:, i]
            multi_max_neuron_rates = multi_max_rates[:, i]
            multi_sum_neuron_rates = multi_sum_rates[:, i]
        else:
            single_neuron_rates = single_rates[:, 0]
            multi_max_neuron_rates = multi_max_rates[:, 0]
            multi_sum_neuron_rates = multi_sum_rates[:, 0]

        # Count peaks in each 1D array
        single_peaks = count_peaks(single_neuron_rates)
        multi_max_peaks = count_peaks(multi_max_neuron_rates)
        multi_sum_peaks = count_peaks(multi_sum_neuron_rates, threshold=0.3)

        print(f"Neuron {i}:")
        print(f"  Single center: {len(single_peaks)} peaks at {single_peaks}")
        print(
            f"  Multi-center (max): {len(multi_max_peaks)} peaks at {multi_max_peaks}"
        )
        print(
            f"  Multi-center (sum): {len(multi_sum_peaks)} peaks at {multi_sum_peaks}"
        )

        # Assertions
        assert len(single_center_encoder.centers[0][i]) == 1, (
            "Single center encoder should have 1 center per neuron"
        )
        assert len(multi_center_max_encoder.centers[0][i]) == num_centers_per_neuron, (
            "Multi-center encoder should have correct number of centers"
        )

        # Only assert about peak counts if we can detect the issue with the shapes
        if neurons_per_dim > 1:
            assert len(single_peaks) <= 1, (
                f"Single center neuron {i} should have at most 1 peak"
            )
            # We may not get exactly num_centers_per_neuron peaks if they're too close together
            assert len(multi_max_peaks) > 1, (
                f"Multi-center (max) neuron {i} should have multiple peaks"
            )

    return single_rates, multi_max_rates, multi_sum_rates


if __name__ == "__main__":
    test_multiple_receptive_fields()
