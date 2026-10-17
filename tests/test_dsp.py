import numpy as np

from audio_analyzer.dsp import (
    build_third_octave_band_bins,
    calculate_spectrum_from_samples,
    calculate_third_octave_power,
    power_to_db,
)


def test_1khz_sine_amplitude_and_level():
    sample_rate = 48_000
    duration_s = 1.0
    amplitude = 0.5
    frequency = 1_000.0
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    spectrum = calculate_spectrum_from_samples(signal, sample_rate)
    idx = int(np.argmax(spectrum.amplitude[1:])) + 1
    assert abs(spectrum.frequencies_hz[idx] - frequency) < 1e-9
    assert abs(spectrum.amplitude[idx] - amplitude) < 2e-4
    assert abs(spectrum.dbfs[idx] - 20 * np.log10(amplitude)) < 0.01


def test_1khz_third_octave_band_matches_tone_level():
    sample_rate = 48_000
    fft_size = 48_000
    amplitude = 0.5
    t = np.arange(fft_size) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 1_000.0 * t)
    spectrum = calculate_spectrum_from_samples(signal, sample_rate)
    centers, _, _, bins = build_third_octave_band_bins(
        spectrum.frequencies_hz,
        min_frequency_hz=20,
        max_frequency_hz=20_000,
    )
    powers = calculate_third_octave_power(spectrum, bins)
    band_idx = int(np.argmin(np.abs(centers - 1_000.0)))
    level = float(power_to_db(powers[band_idx]))
    assert abs(level - 20 * np.log10(amplitude)) < 0.05
