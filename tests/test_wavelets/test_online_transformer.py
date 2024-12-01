import pytest
import pywt
import numpy as np

from time_series_tools.wavelets import online_transformer


def test_online_wavelet_transformer_init(frequencies, wavelet):
    precision = 5
    remove_fraction = 0.5

    transformer = online_transformer.OnlineWaveletTransformer(
        5.0, frequencies, wavelet,
        precision=precision, wavefunction_remove_fraction=remove_fraction,
    )
    base_wavefunctions = pywt.ContinuousWavelet(wavelet).wavefun(precision)
    expected_wavefunction_length = (
        len(base_wavefunctions[0]) * remove_fraction
    )
    assert isinstance(transformer.wavelet, pywt.ContinuousWavelet)
    assert len(transformer.int_psi) == expected_wavefunction_length
    assert len(transformer.scales) == len(frequencies)
    assert len(transformer.int_psi_scales) == len(frequencies)
    assert len(transformer.int_psi_scale_lengths) == len(frequencies)
    assert len(transformer.values) == 0
    assert len(transformer.convolution_cache) == len(frequencies)
    assert all([np.isnan(x) for x in transformer.convolution_cache.values()])


@pytest.mark.parametrize("remove_fraction", [-0.1, 1.0, 5.0])
def test_online_wavelet_transformer_init_raises_on_invalid_fraction(
    remove_fraction, frequencies, wavelet,
):
    with pytest.raises(ValueError):
        transformer = online_transformer.OnlineWaveletTransformer(
            1.0, frequencies, wavelet,
            wavefunction_remove_fraction=remove_fraction,
        )


def test_online_wavelet_transformer_calculate(frequencies, wavelet):
    t = np.linspace(0, 10, 1001)
    sampling_rate = 1.0 / np.nanmin(np.diff(t))
    frequency = 2.0
    correct_frequency_index = np.where(frequencies == frequency)[0][0]
    signal = np.sin(2 * np.pi * frequency * t)
    transformer = online_transformer.OnlineWaveletTransformer(
        sampling_rate, frequencies, wavelet,
    )
    result = np.empty_like(t, dtype="object")
    for i, value in enumerate(signal):
        result[i] = transformer.calculate(value)
    assert result[0].shape == (20,)
    # ignore warmup section where edge effects dominate
    assert all([
        np.argmax(x) == correct_frequency_index
        for x in result[500:]
    ])


@pytest.fixture
def frequencies():
    return np.linspace(1.0, 2.9, 20)


@pytest.fixture
def wavelet():
    return "cmor2.5-1.5"
