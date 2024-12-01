import numpy as np
import pywt


class OnlineWaveletTransformer:
    # Adapted minimal version of source code from pywt

    def __init__(
        self, sampling_rate, frequencies, wavelet,
        precision=10, wavefunction_remove_fraction=0.5,
    ):
        """
        A lightweight calculator to perform online wavelet transform

        Parameters
        ----------
        sampling_rate : float
            The frequency resolution of the input data i.e. 1 / time step
        frequencies : ndarray
            Array of frequencies for which to calculate components
        wavelet : str
            Name of the pywt wavelet to use
        precision : int, optional
            pywt precision of wavelet. Default 10
        wavefunction_remove_fraction : float, optional
            Fraction of the pywt wavefunction to remove when calculating the
            transform. Areas where the wavefunction is close to zero at the
            extremes of the domain are discarded. Increasing improves
            performance at the cost of accuracy. Default 0.5

        Returns
        -------
        transformer : OnlineWaveletTransformer
            object to perform online wavelet transform using the .calculate
            method
        """
        self.time_step = 1 / sampling_rate
        self.frequencies = frequencies
        self.precision = precision
        if (
            (wavefunction_remove_fraction >= 1.0)
            or (wavefunction_remove_fraction < 0)
        ):
            raise ValueError(
                "Invalid wavefunction_remove_fraction, must be between 0 and 1"
            )
        self.wavefunction_remove_fraction = wavefunction_remove_fraction
        self.wavelet = pywt.ContinuousWavelet(wavelet)
        self.scales = pywt.frequency2scale(
            wavelet, self.frequencies * self.time_step,
        )
        (
            self.int_psi,
            self.int_psi_scales,
            self.int_psi_scale_lengths,
        ) = self._get_wavefunctions()
        self.max_wavefunction_length = max(self.int_psi_scale_lengths.values())
        self.values = np.array([], dtype="complex64")
        self.convolution_cache = {scale: np.nan for scale in self.scales}

    def calculate(self, value):
        """
        Calculate step of wavelet transform

        Parameters
        ----------
        value: float
            Next update in time series

        Returns
        -------
        intensities : ndarray
            power spectrum for the requested frequencies at that timestep.
        """
        self.values = np.append(self.values, value)
        intensities = self._compute_time_frequency_power_spectrum(
            self.values[-self.max_wavefunction_length:],
        )
        return intensities

    def _get_wavefunctions(self):
        psi, x = self._get_base_wavelet()
        step = x[1] - x[0]
        domain = x[-1] - x[0]
        int_psi = np.conj(np.cumsum(psi) * step)
        int_psi_scales = self._get_scaled_wavelets(
            int_psi, step, domain,
        )
        int_psi_scale_lengths = {
            scale: len(psi) for scale, psi in int_psi_scales.items()
        }
        return int_psi, int_psi_scales, int_psi_scale_lengths

    def _get_base_wavelet(self):
        # trim wavelet to remove edges of wavefunctions
        # that are close to zero
        psi, x = self.wavelet.wavefun(self.precision)
        lenx = int(len(x) * self.wavefunction_remove_fraction / 2)
        return psi[lenx:-lenx], x[lenx:-lenx]

    def _get_scaled_wavelets(self, int_psi, step, domain):
        int_psi_scales = {}
        for scale in self.scales:
            js = np.arange(scale * domain + 1) / (scale * step)
            js = js[js < int_psi.size]
            js = js.astype(int)
            int_psi_scales[scale] = int_psi[js]
        return int_psi_scales

    def _compute_time_frequency_power_spectrum(self, values):
        out = np.empty_like(self.scales, dtype="complex64")
        len_values = len(values)
        for i, scale in enumerate(self.scales):
            len_wavelet = self.int_psi_scale_lengths[scale]
            conv = self._convolve(
                values, self.int_psi_scales[scale],
                len_values, len_wavelet,
            )
            coeff = - np.sqrt(scale) * (conv - self.convolution_cache[scale])
            self.convolution_cache[scale] = conv
            out[i] = coeff
        return np.abs(out)

    def _convolve(self, values, wavefun, len_values, len_wavelet):
        # assign the last "valid" value of the convolution
        # to that time step which introduces a lag
        v1 = values[-len_wavelet:]
        v2 = wavefun[-len_values:]
        return np.dot(v1, v2)
