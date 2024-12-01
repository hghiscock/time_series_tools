# time_series_tools
Draft tools for time series analysis
## Online wavelet transformer
A lightweight implementation of the `pywt` wavelet transformer for use online. Note, the `OnlineWaveletTransformer` object stores a cache relating to a specific time series, so a separate transformer is needed for each signal
```python
from time_series_tools.wavelets import OnlineWaveletTransformer

transformer = OnlineWaveletTransformer(
    sampling_rate, frequencies, wavelet,
)

while True:
    update = get_time_series_update()
    power_spectrum = transformer.calculate(update)
```
