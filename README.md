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
### Variable length sequences
Tensorflow dataset wrapper for data containing sequences of variable length, each batch being all data with the same sequence length (therefore careful consideration should be given about the batches that will result from the data passed in)
```python
from time_series_tools.object_detection import data

training_batches = data.TFVariableLengthSequenceBatches(
    training_data, feature_list, label,
)
```
