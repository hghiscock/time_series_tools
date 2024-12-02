import pytest
import pandas as pd
import numpy as np

from time_series_tools.object_detection import data


def test_variable_length_sequence_batches(training_data):
    feature_list = ["a", "b"]
    label = "y"
    dataset = data.TFVariableLengthSequenceBatches(
        training_data, feature_list, label,
    )
    assert len(dataset) == 2
    expected_data1 = np.array([
        [[1, 4], [2, 5], [3, 6]],
        [[1, 4], [2, 5], [3, 6]],
        [[1, 4], [2, 5], [3, 6]],
    ])
    expected_y1 = np.zeros((3, 3))
    expected_data2 = np.array([
        [[4, 7], [5, 8]],
        [[4, 7], [5, 8]],
        [[4, 7], [5, 8]],
    ])
    expected_y2 = np.ones((3, 2))
    assert np.allclose(dataset[0][0], expected_data1)
    assert np.allclose(dataset[0][1], expected_y1)
    assert np.allclose(dataset[1][0], expected_data2)
    assert np.allclose(dataset[1][1], expected_y2)


@pytest.fixture
def training_data(df1, df2):
    return [df1, df2, df1, df1, df2, df2]


@pytest.fixture
def df1():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
        "y": [0, 0, 0],
    })


@pytest.fixture
def df2():
    return pd.DataFrame({
        "a": [4, 5],
        "b": [7, 8],
        "c": [10, 11],
        "y": [1, 1],
    })
