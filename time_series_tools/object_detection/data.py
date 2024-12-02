import numpy as np
from collections import defaultdict
from tensorflow.keras.utils import Sequence


class TFVariableLengthSequenceBatches(Sequence):

    def __init__(self, data, feature_list, label):
        """
        Batch data into batches of the same length to train a Tensorflow
        model

        Parameters
        ----------
        data : list
            Data to transform into training data. List of Pandas dataframes
        feature_list : list
            Columns to use as features
        label : str
            Column containing the target

        Returns
        -------
        dataset : TFVariableLengthSequenceBatches
            Tensorflow Dataset with batches determined by matching sequence
            length
        """
        features, labels = self._extract_features_and_labels(
            data, feature_list, label,
        )
        self.x_data = [np.concatenate(x) for x in features.values()]
        self.y_data = [np.concatenate(y) for y in labels.values()]
        self.num_batches = len(self.x_data)

    def _extract_features_and_labels(self, data, feature_list, label):
        features = defaultdict(list)
        labels = defaultdict(list)
        for df in data:
            sequence_length = len(df)
            x = df[feature_list].values[np.newaxis, :, :]
            y = df[label].values[np.newaxis, :]
            features[sequence_length].append(x)
            labels[sequence_length].append(y)
        return features, labels

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
