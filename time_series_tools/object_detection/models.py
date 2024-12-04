from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model


class ObjectDetector1D(Model):

    def __init__(self, num_features):
        super().__init__()
        self.encoder = Sequential([
            layers.Input(shape=(None, num_features)),
            layers.Conv1D(
                32, kernel_size=7, strides=1,
                activation="relu", padding='same',
            ),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(
                32, kernel_size=5, strides=1,
                activation="relu", padding='same',
            ),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.SpatialDropout1D(0.1),
            layers.Conv1D(
                64, kernel_size=3, strides=1,
                activation="relu", padding='same',
            ),
            layers.SpatialDropout1D(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
        ])

        self.decoder = Sequential([
            layers.Conv1DTranspose(
                64, kernel_size=3, strides=1,
                activation="relu", padding='same',
            ),
            layers.SpatialDropout1D(0.1),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(
                32, kernel_size=5, strides=1,
                activation="relu", padding='same',
            ),
            layers.UpSampling1D(size=2),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(
                32, kernel_size=7, strides=1,
                activation="relu", padding='same',
            ),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
