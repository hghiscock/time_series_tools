from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model


class ObjectDetector1D(Model):

    def __init__(
        self, num_features, latent_space_attention=False,
    ):
        super().__init__()
        self.latent_space_attention = latent_space_attention
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

        if latent_space_attention:
            self.multiheadattention = layers.MultiHeadAttention(
                num_heads=8, key_dim=8, dropout=0.1,
            )
            self.add = layers.Add()

    def call(self, x):
        encoded = self.encoder(x)
        if self.latent_space_attention:
            attn = self.multiheadattention(query=encoded, value=encoded)
            encoded = self.add([encoded, attn])
        decoded = self.decoder(encoded)
        return decoded
