import tensorflow as tf
import numpy as np



class FrameUNet:
    def __init__(self, input_dim: np.ndarray):
        self.input_dim = input_dim

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_dim)

        # Contracting Path
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        # Bottleneck
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)

        # Expansive Path
        u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
        u1 = tf.keras.layers.concatenate([u1, c2], axis=3)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

        u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
        u2 = tf.keras.layers.concatenate([u2, c1], axis=3)
        c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)
        c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

        # Output
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


FrameUNet(np.array([130, 130, 3])).build_model()