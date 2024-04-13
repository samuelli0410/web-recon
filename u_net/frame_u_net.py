import tensorflow as tf
import numpy as np



class FrameUNet:
    def __init__(self, input_dim: np.ndarray):
        self.input_dim = input_dim

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_dim)

        # Contracting Path
        conv_relu_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        conv_relu_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_1)
        pooling_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_relu_1)

        conv_relu_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling_1)
        conv_relu_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_2)
        pooling_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_relu_2)

        # Bottleneck
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling_2)
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)

        # Expansive Path
        u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
        u1 = tf.keras.layers.concatenate([u1, conv_relu_2], axis=3)
        conv_relu_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
        conv_relu_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_3)

        u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_relu_3)
        u2 = tf.keras.layers.concatenate([u2, conv_relu_1], axis=3)
        conv_relu_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)
        conv_relu_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_4)

        # Output
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_relu_4)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


FrameUNet(np.array([1080, 1920, 1])).build_model()