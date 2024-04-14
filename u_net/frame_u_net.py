import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os



class FrameUNet:
    def __init__(self, input_dim: np.ndarray):
        self.input_dim = input_dim

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_dim)

        # Contracting Path
        conv_relu_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        conv_relu_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_1)
        pooling_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_relu_1) # consider average pooling

        conv_relu_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling_1)
        conv_relu_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_relu_2)
        pooling_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_relu_2)

        # Bottleneck
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling_2)
        b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)

        # Expansive Path
        u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b) # upsample
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
    

# Load training data and split into training and testing sets
noisy_images = "./web_simulator/noisy_frames"
ideal_images = "./web_simulator/ideal_frames"

x = []
y = []

for file in os.listdir(noisy_images):
    arr = np.load(os.path.join(noisy_images, file))
    x.append(arr[np.newaxis, :, :, np.newaxis])  # Add new axis for batch and channel

for file in os.listdir(ideal_images):
    arr = np.load(os.path.join(ideal_images, file))
    y.append(arr[np.newaxis, :, :, np.newaxis])  # Same as above

# Convert lists to numpy arrays and ensure they are of float type for model compatibility
x = np.vstack(x).astype(np.float32)
y = np.vstack(y).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Load and train model
image_dimensions = np.array([1080, 1920, 1])

model = FrameUNet(image_dimensions).build_model()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

model.save("./saved_model")