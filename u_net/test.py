import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.saved_model.load("./u_net/saved_model")

for i in range(12):  # Adjust the range based on how many 'layer_with_weights' you have
    layer = getattr(model, f'layer_with_weights-{i}')
    weights = layer.kernel.numpy()
    biases = layer.bias.numpy()

    # Print weights and biases
    print("Weights:", weights)
    print("Biases:", biases)

    # Check for NaN values
    if np.isnan(weights).any() or np.isnan(biases).any():
        print("NaN values found in layer parameters")

infer = model.signatures['serving_default']

input = np.load("./web_simulator/noisy_frames/frame_17.npy")
input = tf.constant(tf.cast(input, dtype=tf.float32))
input = tf.expand_dims(input, axis=-1)
input = tf.expand_dims(input, axis=0)

output = infer(input)['conv2d_10'].numpy()
output = np.squeeze(output)

print(output)

plt.imshow(output, cmap='gray')
plt.show()


