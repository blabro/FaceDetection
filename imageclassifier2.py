import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# print(train_labels[0])
# print(train_images[0])
# plt.imshow(train_images[0], cmap='gray', vmin = 0, vmax = 255)
# plt.show()



# define our neural net structure
model = keras.Sequential([
    # Flattens the 28x28 into a 784x1 input layer
    keras.layers.Flatten(input_shape=(28,28)),

    # hidden layer is 128 deep. relu returns the value or 0
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10 (depending on what piece of clothing it is
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])

# Compile our model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

# Train our model, using our training data
model.fit(train_images, train_labels, epochs=5)

# test our model, using our testing data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(train_images[1], cmap='gray', vmin = 0, vmax = 255)
plt.show()

print(test_labels[1])

# make predictions
predictions = model.predict(test_images)

print(predictions[0])

print(list(predictions[0]).index(max(predictions[0])))



