import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.optimizers import SGD

import os
import copy

# Paths to the datasets
train_dir = "/content/train"
test_dir = "/content/test"

# Parameters
img_height = 180
img_width = 180
batch_size = 32

# Load datasets using image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',  # For binary classification
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# Extract class names from the dataset
class_names = train_ds.class_names

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True 
)

test_ds = test_ds.shuffle(buffer_size=len(list(test_ds)))

# Prefetch datasets for performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Standardizing the datasets
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation for training data
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2)
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Define the model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(96, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
lr = 0.001
opt = SGD(learning_rate=lr)
model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

model.save_weights('model_weights.weights.h5')

# Visualization of training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predictions
def plot_image(i, predictions_array, true_label, image_batch):
    true_label = int(true_label[i])
    img = copy.deepcopy(image_batch[i]).numpy() 
    img = (img * 255).astype("uint8")

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    score_scalar = predictions_array[i][0]  # Probability of positive class
    predicted_label = int(predictions_array[i] > 0.5)  # Threshold at 0.5

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * score_scalar if predicted_label == 1 else 100 * (1 - score_scalar),  # Conditional probability
        class_names[true_label]
    ), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = int(true_label[i])
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks([])
    prediction_value = predictions_array[i][0]  # Probability of positive class
    thisplot = plt.bar(range(len(class_names)), [1 - prediction_value, prediction_value], color="#777777")
    plt.ylim([0, 1])
    predicted_label = int(predictions_array[i] > 0.5)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Prediction and Visualization
for image_batch, label_batch in test_ds.take(1):
    predictions = model.predict(image_batch)  # Outputs probabilities for the positive class

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, label_batch, image_batch)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, label_batch)
    plt.tight_layout()
    plt.show()
