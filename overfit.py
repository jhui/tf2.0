import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


NUM_WORDS = 10000

# Load IMDB data
# train_data, train_labels, test_data, test_labels (25,000,)
# Contain 25,000 lists.
# Each list contains the indexes of words in a review.
(train_data, train_labels), (test_data, test_labels) = \
        keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# Create a 1-hot-matrix
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        # set specific indices of results[i] to 1s
        results[i, word_indices] = 1.0
    return results

# Create a high-dimensional space in representing the reviews.
# train_data, test_data (25,000, 1000)
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# Define the deep network.
baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation='relu',
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Configure the training.
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

# Review the model definition.
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                160016
# _________________________________________________________________
# dense_1 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 160,305
# Trainable params: 160,305
# Non-trainable params: 0
# _________________________________________________________________
baseline_model.summary()

# Train the model.
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# Create a smaller second model for comparison.
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu',
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_3 (Dense)              (None, 4)                 40004
# _________________________________________________________________
# dense_4 (Dense)              (None, 4)                 20
# _________________________________________________________________
# dense_5 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 40,029
# Trainable params: 40,029
# Non-trainable params: 0
# _________________________________________________________________
smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data,
                                                     test_labels),
                                    verbose=2)

# Create a even bigger model.
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu',
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_6 (Dense)              (None, 512)               5120512
# _________________________________________________________________
# dense_7 (Dense)              (None, 512)               262656
# _________________________________________________________________
# dense_8 (Dense)              (None, 1)                 513
# =================================================================
# Total params: 5,383,681
# Trainable params: 5,383,681
# Non-trainable params: 0
# _________________________________________________________________
bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data,
                                                   test_labels),
                                  verbose=2)

# Plot the cross-entropy for the training and
# validation on different models
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key],
             color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])
plt.show()

# Introduce L2-regularization into the model.
l2_model = keras.models.Sequential([
    keras.layers.Dense(16,
                kernel_regularizer=keras.regularizers.l2(0.001),
                activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16,
                kernel_regularizer=keras.regularizers.l2(0.001),
                activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data,
                                                 test_labels),
                                verbose=2)

# Compare the baseline with the L2 regularization model.
plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])
plt.show()

# Define dropout models
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu',
                       input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data,
                                                   test_labels),
                                  verbose=2)


# Compare the baseline with the dropout model.
plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])
plt.show()
