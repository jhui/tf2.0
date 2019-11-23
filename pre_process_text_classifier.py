import numpy as np
import tensorflow as tf

from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Load IMDB data
(train_data, test_data), info = tfds.load(
    # Use the pre-encoded version
    # with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the
    # dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure.
    with_info=True)

# A function converts text to/from a sequence of integers.
# It is used for demonstration purpose here.
encoder = info.features['text'].encoder

# Vocabulary size: 8185
print ('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

# Show how string is encoded.
# Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]
# The original string: "Hello TensorFlow."
# 4025 ----> Hell
# 222 ----> o
# 6307 ----> Ten
# 2327 ----> sor
# 4043 ----> Fl
# 2120 ----> ow
# 7975 ----> .
encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))
original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))
assert original_string == sample_string
for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

# The loaded training data is already
# encoded as a sequence of integers.
# Encoded text: [ 133 67 1011 5 5225 7961 1482 2252 755 6]
# Label: 1
for train_example, train_label in train_data.take(1):
  print('Encoded text:', train_example[:10].numpy())
  print('Label:', train_label.numpy())

# "I have no idea what the other reviewer ..."
print(encoder.decode(train_example))

BUFFER_SIZE = 1000

# Display the shape of 2 sampled batches
# train_batches data: (None, 32, 1355), (None, 32, 1002)
# train_batches label: (None, 32,), (None, 32,)
train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes))

test_batches = (
    test_data
    .padded_batch(32, train_data.output_shapes))

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

# Create the model
# The training data contains a sequence of integers.
# It will convert to a sequence of 16-D vectors.
# Then, it is average out in representing the review.
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 16)          130960
# _________________________________________________________________
# global_average_pooling1d (Gl (None, 16)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 17
# =================================================================
# Total params: 130,977
# Trainable params: 130,977
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

loss, accuracy = model.evaluate(test_batches)

# Loss:  0.33158007718603627
# Accuracy:  0.87552
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# What history contains:
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
history_dict = history.history
print(history_dict.keys())

# Plot 2 graphs.
# The loss for the training and validation
# The accuracy for the training and validation
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
