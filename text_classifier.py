import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Split the training set into 60% and 40% for training & validation
# 15,000 samples for training
# 10,000 samples for validation
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

# Load the imdb reviews
# 25,000 samples for testing
# as_supervised = True returns (data, label).
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# Display two input training sample
# train_examples_batch (2,)
# tf.Tensor(
# [b"As a lifelong fan of Dickens, ..."
#  b"Oh yeah! Jenna Jameson did it again! ..."],
#  shape=(2,), dtype=string)
train_examples_batch, train_labels_batch = \
    next(iter(train_data.batch(2)))
print(train_examples_batch)

# Load the embedding layer model from the TensorFlow Hub.
# It maps a string to a 20-D vector.
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

# Print out the embedding of the first 2 training data (strings)
# shape: (2, 20)
# tf.Tensor(
# [[ 3.98198  -4.48380   5.1773   -2.36434  -3.29386  -3.53645
#   -2.47869   2.55254   6.6885   -2.30767  -1.98078   1.13158
#   -3.03398  -0.76041  -5.7434    3.42425   4.7900   -4.030
#   -5.9921   -1.72974 ]
#  [ 3.42329  -4.2308    4.14885  -0.295535 -6.8023   -2.51638
#   -4.40023   1.9057    4.75127  -0.405380 -4.34016   1.03614
#    0.97440   0.715071 -6.26570   0.165339  4.5602   -1.31069
#   -3.11213  -2.13387 ]], shape=(2, 20), dtype=float32)
print(hub_layer(train_examples_batch[:2]))

# Create a model including the word embedding
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# keras_layer (KerasLayer)     (None, 20)                400020
# _________________________________________________________________
# dense (Dense)                (None, 16)                336
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 400,373
# Trainable params: 400,373
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# history store the training metrics:
# loss and accuracy for the training and the validation
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# review_result (2,)
# 0.9374
# 0.9996
review_results = model.predict(train_examples_batch[:2])

# Evaluate the testing data
# results contain the loss and accuracy metrics
# loss: 0.310
# accuracy: 0.870
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

