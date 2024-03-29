import tensorflow as tf
from tensorflow import keras
import numpy as np

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu',
                                  input_shape=(10,)))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 16)                176
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 193
# Trainable params: 193
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

model.fit(dataset, epochs=1)

pass