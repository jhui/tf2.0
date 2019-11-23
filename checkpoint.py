# !pip install -q pyyaml h5py
# Required to save models in HDF5 format

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu',
                       input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 512)               401920
# _________________________________________________________________
# dropout (Dropout)            (None, 512)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                5130
# =================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                           filepath=checkpoint_path,
                           save_weights_only=True,
                           verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving
# the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# Create a basic model instance
model = create_model()

# Evaluate the model
# Since it is not trained, the performance
# should be at chance only
# 1000/1 - 0s - loss: 2.3664 - accuracy: 0.1050
# Untrained model, accuracy: 10.50%
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Loads the weights of the model from the checkpoint
model.load_weights(checkpoint_path)

# Re-evaluate the model
# 1000/1 - 0s - loss: 0.4837 - accuracy: 0.8630
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Include the epoch in the file name (uses `str.format`)
# So we can have multiple checkpoints during the training
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
              train_labels,
              epochs=50,
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels),
              verbose=0)

# Find the latest checkpoint model.
# training_2/cp-0050.ckpt
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
# 1000/1 - 0s - loss: 0.4254 - accuracy: 0.8810
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
# Restored model, accuracy: 88.10%
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model
# shuold be saved to HDF5.
# This include the model, the parameters and
# the training configuration.
model.save('my_model.h5')

# Recreate the exact same model, including its
# weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_10 (Dense)             (None, 512)               401920
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_11 (Dense)             (None, 10)                5130
# =================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# _________________________________________________________________
new_model.summary()

# Restored model, accuracy: 88.10%
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_12 (Dense)             (None, 512)               401920
# _________________________________________________________________
# dropout_6 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_13 (Dense)             (None, 10)                5130
# =================================================================
# Total params: 407,050
# Trainable params: 407,050
# Non-trainable params: 0
# _________________________________________________________________
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)

