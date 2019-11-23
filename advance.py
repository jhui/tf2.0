import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

# x_train (60000, 28, 28), y_train (60000,)
# x_test  (10000, 28, 28), y_text (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train (60000, 28, 28, 1)
# x_test  (10000, 28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# train_ds (None, 28, 28, 1)
# Create a datasource for 1875 batches of 32 samples.
# It uses an internal 10K buffer with shuffling.
# And selected elements will be replaced with new elements.
train_ds = tf.data.Dataset.from_tensor_slices(
                    (x_train, y_train)).shuffle(10000).batch(32)

# test_ds  (None, 28, 28, 1)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,
                                              y_test)).batch(32)

# Define the deep network using tensorflow.keras.Model
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # Define Model's layers
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  # Define the foward pass.
  # Process the input x.
  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model.
model = MyModel()

# Define the loss function and the optimizer to be used.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define the metrics & accuracy to be used in the training & testing
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='test_accuracy')

# Create a callable TensorFlow graph from a Python function.
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # Make predictions and compute the loss
    predictions = model(images)
    loss = loss_object(labels, predictions)
  # Compute the gradients and optimize the trainable parameters
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,
                                model.trainable_variables))

  # Accumulate loss and accuracy information
  train_loss(loss)
  train_accuracy(labels, predictions)

# Function in testing
@tf.function
def test_step(images, labels):
  # Make predictions and compute the loss
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  # Accumulate loss and accuracy information
  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # For 60K sample, it loops 1875 times
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, ' \
             'Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

