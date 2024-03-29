import tensorflow as tf

# Load the MNIST data
# x_train (60000, 28, 28), y_train (60000,)
# x_test  (10000, 28, 28), y_text (10000,)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the deep network
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Configure, train and evaluate.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=2)

model.evaluate(x_test,  y_test, verbose=2)

# Make predictions
predictions = model.predict(x_test[:2])

pass