import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Load the mile per gallon dataset data
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# Use Panda to read the data
column_names = \
    ['MPG','Cylinders','Displacement','Horsepower','Weight',
     'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

# Examine some data
#      MPG  Cylinders  Displacement  ...  Accel  Model Year  Origin
# 393  27.0          4         140.0  ...  15.6     82       1
# 394  44.0          4          97.0  ...  24.6     82       2
# 395  32.0          4         135.0  ...  11.6     82       1
# 396  28.0          4         120.0  ...  18.6     82       1
# 397  31.0          4         119.0  ...  19.4     82       1
dataset = raw_dataset.copy()
print(dataset.tail())

# Remove rows that have invalid entry
dataset.isna().sum()
dataset = dataset.dropna()

# Convert categorical data into 1-hot-vector
#       MPG  Cylinders  ...  Model Year  USA  Europe  Japan
# 393  27.0          4       82          1.0    0.0    0.0
# 394  44.0          4       82          0.0    1.0    0.0
# 395  32.0          4       82          1.0    0.0    0.0
# 396  28.0          4       82          1.0    0.0    0.0
# 397  31.0          4       82          1.0    0.0    0.0
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# Create the training and testing dataset
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Exploratory data analysis
# Plot graphs in understanding co-relations.
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
    diag_kind="kde")
plt.show()

# Examine the data statistics
#               count     mean     std  ...     50%      75%     max
# Cylinders     314.0     5.47    1.69  ...     4.0     8.00     8.0
# Displacement  314.0   195.31  104.33  ...   151.0   265.75   455.0
# Horsepower    314.0   104.86   38.09  ...    94.5   128.00   225.0
# Weight        314.0  2990.25  843.89  ...  2822.5  3608.00  5140.0
# Acceleration  314.0    15.55    2.78  ...    15.5    17.20    24.8
# Model Year    314.0    75.89    3.67  ...    76.0    79.00    82.0
# USA           314.0     0.62    0.48  ...     1.0     1.00     1.0
# Europe        314.0     0.17    0.38  ...     0.0     0.00     1.0
# Japan         314.0     0.19    0.39  ...     0.0     0.00     1.0
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# Extract MPG as the labels.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize input features.
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Create a model and configure the training
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu',
                 input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Instantiate a model
model = build_model()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 64)                640
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 4,865
# Trainable params: 4,865
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

# Some insanity check
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# Train the model
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# Examine the training statistics
# loss       mae       mse  val_loss   val_mae   val_mse  epoch
# 995  2.359009  0.993998  2.359010  8.484052  2.203  8.484   995
# 996  2.491158  1.014341  2.491158  7.810521  2.070  7.81    996
# 997  2.574141  1.037746  2.574141  8.373812  2.173  8.37    997
# 998  2.518765  1.034081  2.518765  7.546564  2.061  7.54    998
# 999  2.655058  1.043352  2.655058  8.503022  2.186  8.50    999
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

# Plot the errors for training & validation
plot_history(history)
plt.show()

# Validation error is increasing.
# The model is overfited.
# We will create a new model and stop
# the training early when there is no improvement.
model = build_model()

# The patience parameter is the amount of epochs
# to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()

# Making predictions
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()