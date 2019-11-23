# !pip install -q git+https://github.com/tensorflow/examples.git

import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

# Load a dataset
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

# Prepare a training and validation dataset.

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'],
                                (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'],
                               (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'],
                                (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'],
                               (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# Display an image, its ground truth mask
# * Belong to an object
# * Not belong to an object
# * Surrounding pixels
# & the model prediction
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(
        display_list[i]))
    plt.axis('off')
  plt.show()

# Display a training image for examination.
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
plt.show()

OUTPUT_CHANNELS = 3

# Load the MobileNet v2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

# Model: "model"
# _____________________________________________________________
# Layer (type)                    Output Shape         Param #
# =============================================================
# input_1 (InputLayer)            [(None, 128, 128, 3) 0
# _____________________________________________________________
# Conv1_pad (ZeroPadding2D)       (None, 129, 129, 3)  0
# _____________________________________________________________
#
# ...
#
# block_16_project (Conv2D)       (None, 4, 4, 320)    307200
# =============================================================
# Total params: 1,841,984
# Trainable params: 1,811,072
# Non-trainable params: 30,912
# _____________________________________________________________
down_stack.summary()

down_stack.trainable = False

# Create an upsampling model
# This is trained to create a mask from
# the extracted features.
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

# Model used for the image segmentation.
def unet_model(output_channels):

  # This is the last layer of the model
  # It is a transpose convolution layer.
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  # skips:
  # 0 = {Tensor} shape=(None, 64, 64, 96), dtype=float32
  # 1 = {Tensor} shape=(None, 32, 32, 144), dtype=float32
  # 2 = {Tensor} shape=(None, 16, 16, 192), dtype=float32
  # 3 = {Tensor} shape=(None, 8, 8, 576), dtype=float32
  # 4 = {Tensor} shape=(None, 4, 4, 320), dtype=float32
  skips = down_stack(x)

  # Last element in skips
  # Tensor shape=(None, 4, 4, 320), dtype=float32
  x = skips[-1]

  # Prepare an iterator similar to
  # 0 = {Tensor} shape=(None, 8, 8, 576), dtype=float32
  # 1 = {Tensor} shape=(None, 16, 16, 192), dtype=float32
  # 2 = {Tensor} shape=(None, 32, 32, 144), dtype=float32
  # 3 = {Tensor} shape=(None, 64, 64, 96), dtype=float32
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    # e.g. x: Tensor shape=(None, 8, 8, 512), dtype=float32
    # skip: Tensor shape=(None, 8, 8, 576), dtype=float32
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# Create an image segmentation model.
# First, features are extracted and
# upsampled to create the mask.
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# _____________________________________________________________
# Model: "model_1"
# ______________________________________________________________
# Layer (type)                    Output Shape         Param #
# ==============================================================
# input_2 (InputLayer)            [(None, 128, 128, 3) 0
# ______________________________________________________________
# model (Model)                   [(None, 64, 64, 96), 1841984
# ______________________________________________________________
# sequential (Sequential)         (None, 8, 8, 512)    1476608
# ______________________________________________________________
# concatenate (Concatenate)       (None, 8, 8, 1088)   0
# ______________________________________________________________
# sequential_1 (Sequential)       (None, 16, 16, 256)  2507776
# ______________________________________________________________
# concatenate_1 (Concatenate)     (None, 16, 16, 448)  0
# ______________________________________________________________
# sequential_2 (Sequential)       (None, 32, 32, 128)  516608
# ______________________________________________________________
# concatenate_2 (Concatenate)     (None, 32, 32, 272)  0
# ______________________________________________________________
# sequential_3 (Sequential)       (None, 64, 64, 64)   156928
# ______________________________________________________________
# concatenate_3 (Concatenate)     (None, 64, 64, 160)  0
# ______________________________________________________________
# conv2d_transpose_4 (Conv2DTrans (None, 128, 128, 3)  4323
# ==============================================================
# Total params: 6,504,227
# Trainable params: 4,660,323
# Non-trainable params: 1,843,904
# ______________________________________________________________
model.summary()

def create_mask(pred_mask):
  # pred_mask (None, 128, 128, 3)
  pred_mask = tf.argmax(pred_mask, axis=-1)
  # pred_mask (None, 128, 128)
  pred_mask = pred_mask[..., tf.newaxis]
  # pred_mask (None, 128, 128, 1)
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(
                 model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()
plt.show()

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)
plt.show()



