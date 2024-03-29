# !pip install -q imageio

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

# Loading MNIST
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0],
                                    28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0],
                                    28, 28, 1).astype('float32')
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000

# Create Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
    shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).\
    shuffle(TEST_BUF).batch(BATCH_SIZE)

# Create model
class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3,
              strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3,
              strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    # Model: "sequential"
    # _____________________________________________________________
    # Layer (type)                 Output Shape          Param #
    # =============================================================
    # conv2d (Conv2D)              (None, 13, 13, 32)    320
    # _____________________________________________________________
    # conv2d_1 (Conv2D)            (None, 6, 6, 64)      18496
    # _____________________________________________________________
    # flatten (Flatten)            (None, 2304)          0
    # _____________________________________________________________
    # dense (Dense)                (None, 100)           230500
    # =============================================================
    # Total params: 249,316
    # Trainable params: 249,316
    # Non-trainable params: 0
    # _____________________________________________________________
    self.inference_net.summary()

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32,
                                activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3,
              strides=(1, 1), padding="SAME"),
        ]
    )

    # Model: "sequential_1"
    # _____________________________________________________________
    # Layer (type)                 Output Shape          Param #
    # =============================================================
    # dense_1 (Dense)              (None, 1568)          79968
    # _____________________________________________________________
    # reshape (Reshape)            (None, 7, 7, 32)      0
    # _____________________________________________________________
    # conv2d_transpose (Conv2DTran (None, 14, 14, 64)    18496
    # _____________________________________________________________
    # conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)    18464
    # _____________________________________________________________
    # conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)     289
    # =============================================================
    # Total params: 117,217
    # Trainable params: 117,217
    # Non-trainable params: 0
    # _____________________________________________________________
    self.generative_net.summary()

  @tf.function
  def sample(self, eps=None):
    # eps (16, 50) - 16 samples
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    # mean, logva (100, 50)
    mean, logvar = tf.split(self.inference_net(x),
                            num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    # eps (100, 50)
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    # z (16, 50), logits (16, 28, 28, 1)
    logits = self.generative_net(z)
    if apply_sigmoid:
      # probs (16, 28, 28, 1)
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

# log pdf for normal distribution
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar)
             + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  # mean, logvar, z (100, 50)
  # x_logit (100, 28, 28, 1)
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  # cross_ent (100, 28, 28, 1), logpx_z, logpz, logqz_x (100,)
  cross_ent = \
      tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
     #  loss ()
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 100
latent_dim = 50
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
# random_vector_for_generation (16, 50)
random_vector_for_generation = tf.random.normal(
           shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_input):
  # predictions (16, 28, 28, 1)
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  # Plot all the images in the predictions
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Plot predictions before training
generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                           elbo,
                                           end_time - start_time))
    generate_and_save_images(
        model, epoch, random_vector_for_generation)

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

plt.imshow(display_image(epochs))
plt.axis('off')# Display images

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info >= (6,2,0,''):
  display.Image(filename=anim_file)


