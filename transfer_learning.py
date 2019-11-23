import matplotlib.pylab as plt

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

# Load a pre-trained model (mobilenet)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)

# <PIL.Image.Image image mode=RGB size=224x224 at 0x132381E80>
print(grace_hopper)

# (224, 224, 3)
grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)

# (1, 1001)
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

# Label the image as "military uniform".
# 653
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Plot the image and its prediction
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()

# Load the TensorFlow flower dataset
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root),
                            target_size=IMAGE_SHAPE)

# Image batch shape:  (32, 224, 224, 3)
# Label batch shape:  (32, 5)
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

# Classify the flower dataset
# (32, 1001)
result_batch = classifier.predict(image_batch)
print(result_batch.shape)

# array(['daisy', 'daisy', 'daisy', 'plastic bag', 'bee', 'cauliflower',
#        'daisy', 'bee', 'cauliflower', 'vase', 'greenhouse', 'daisy',
#        'lemon', 'daisy', 'daisy', 'picket fence', 'daisy', 'bee', 'nail',
#        'daisy', 'bee', 'paper towel', 'hair slide', 'volcano', 'ant',
#        'pot', 'daisy', 'daisy', 'daisy', 'hip', 'brassiere', 'hip'],
#       dtype='<U30')
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

# Plot the predictions
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
plt.show()

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

# Load a headless model (a feature extractor without
# the fully-connected classification layers.
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

# Extract features. a.k.a. making a prediction
# with the model
# (32, 1280)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

feature_extractor_layer.trainable = False

# Add a fully-connected layer for classification.
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# keras_layer_1 (KerasLayer)   (None, 1280)              2257984
# _________________________________________________________________
# dense (Dense)                (None, 5)                 6405
# =================================================================
# Total params: 2,264,389
# Trainable params: 6,405
# Non-trainable params: 2,257,984
# _________________________________________________________________
model.summary()

predictions = model(image_batch)
# (32, 5)
print(predictions.shape)

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])


# To visualize the training progress,
# use a custom callback to log the loss and accuracy of
# each batch individually, instead of displaying
# the epoch average.
# Epoch 1/2
# 115/115 [==============================] - 703s 6s/step
# - loss: 0.6939 - acc: 0.9375
# Epoch 2/2
# 115/115 [==============================] - 557s 5s/step
# - loss: 0.3481 - acc: 0.9062
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()


steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])

# Plot the training statistics
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
plt.show()

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
plt.show()

class_names = sorted(image_data.class_indices.items(),
                     key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
# ['Daisy' 'Dandelion' 'Roses' 'Sunflowers' 'Tulips']
print(class_names)

# Make predictions with the new model.
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)

# Plot the result with red
# being incorrectly labeled.
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()

import time
t = time.time()

# Export the new model out.
# Load it back to make sure we get the same model.
export_path = "/tmp/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)
# /tmp/saved_models/1571776041
print(export_path)

reloaded = tf.keras.experimental.load_from_saved_model(
    export_path, custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
# 0.0 (They make the same predictions.)
print(abs(reloaded_result_batch - result_batch).max())





