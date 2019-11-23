import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Convert a word in a 1K vocabulary to a 5-D vector.
embedding_layer = layers.Embedding(1000, 5)

# Create the word embedding for word with index 1, 2 & 3
# [[-0.0004058   0.02741    -0.00751717  0.03805766 -0.02493097]
#  [-0.01030381 -0.00866082  0.04199603 -0.04461098 -0.02717341]
#  [-0.02098732  0.00564277  0.02641683  0.0214516  -0.02245522]]
result = embedding_layer(tf.constant([1,2,3]))
print(result.numpy())

# (2, 3, 5)
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
print(result.shape)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
# ['the_', ', ', '. ', 'a_', 'and_',
# 'of_', 'to_', 's_', 'is_', 'br', 'in_', 'I_', 'that_',
# 'this_', 'it_', ' /><', ' />', 'was_', 'The_', 'as_']
print(encoder.subwords[:20])

padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10,
                        padded_shapes = padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10,
                        padded_shapes = padded_shapes)

train_batch, train_labels = next(iter(train_batches))
# [[2016  481  105 ...    0    0    0]
#  [7963   19 4558 ...   45 3962 7975]
#  [ 173    9   84 ...    0    0    0]
#  ...
#  [  19 2534  209 ...    0    0    0]
#  [  62   32    9 ...    0    0    0]
#  [ 324   12  118 ...    0    0    0]]
print(train_batch.numpy())

embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(1, activation='sigmoid')
])

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 16)          130960
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

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()

e = model.layers[0]
weights = e.get_weights()[0]
# (8185, 16)
print(weights.shape)

encoder = info.features['text'].encoder

import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
  vec = weights[num+1] # skip 0, it's padding.
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

