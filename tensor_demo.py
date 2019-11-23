import tensorflow as tf

# tf.Tensor(3, shape=(), dtype=int32)
# tf.Tensor([4 6], shape=(2,), dtype=int32)
# tf.Tensor(25, shape=(), dtype=int32)
# tf.Tensor(6, shape=(), dtype=int32)
# tf.Tensor(13, shape=(), dtype=int32)

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

# tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
# (1, 2)
# <dtype: 'int32'>
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)