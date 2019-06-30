import tensorflow as tf

feature_bias = tf.variable(tf.random_uniform([feature_size, 1],
                                             ))