import tensorflow as tf
import numpy as np

class AutoEncoder:
    def __init__(self, _n_instances, _n_features, _n_components, _denoising=False):
        self.n_instances = _n_instances
        self.n_features = _n_features
        self.n_components = _n_components
        self.denoising = _denoising

    def weight_variable(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def bias_variable(self, shape):
        bias = tf.Constant(shape, 1.0)
        return tf.Variable(bias)

    def encode(self, X, weight, bias):
        return tf.matmul(X, weight) + bias

    def decode(self, X, weight, bias):
        return tf.matmul(X, weight, bias)


block = np.array([0, 64, 255, 32, 78])
c = np.bincount(block >> 4, minlength=16)
print (c)