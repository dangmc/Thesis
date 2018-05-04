import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev=0.1)
    return tf.Variable(initial)

# Init bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# L2 regularization
def get_l2(weights):
    return sum(map(tf.nn.l2_loss, weights))

# one hot encoding
def one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]

def model(input_sz, output_sz):
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[input_sz], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y_ = tf.placeholder('float', shape=[None, output_sz])
    weight_decay = tf.placeholder(tf.float32)

    w_1 = weight_variable([257, 30])
    b_1 = bias_variable([30])

    w_2 = weight_variable([30, 10])
    b_2 = bias_variable([10])

    weights = [w_1, w_2]

    a_1 = tf.add(tf.matmul(x, w_1), b_1, 'a1')
    z_1 = tf.sigmoid(a_1)

    a_2 = tf.matmul(z_1, w_2) + b_2
    z_2 = tf.sigmoid(a_2)

    y = z_2

    params = dict()
    params['in_name'] = serialized_tf_example
    params['in'] = x
    params['weight_decay'] = weight_decay
    params['logits'] = y
    params['labels'] = y_
    params['l2_weights'] = get_l2(weights)

    return params