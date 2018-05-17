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

def model(input_sz, output_sz, layers):
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[input_sz], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y_ = tf.placeholder('float', shape=[None, output_sz])
    weight_decay = tf.placeholder(tf.float32)
    dropout_rate = tf.placeholder(tf.float32)
    w = []
    b = []
    for iter in range(len(layers)):
        weight = weight_variable([input_sz, layers[iter]])
        bias = bias_variable([layers[iter]])
        w.append(weight)
        b.append(bias)
        input_sz = layers[iter]

    a = x

    # forward
    for iter in range(len(layers) - 1):
        a = tf.add(tf.matmul(a, w[iter]), b[iter])
        z = tf.nn.relu(a)
        z = tf.nn.dropout(z, keep_prob=dropout_rate)
        a = z
    a = tf.add(tf.matmul(a, w[len(layers)-1]), b[len(layers)-1])
    z = tf.nn.softmax(a)


    l2_weight = l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    params = dict()
    params['in_name'] = serialized_tf_example
    params['in'] = x
    params['weight_decay'] = weight_decay
    params['logits'] = a
    params['softmax'] = z
    params['labels'] = y_
    params['l2_weights'] = l2_weight
    params['dropout_rate'] = dropout_rate

    return params



    # w_1 = weight_variable([input_sz, 255])
    # b_1 = bias_variable([255])
    #
    # w_2 = weight_variable([255, 255])
    # b_2 = bias_variable([255])
    #
    # w_3 = weight_variable([255, 30])
    # b_3 = bias_variable([30])
    #
    # w_4 = weight_variable([30, 2])
    # b_4 = bias_variable([2])
    #
    # a_1 = tf.add(tf.matmul(x, w_1), b_1, 'a1')
    # z_1 = tf.nn.relu(a_1)
    #
    # a_2 = tf.matmul(z_1, w_2) + b_2
    # z_2 = tf.nn.relu(a_2)
    #
    # a_3 = tf.matmul(z_2, w_3) + b_3
    # z_3 = tf.nn.relu(a_3)
    #
    # a_4 = tf.matmul(z_3, w_4) + b_4
    # z_4 = tf.nn.softmax(a_4)