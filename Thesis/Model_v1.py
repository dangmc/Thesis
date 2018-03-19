import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev=0.1)
    return tf.Variable(initial)

# Init bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def model(input_sz, output_sz):
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[input_sz], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y_ = tf.placeholder('float', shape=[None, output_sz])

    w_1 = weight_variable([257, 64])
    b_1 = bias_variable([64])

    w_2 = weight_variable([64, 10])
    b_2 = bias_variable([10])

    h_1 = tf.add(tf.matmul(x, w_1), b_1, 'h1')
    y = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2, name='y')

    return serialized_tf_example, x, y_, y