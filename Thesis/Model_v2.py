import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Resnet_v2:
    def __init__(self, _num_blocks, _dense_units, _num_classes, _input_sz):

        # TO DO
        self.num_blocks = _num_blocks
        self.dense_units = _dense_units
        self.num_classes = _num_classes
        self.input_sz = _input_sz

    def fixed_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_berfore = pad_total // 2
        pad_after = pad_total - pad_berfore

        inputs_pad = tf.pad(inputs, [[0, 0], [pad_berfore, pad_after], [pad_berfore, pad_after], [0, 0]])
        return inputs_pad

    def batch_norm(self, inputs, training):
        return tf.layers.batch_normalization(inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                             scale=True, fused=True, axis=3, training=training)

    def convolution(self, inputs, filters, kernel_size, strides):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=('SAME' if strides == 1 else 'VALID'),
                                kernel_initializer=tf.variance_scaling_initializer(), use_bias=False)

    def block_v2(self, inputs, filters, strides, training, projection):
        # TO DO
        # block: batch norm - relu - conv

        copy = inputs
        if projection is not None:
            copy = projection(inputs)

        btn_1 = self.batch_norm(inputs, training)
        relu_1 = tf.nn.relu(btn_1)
        conv_1 = self.convolution(inputs=relu_1, filters=filters, kernel_size=3, strides=strides)

        btn_2 = self.batch_norm(conv_1, training)
        relu_2 = tf.nn.relu(btn_2)
        conv_2 = self.convolution(inputs=relu_2, filters=filters, kernel_size=3, strides=1)

        return conv_2 + copy

    def block_layer(self, inputs, num_blocks, filters, strides, training):
        def projection(inputs):
            return self.convolution(inputs, filters=filters, strides=strides, kernel_size=1)

        inputs = self.block_v2(inputs, filters=filters, strides=strides, training=training, projection=projection)
        for _ in range(1, num_blocks):
            inputs = self.block_v2(inputs, filters=filters, strides=1, training=training, projection=None)

        return inputs

    def build_model(self, filters_init, strides_layers, kernel_size, strides, pool_size):
        # TO DO

        serialized_tf_example = tf.placeholder(tf.string, name='image')
        feature_configs = {'x': tf.FixedLenFeature(shape=[self.input_sz], dtype=tf.float32), }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        inputs = tf.identity(tf_example['x'], name='x')
        copy = inputs

        y_ = tf.placeholder('float', shape=[None, self.num_classes])

        training = tf.placeholder(dtype=tf.bool, name='training')

        weight_decay = tf.placeholder(tf.float32)

        inputs = self.convolution(inputs=copy, filters=filters_init, strides=strides, training=training,
                                  kernel_size=kernel_size)
        tf.identity(inputs, "initial convolution")

        inputs = tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=strides, padding="SAME")
        tf.identity(inputs, "initial max-pooling")

        for i in range(self.num_blocks):
            filters = filters_init * (2 ** i)
            inputs = self.block_layer(inputs, num_blocks=self.num_blocks[i], training=training, filters=filters,
                                      strides=strides_layers[i])

        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True)
        tf.identity(inputs, "final reduce mean")

        inputs = tf.reshape(inputs, shape=[-1, self.dense_units])
        outputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
        tf.identity(inputs, "final dense")

        l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        params = {}
        params['in_name'] = serialized_tf_example
        params['in'] = copy
        params['logits'] = outputs
        params['labels'] = y_
        params['l2_weights'] = l2_loss
        params['weight_decay'] = weight_decay
        return inputs
