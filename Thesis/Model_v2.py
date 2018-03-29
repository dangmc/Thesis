import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Resnet_basic:
    def __init__(self):
        # TO DO
        return 0

    def fixed_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_berfore = pad_total // 2
        pad_after = pad_total - pad_berfore

        inputs_pad = tf.pad(inputs, [[0, 0], [0, 0], [pad_berfore, pad_after], [pad_berfore, pad_after]])
        return inputs_pad

    def batch_norm(self, inputs, training):
        # TO DO

        return tf.layers.batch_normalization(inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                             scale=True, fused=True, axis=3, training=training)

    def convolution(self, inputs, filters, kernel_size, strides):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=('SAME' if strides == 1 else 'VALID'),
                                kernel_initializer=tf.variance_scaling_initializer(), use_bias=False)

    def build_block(self):
        # TO DO
        return 0

    def build_model(self):
        # TO DO
        return 0
