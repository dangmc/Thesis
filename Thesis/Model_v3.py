import tensorflow as tf
import numpy as np
import math

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Resnet_v2:
    def __init__(self, _num_blocks_layers, _num_blocks, _num_classes, _input_sz):

        # TO DO
        self.num_blocks_layers = _num_blocks_layers
        self.num_blocks = _num_blocks
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

    def arcface_loss(self, embedding, labels, out_num, w_init=None, s=64., m=0.1):
        '''
        :param embedding: the input embedding vectors
        :param labels:  the input labels, the shape should be eg: (batch_size, 1)
        :param s: scalar value default is 64
        :param out_num: output class num
        :param m: the margin value, default is 0.5
        :return: the final cacualted output, this output is send into the tf.nn.softmax directly
        '''
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m  # issue 1
        threshold = math.cos(math.pi - m)
        with tf.variable_scope('arcface_loss'):
            # inputs and weights norm
            embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
            embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
            weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                      initializer=w_init, dtype=tf.float32)
            weights_norm = tf.norm(weights, axis=0, keep_dims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')
            # cos(theta+m)
            cos_t = tf.matmul(embedding, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s * (cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            # mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            mask = labels
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return output, cos_t

    def build_model(self, filters_init, strides_layers, kernel_size, strides, pool_size, his_sz):
        # TO DO
        # image
        serialized_tf_image = tf.placeholder(tf.string, name='image')
        feature_image = {'x': tf.FixedLenFeature(shape=self.input_sz, dtype=tf.float32), }
        tf_image = tf.parse_example(serialized_tf_image, feature_image)
        inputs = tf.identity(tf_image['x'], name='x')
        copy = inputs

        # one gram feature
        # serialized_tf_gram = tf.placeholder(tf.string, name='one_gram')
        # feature_gram = {'y': tf.FixedLenFeature(shape=[255], dtype=tf.float32), }
        # tf_gram = tf.parse_example(serialized_tf_gram, feature_gram)
        # one_gram = tf.identity(tf_gram['y'], name='y')

        # dll feature
        serialized_tf_histogram = tf.placeholder(tf.string, name='dll')
        feature_histogram = {'z': tf.FixedLenFeature(shape=[his_sz], dtype=tf.float32), }
        tf_histogram = tf.parse_example(serialized_tf_histogram, feature_histogram)
        histogram = tf.identity(tf_histogram['z'], name='z')


        y_ = tf.placeholder('float', shape=[None, self.num_classes])

        training = tf.placeholder(dtype=tf.bool, name='training')

        weight_decay = tf.placeholder(tf.float32)

        inputs = self.convolution(inputs=copy, filters=filters_init/2, strides=strides,
                                  kernel_size=kernel_size)
        tf.identity(inputs, "initial_convolution")

        # inputs = tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=strides, padding="SAME")
        # tf.identity(inputs, "initial_max-pooling")

        for i in range(self.num_blocks_layers):
            filters = filters_init * (2 ** i)
            inputs = self.block_layer(inputs, num_blocks=self.num_blocks[i], training=training, filters=filters,
                                      strides=strides_layers[i])

        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True)
        tf.identity(inputs, "final_reduce_mean")

        dense_units = inputs.shape[3]
        inputs = tf.reshape(inputs, shape=[-1, dense_units])


        # dense layer
        inputs = tf.concat([inputs, histogram], 1)

        hidden = tf.layers.dense(inputs=inputs, units=256)
        hidden_acc = tf.nn.relu(hidden)
        outputs = tf.layers.dense(inputs=hidden_acc, units=self.num_classes)
        logits = outputs

        softmax = tf.nn.softmax(logits)
        tf.identity(softmax, "softmax_value")

        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])



        params = {}
        params['in_name'] = serialized_tf_image
        params['in'] = copy
        # params['in_name_gram'] = serialized_tf_gram
        # params['in_gram'] = one_gram
        params['in_name_dll'] = serialized_tf_histogram
        params['in_histogram'] = histogram
        params['logits'] = outputs
        params['labels'] = y_
        params['l2_weights'] = l2_loss
        params['training'] = training
        params['weight_decay'] = weight_decay
        params['softmax'] = softmax
        return params
