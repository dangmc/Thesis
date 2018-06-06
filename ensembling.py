import numpy as np
import tensorflow as tf
import Model_v1, Model_v2, Model_v3
import dataset
import sklearn as sk

tf.app.flags.DEFINE_integer('training_iteration', 40000, 'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('path_malware_his', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign_his', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_string('path_malware_api', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign_api', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_string('path_malware_img', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign_img', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_integer('model_version', 1, 'model version.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.005, 'weights decay.')
tf.app.flags.DEFINE_string('checkpoint_all_dir', '', "check point directory")
tf.app.flags.DEFINE_string('checkpoint_api_dir', '', "check point directory")
tf.app.flags.DEFINE_string('checkpoint_his_dir', '', "check point directory")
tf.app.flags.DEFINE_string('summaries_dir', '/tmp', "summaries directory")
tf.app.flags.DEFINE_integer('input_size', 64, "input size")
tf.app.flags.DEFINE_integer('learning_rate_decay_epoch', 50, 'epoch when decay learning rate')

FLAGS = tf.app.flags.FLAGS
def model_histogram(input_size, n_models, data, checkpoint_dir):
    num_labels = 2
    batch_size = 64

    tf.reset_default_graph()

    #data = dataset.load_test_data_histogram(FLAGS.path_malware, FLAGS.path_benign, one_hot_encode=True, sz=FLAGS.input_size, num_labels=2)


    params = Model_v1.model(input_sz=input_size, output_sz=num_labels, layers=[256, 256, 2])


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    sess = tf.InteractiveSession(config=config)
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy + regularization
    tf.summary.scalar("loss", cost_function)

    # print ("number training instance = %d" % (n_ins))
    # global_steps = tf.train.get_or_create_global_step()
    # lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[10, 50, 100], global_steps=global_steps, n_ins=n_ins)
    step_decay = int(10 * FLAGS.learning_rate_decay_epoch / batch_size)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               step_decay, 0.5, staircase=True)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads = opt.compute_gradients(cost_function)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(grads, global_step=global_step)

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(params['softmax'], 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ensemble = np.zeros(shape=(data.test.labels.shape[0], 2))
    saver = tf.train.Saver()
    for k in range(n_models):
        model_path = checkpoint_dir + str(k + 1)
        with tf.Session() as sess:
            saver.restore(sess, model_path + '/' + 'model')
            pred = sess.run(
                    params['softmax'], feed_dict={
                        params['in']: data.test.gram,
                        params['labels']: data.test.labels,
                        params['weight_decay']: 0.005,
                        params['training']: False
                    })
            # print(pred)
            ensemble += pred
            sess.close()

    ensemble = ensemble / n_models
    #ensemble = np.argmax(ensemble, 1)
    labels = np.argmax(data.test.labels, 1)
    return ensemble, labels

def model_ngram(input_size, n_models, data, checkpoint_dir):
    num_labels = 2
    batch_size = 64

    tf.reset_default_graph()

    #data = dataset.load_test_data_histogram(FLAGS.path_malware, FLAGS.path_benign, one_hot_encode=True, sz=FLAGS.input_size, num_labels=2)


    params = Model_v1.model(input_sz=input_size, output_sz=num_labels, layers=[256, 256, 2])


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    sess = tf.InteractiveSession(config=config)
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy + regularization
    tf.summary.scalar("loss", cost_function)

    # print ("number training instance = %d" % (n_ins))
    # global_steps = tf.train.get_or_create_global_step()
    # lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[10, 50, 100], global_steps=global_steps, n_ins=n_ins)
    step_decay = int(10 * FLAGS.learning_rate_decay_epoch / batch_size)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               step_decay, 0.5, staircase=True)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads = opt.compute_gradients(cost_function)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(grads, global_step=global_step)

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(params['softmax'], 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ensemble = np.zeros(shape=(data.test.labels.shape[0], 2))
    saver = tf.train.Saver()
    for k in range(n_models):
        model_path = checkpoint_dir + str(k + 1)
        with tf.Session() as sess:
            saver.restore(sess, model_path + '/' + 'model')
            pred = sess.run(
                    params['softmax'], feed_dict={
                        params['in']: data.test.gram,
                        params['labels']: data.test.labels,
                        params['weight_decay']: 0.005,
                        params['training']: False
                    })
            # print(pred)
            ensemble += pred
            sess.close()

    ensemble = ensemble / n_models
    #ensemble = np.argmax(ensemble, 1)
    labels = np.argmax(data.test.labels, 1)
    return ensemble, labels

def model_api(input_size, n_models, data, checkpoint_dir):
    num_labels = 2
    batch_size = 64

    tf.reset_default_graph()

    #data = dataset.load_dll_data(FLAGS.path_malware, FLAGS.path_benign, one_hot_encode=True,
    #                                        sz=input_size, num_labels=2, is_train=False)

    params = Model_v1.model(input_sz=input_size, output_sz=num_labels, layers=[256, 256, 2])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    sess = tf.InteractiveSession(config=config)
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy + regularization
    tf.summary.scalar("loss", cost_function)

    # print ("number training instance = %d" % (n_ins))
    # global_steps = tf.train.get_or_create_global_step()
    # lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[10, 50, 100], global_steps=global_steps, n_ins=n_ins)
    step_decay = int(10 * FLAGS.learning_rate_decay_epoch / batch_size)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               step_decay, 0.5, staircase=True)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads = opt.compute_gradients(cost_function)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(grads, global_step=global_step)

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(params['softmax'], 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ensemble = np.zeros(shape=(data.test.labels.shape[0], 2))
    saver = tf.train.Saver()
    for k in range(n_models):
        model_path = checkpoint_dir + str(k + 1)
        with tf.Session() as sess:
            saver.restore(sess, model_path + '/' + 'model')
            pred = sess.run(
                params['softmax'], feed_dict={
                    params['in']: data.test.gram,
                    params['labels']: data.test.labels,
                    params['weight_decay']: 0.005,
                    params['training']: False
                })
            # print(pred)
            ensemble += pred
            sess.close()

    ensemble = ensemble / n_models
    #ensemble = np.argmax(ensemble, 1)
    labels = np.argmax(data.test.labels, 1)
    return ensemble, labels

def model_img(data, n_models, checkpoint_dir):
    input_size =[64, 64, 1]
    num_labels = 2
    batch_size = 64
    tf.reset_default_graph()

    model = Model_v2.Resnet_v2(3, [2, 2, 2], num_labels, input_size)
    params = model.build_model(filters_init=32, strides_layers=[2, 2, 2], kernel_size=3, pool_size=3, strides=2)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    sess = tf.InteractiveSession(config=config)
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy + regularization
    tf.summary.scalar("loss", cost_function)

    # print ("number training instance = %d" % (n_ins))
    # global_steps = tf.train.get_or_create_global_step()
    # lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[10, 50, 100], global_steps=global_steps, n_ins=n_ins)
    step_decay = int(10 * FLAGS.learning_rate_decay_epoch / batch_size)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               step_decay, 0.5, staircase=True)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads = opt.compute_gradients(cost_function)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(grads, global_step=global_step)

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(params['softmax'], 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ensemble = np.zeros(shape=(data.test.labels.shape[0], 2))
    saver = tf.train.Saver()
    for k in range(n_models):
        model_path = checkpoint_dir + str(k + 1)
        with tf.Session() as sess:
            saver.restore(sess, model_path + '/' + 'model')
            pred = sess.run(
                params['softmax'], feed_dict={
                    params['in']: data.test.gram,
                    params['labels']: data.test.labels,
                    params['weight_decay']: 0.005,
                    params['training']: False
                })
            ensemble += pred
            sess.close()

    ensemble = ensemble / n_models
    #ensemble = np.argmax(ensemble, 1)
    labels = np.argmax(data.test.labels, 1)
    return ensemble, labels

def model_all(data_img, data_his, data_api, checkpoint_dir, n_models):
    input_size = [64, 64, 1]
    num_labels = 2
    batch_size = 64
    tf.reset_default_graph()

    # data = dataset.load_histogram_image(path_malware_his=FLAGS.path_malware_his,
    #                                        path_malware_img=FLAGS.path_malware_img,
    #                                        path_benign_his=FLAGS.path_benign_his, path_benign_img=FLAGS.path_benign_img,
    #                                        one_hot_encode=True)

    model = Model_v3.Resnet_v2(3, [2, 2, 2], num_labels, input_size)
    params = model.build_model(filters_init=32, strides_layers=[2, 2, 2], kernel_size=3, pool_size=3, strides=2, his_sz=256+794)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    sess = tf.InteractiveSession(config=config)
    cross_entropy = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy + regularization
    tf.summary.scalar("loss", cost_function)

    # print ("number training instance = %d" % (n_ins))
    # global_steps = tf.train.get_or_create_global_step()
    # lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[10, 50, 100], global_steps=global_steps, n_ins=n_ins)
    step_decay = int(10 * FLAGS.learning_rate_decay_epoch / batch_size)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               step_decay, 0.5, staircase=True)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads = opt.compute_gradients(cost_function)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.apply_gradients(grads, global_step=global_step)

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(params['softmax'], 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ensemble = np.zeros(shape=(data_api.test.labels.shape[0], 2))
    saver = tf.train.Saver()
    for k in range(n_models):
        model_path = checkpoint_dir + str(k + 1)
        with tf.Session() as sess:
            saver.restore(sess, model_path + '/' + 'model')
            pred = sess.run(
                params['softmax'], feed_dict={
                    params['in']: data_img.test.gram,
                    params['labels']: data_img.test.labels,
                    params['weight_decay']: 0.005,
                    params['training']: False,
                    params['in_histogram']: np.concatenate((data_his.test.gram, data_api.test.gram), axis=1)
                })

            ensemble += pred
            sess.close()

    ensemble = ensemble / n_models
    #ensemble = np.argmax(ensemble, 1)

    labels = np.argmax(data_api.test.labels, 1)
    return ensemble, labels

# print(ensemble)
# print(labels)
#print("confusion matrix")
# print(sk.metrics.confusion_matrix(labels, ensemble))
def predict():
    data_img, data_his, data_api = dataset.read_all_feature_malware_benign(path_malware_img=FLAGS.path_malware_img, path_malware_his=FLAGS.path_malware_his, path_malware_dll=FLAGS.path_malware_api,
                                                                           path_benign_img=FLAGS.path_benign_img, path_benign_his=FLAGS.path_benign_his, path_benign_dll=FLAGS.path_benign_api)



    print('*'*20)
    print("model entropy histogram")
    ensemble_his, labels = model_histogram(input_size=256, n_models=6, data=data_his, checkpoint_dir=FLAGS.checkpoint_his_dir)
    print(sk.metrics.confusion_matrix(labels, ensemble_his))

    print('*'*20)
    print("model api")
    ensemble_api, labels = model_api(input_size=794, n_models=6, data=data_api, checkpoint_dir=FLAGS.checkpoint_api_dir)
    print(sk.metrics.confusion_matrix(labels, ensemble_api))

    print('*'*20)
    print("model all")
    ensemble_all, labels = model_all(data_img, data_his, data_api, checkpoint_dir=FLAGS.checkpoint_all_dir, n_models=6)
    #print("confusion matrix")
    print(sk.metrics.confusion_matrix(labels, ensemble_all))