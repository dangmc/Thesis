import numpy as np
import tensorflow as tf
import Model_v1
import dataset
import sklearn as sk

tf.app.flags.DEFINE_integer('training_iteration', 40000, 'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('path_malware', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_string('path_real', '/tmp', 'Working directory of real data.')
tf.app.flags.DEFINE_integer('model_version', 1, 'model version.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.005, 'weights decay.')
tf.app.flags.DEFINE_string('checkpoint_dir', '', "check point directory")
tf.app.flags.DEFINE_string('summaries_dir', '/tmp', "summaries directory")
tf.app.flags.DEFINE_integer('input_size', 64, "input size")
tf.app.flags.DEFINE_integer('learning_rate_decay_epoch', 50, 'epoch when decay learning rate')

input_size = 256
num_labels = 2
batch_size = 64
FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()

dataset = dataset.load_test_data_histogram(FLAGS.path_malware, FLAGS.path_benign, one_hot_encode=True, sz=FLAGS.input_size, num_labels=2)


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

ensemble = np.zeros(shape=(dataset.real.labels.shape[0], 2))
saver = tf.train.Saver()
n_models = 1
for k in range(n_models):
    model_path = FLAGS.checkpoint_dir + str(k + 1)
    with tf.Session() as sess:
        saver.restore(sess, model_path + '/' + 'model')
        pred = sess.run(
                params['softmax'], feed_dict={
                    params['in']: dataset.real.gram,
                    params['labels']: dataset.real.labels,
                    params['weight_decay']: 0.005
                })
        # print(pred)
        ensemble += pred
        sess.close()

ensemble = ensemble / n_models
ensemble = np.argmax(ensemble, 1)
labels = np.argmax(dataset.real.labels, 1)

# print(ensemble)
# print(labels)
print("confusion matrix")
print(sk.metrics.confusion_matrix(labels, ensemble))

print("F1 score")
print(sk.metrics.f1_score(labels, ensemble))