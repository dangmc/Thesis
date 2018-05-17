import numpy as np
import tensorflow as tf
from Model_v3 import Resnet_v2
import dataset
import sklearn as sk

tf.app.flags.DEFINE_integer('training_iteration', 40000, 'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('path_malware_his', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign_his', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_string('path_malware_img', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign_img', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_string('path_real', '/tmp', 'Working directory of real data.')
tf.app.flags.DEFINE_integer('model_version', 1, 'model version.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.005, 'weights decay.')
tf.app.flags.DEFINE_string('checkpoint_dir', '', "check point directory")
tf.app.flags.DEFINE_string('summaries_dir', '/tmp', "summaries directory")
tf.app.flags.DEFINE_integer('input_size', 64, "input size")
tf.app.flags.DEFINE_integer('learning_rate_decay_epoch', 50, 'epoch when decay learning rate')

input_size = [64, 64, 1]
num_labels = 2
batch_size = 64
FLAGS = tf.app.flags.FLAGS

tf.reset_default_graph()

dataset = dataset.load_histogram_image(path_malware_his=FLAGS.path_malware_his, path_malware_img=FLAGS.path_malware_img,
                                       path_benign_his=FLAGS.path_benign_his, path_benign_img=FLAGS.path_benign_img,
                                       one_hot_encode=True)

model = Resnet_v2(3, [2, 2, 2], num_labels, input_size)
params = model.build_model(filters_init=32, strides_layers=[2, 2, 2], kernel_size=3, pool_size=3, strides=2)

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

ensemble_set1 = np.zeros(shape=(dataset.train.labels.shape[0], 2))
ensemble_set2 =  np.zeros(shape=(dataset.test.labels.shape[0],2))
saver = tf.train.Saver()
n_models = 2
for k in range(n_models):
    model_path = FLAGS.checkpoint_dir + str(k + 1)
    with tf.Session() as sess:
        saver.restore(sess, model_path + '/' + 'model')
        pred_set1 = sess.run(
            params['softmax'], feed_dict={
                params['in']: dataset.train.img,
                params['labels']: dataset.train.labels,
                params['weight_decay']: 0.005,
                params['training']: False,
                params['in_histogram']: dataset.train.his,
            })

        pred_set2 = sess.run(
            params['softmax'], feed_dict={
                params['in']: dataset.test.img,
                params['labels']: dataset.test.labels,
                params['weight_decay']: 0.005,
                params['training']: False,
                params['in_histogram']: dataset.test.his,
            })

        pred = tf.concat([pred_set1, pred_set2], axis=0)

        ensemble_set1 += pred_set1
        ensemble_set2 += pred_set2
        sess.close()

ensemble_set1 = ensemble_set1 / n_models
ensemble_set2 = ensemble_set2 / n_models

ensemble_set1 = np.argmax(ensemble_set1, 1)
ensemble_set2 = np.argmax(ensemble_set2, 1)

labels_set1 = np.argmax(dataset.train.labels, 1)
labels_set2 = np.argmax(dataset.test.labels, 1)

print("confusion matrix")
print(sk.metrics.confusion_matrix(labels_set1, ensemble_set1))
print(sk.metrics.confusion_matrix(labels_set2, ensemble_set2))
# print("F1 score")
# print(sk.metrics.f1_score(labels, ensemble))
