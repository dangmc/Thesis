import csv
from Model_v2 import Resnet_v2
import tensorflow as tf
import dataset
from cross_validation import CrossValidationFolds
import sys, os

tf.app.flags.DEFINE_integer('training_iteration', 40000, 'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('path_malware', '/tmp', 'Working directory of malware.')
tf.app.flags.DEFINE_string('path_benign', '/tmp', 'Working directory of benign.')
tf.app.flags.DEFINE_integer('model_version', 1, 'model version.')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.005, 'weights decay.')
tf.app.flags.DEFINE_string('summaries_dir', '/tmp', "summaries directory")
tf.app.flags.DEFINE_integer('input_size', 64, "input size" )

FOLDS = 10

FLAGS = tf.app.flags.FLAGS
MNIST = "D:/Data/mnist/mnist_train.csv"


def show(name, scalar, sess, feed_dict):
    print(name + " = %s" % (sess.run(scalar, feed_dict=feed_dict)))


def export_model(export_path_base, version, sess, input_name, input, output_softmax, output_score):
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(version))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        input_name)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(output_score)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(input)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(output_softmax)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    return builder

def learning_rate_schedule(lr_init, decay_rates, boundary_epochs, n_ins, batch_sz, global_steps):
    ins_per_epoch = n_ins / batch_sz
    boundaries = [ins_per_epoch * iter for iter in boundary_epochs]
    vals = [lr_init * decay for decay in decay_rates]
    return tf.train.piecewise_constant(global_steps, boundaries=boundaries, values=vals)


# build model
print('-' * 20)
print("build model")

# dataset = dataset.mnist(MNIST, one_hot_encode=True)

dataset = dataset.load_data(FLAGS.path_malware, FLAGS.path_benign, one_hot_encode=True, sz=FLAGS.input_size)
cross_validation = CrossValidationFolds(dataset.train.get_gram(), dataset.train.get_labels(), FOLDS)

input_size = [FLAGS.input_size, FLAGS.input_size, 1]
num_labels = 2
batch_size = 64

model = Resnet_v2(3, [2, 2, 2], num_labels, input_size)
params = model.build_model(filters_init=32, strides_layers=[2, 2, 2], kernel_size=3, pool_size=3, strides=2)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7

sess = tf.InteractiveSession(config=config)
cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=params['logits'], onehot_labels=params['labels']))

regularization = params['l2_weights'] * params['weight_decay']
cost_function = cross_entropy
tf.summary.scalar("loss", cost_function)


n_ins = int (cross_validation.labels.shape[0] / FOLDS)
global_steps = tf.train.get_or_create_global_step()
lr = learning_rate_schedule(lr_init=FLAGS.learning_rate, decay_rates=[1, 0.1, 0.01, 0.001], batch_sz=batch_size, boundary_epochs=[100, 150, 200], global_steps=global_steps, n_ins=n_ins)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost_function)

values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
    tf.constant([str(i) for i in range(num_labels)]))
prediction_classes = table.lookup(tf.to_int64(indices))
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(params['logits']), 1), tf.argmax(params['labels'], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

# export model
sess.run(tf.global_variables_initializer())
builder = export_model(export_path_base=sys.argv[-1], version=str(FLAGS.model_version), input_name=params['in_name'],
                       input=params['in'], output_score=values, output_softmax=params['softmax'], sess=sess)

# Training model
cv_accuracy = 0
best_accuracy = 0
for k in range(FOLDS):
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + str(k + 1) + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + str(k + 1) + '/validation')
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + str(k + 1) + '/test')
    sess.run(tf.global_variables_initializer())
    print("FOLDS = %d" % (k + 1))
    print('-' * 20)
    print("training model")
    data = cross_validation.split()
    best_accuracy_fold = 0
    for iter in range(FLAGS.training_iteration):
        # print('iteration = %d' % (iter))
        batch = data.train.next_batch(batch_size)
        train_step.run(
            feed_dict={params['in']: batch[0], params['labels']: batch[1], params['weight_decay']: FLAGS.weight_decay,
                       params['training']: True})

        if data.train.is_new_epoch() == True:
            print('-' * 20)
            print('Folds = %d, iteration = %d, epoch = %d' % (k + 1, iter, data.train.get_epochs_completed()))

            loss_train, acc_train, summary = sess.run(
                [cost_function, accuracy, merged], feed_dict={
                    params['in']: batch[0],
                    params['labels']: batch[1],
                    params['weight_decay']: FLAGS.weight_decay,
                    params['training']: False
                })
            print('cost function %g' % loss_train)
            print("training's accuracy %g" % acc_train)
            train_writer.add_summary(summary, iter)

            if data.train.get_epochs_completed() % 5 == 0:
                acc_valid, summary = sess.run(
                    [accuracy, merged], feed_dict={
                        params['in']: data.validation.gram,
                        params['labels']: data.validation.labels,
                        params['weight_decay']: FLAGS.weight_decay,
                        params['training']: False
                    })
                print('-' * 20)
                print("validation's accuracy %g" % acc_valid)
                valid_writer.add_summary(summary, iter)
                if best_accuracy_fold < acc_valid:
                    best_accuracy_fold = acc_valid

                acc_test, summary = sess.run(
                    [accuracy, merged], feed_dict={
                        params['in']: dataset.test.gram,
                        params['labels']: dataset.test.labels,
                        params['weight_decay']: FLAGS.weight_decay,
                        params['training']: False
                    })
                print('-' * 20)
                print('test accuracy %g' % acc_test)
                test_writer.add_summary(summary, iter)
    print("best accuracy of fold %d = %g" % (k+1, best_accuracy_fold))
    cv_accuracy += best_accuracy_fold
    if best_accuracy < best_accuracy_fold:
        best_accuracy = best_accuracy_fold
        builder.save()
        print("model updated")

print('Done training!')
print("accuracy = %g" % (cv_accuracy / FOLDS))

# Export model


print('Done exporting!')