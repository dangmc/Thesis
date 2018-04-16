import tensorflow as tf
from dataset import read_malware_dataset
import Model_v1, sys


tf.app.flags.DEFINE_integer('training_iteration', 500000, 'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_integer('model_ver', 1, 'model version.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.5, 'weights decay.')

FLAGS = tf.app.flags.FLAGS

input_size = 257
num_labels = 10
batch_size = 64

# training_iteration = 100000
# work_dir = '/tmp'
# model_ver = 1
# learning_rate = 0.005
# weight_decay = 0.01



def main(_):


    # build model
    print('-'*20)
    print("build model")
    # work_dir = sys.argv[-1]

    dataset = read_malware_dataset(FLAGS.work_dir, True, num_labels)

    sess = tf.InteractiveSession()
    params = Model_v1.model(input_size, num_labels)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=params['logits'], labels=params['labels']))

    regularization = params['l2_weights'] * params['weight_decay']
    cost_function = cross_entropy
    train_step = tf.train.AdamOptimizer().minimize(cost_function)

    sess.run(tf.global_variables_initializer())

    values, indices = tf.nn.top_k(tf.nn.softmax(params['logits']), num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(params['logits']), 1), tf.argmax(params['labels'], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # Training model
    print('-'*20)
    print("training model")
    for iter in range(FLAGS.training_iteration):
        batch = dataset.train.next_batch(batch_size)
        train_step.run(
            feed_dict={params['in']: batch[0], params['labels']: batch[1], params['weight_decay']: FLAGS.weight_decay})

        if dataset.train.is_new_epoch() == True and dataset.train.get_epochs_completed() % 50 == 0:
            print('-' * 20)
            print('iteration = %d, epoch = %d' % (iter, dataset.train.get_epochs_completed()))
            print('cost function %g' % sess.run(
                cross_entropy, feed_dict={
                    params['in']: batch[0],
                    params['labels']: batch[1],
                    params['weight_decay']: FLAGS.weight_decay
                }))
            print('training accuracy %g' % sess.run(
                accuracy, feed_dict={
                    params['in']: batch[0],
                    params['labels']: batch[1],
                    params['weight_decay']: FLAGS.weight_decay
                }))

            if dataset.train.get_epochs_completed() % 100 == 0:
                print('-' * 20)
                print('test accuracy %g' % sess.run(
                    accuracy, feed_dict={
                        params['in']: dataset.test.gram,
                        params['labels']: dataset.test.labels,
                        params['weight_decay']: FLAGS.weight_decay
                    }))
    print('-'*20)
    print('test accuracy %g' % sess.run(
        accuracy, feed_dict={
            params['in']: dataset.test.gram,
            params['labels']: dataset.test.labels,
            params['weight_decay']: FLAGS.weight_decay
        }))

    print('Done training!')



    # Export model
    # export_path_base = sys.argv[-1]
    # export_path = os.path.join(
    #     tf.compat.as_bytes(export_path_base),
    #     tf.compat.as_bytes(str(FLAGS.model_version)))
    # print('Exporting trained model to', export_path)
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #
    # # Build the signature_def_map.
    # classification_inputs = tf.saved_model.utils.build_tensor_info(
    #     params['in_name'])
    # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
    #     prediction_classes)
    # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)
    #
    # classification_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={
    #             tf.saved_model.signature_constants.CLASSIFY_INPUTS:
    #                 classification_inputs
    #         },
    #         outputs={
    #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
    #                 classification_outputs_classes,
    #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
    #                 classification_outputs_scores
    #         },
    #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
    #
    # tensor_info_x = tf.saved_model.utils.build_tensor_info(params['in'])
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(params['logits'])
    #
    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'images': tensor_info_x},
    #         outputs={'scores': tensor_info_y},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'predict_images':
    #             prediction_signature,
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #             classification_signature,
    #     },
    #     legacy_init_op=legacy_init_op)
    #
    # builder.save()
    #
    # print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
