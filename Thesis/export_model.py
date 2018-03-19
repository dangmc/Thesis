import tensorflow as tf
import malware_dataset
import Model_v1
import os, sys


tf.app.flags.DEFINE_integer('training_iteration', 1000, "number of training iterations")
tf.app.flags.DEFINE_string('work_dir', "/tmp", "Working directory")
tf.app.flags.DEFINE_integer('model_ver', 1, "model version")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate")

FLAGS = tf.app.flags.FLAGS

input_size = 257
num_labels = 10
batch_size = 64



def main(_):

    # Training model
    dataset = malware_dataset.read_malware_dataset(FLAGS.work_dir)

    sess = tf.InteractiveSession()
    x_name, x, y_, y = Model_v1.model(input_size, num_labels)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())

    values, indices = tf.nn.top_k(y, num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    for _ in range(FLAGS.training_iteration):
        batch = dataset.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print('training accuracy %g' % sess.run(
        accuracy, feed_dict={
            x: dataset.test.gram,
            y_: dataset.test.labels
        }))
    print('Done training!')



    # Export model
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        x_name)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

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

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

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

    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
